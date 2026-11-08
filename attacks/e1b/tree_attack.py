#!/usr/bin/env python3
"""Admissible tree search for the E1b decision attack (per E1b-tree-attack.md).

Replaces "inject one wrong token, then free-decode via model.generate + audit"
with a search whose every emitted token is constrained to that step's reachable
set. Correctness invariant: each trajectory token t_j is in reach_fn(logits_j,
prefix_j) at emission, so the whole trajectory is admissible BY CONSTRUCTION
(strict == lenient; no post-hoc prompt_logprobs audit needed).

The attacker's only freedom is the set-valued proof relation: at a step whose
reachable set has size > 1, any reachable token may replace the canonical one;
the prefix change re-derives all downstream coins, so downstream reach sets are
recomputed step by step along each branch (never free-sampled).

Reusable and deliberately decoupled from argparse/model loading so the same
function can be ported to the GPU harness. Terms:
  step_logits_fn(prefix_ids) -> logits for the next position (np array)
  reach_fn(logits, prefix_ids) -> set of reachable token ids (free-r or paper)
  canon_fn(logits, reach) -> the canonical token. NOTE: canon is the coin-chosen
    token; argmax-within-reach is a PROXY for the coin selection (the honest
    inverse-CDF pick is usually, not always, the highest-logit reachable token)
    — documented deviation, same proxy as the GPU harness.
  answer_fn(text) -> normalized final answer or None (truncated never counts).
"""
import re

CORRECTION = ["Wait", "wait", " Wait", " wait", "But", "However", " But", " However",
              "Actually", " Actually", "recheck", "Recheck", "Let", "let", " Let", " let",
              "Hmm", " Hmm", "mistake", "wrong", "Wrong", "recompute", "check", "Check"]


def default_cfg(**over):
    cfg = dict(max_steps=300, max_groups=4, max_forks=8,
               late_weight=1.0, multi_weight=1.0, steer=False, markers=frozenset(),
               term_ids=frozenset(), is_digit=lambda i: False, decode=lambda i: '')
    cfg.update(over)
    return cfg


def operand_digits(prompt):
    return set(re.findall(r'\d', prompt or ''))


def argmax_in_reach(logits, reach):
    """canon proxy: highest-logit token WITHIN the reachable set (never plain
    argmax — argmax may fall outside reach_paper)."""
    if not reach:
        raise ValueError('empty reachable set — relation has no in-band state')
    return max(reach, key=lambda t: float(logits[t]))


class Stepper:
    """Drives one admissible trajectory, asserting the invariant per step."""

    def __init__(self, step_logits_fn, reach_fn, canon_fn, cfg, budget):
        self.step_logits_fn = step_logits_fn
        self.reach_fn = reach_fn
        self.canon_fn = canon_fn
        self.cfg = cfg
        self.budget = budget          # [remaining tokens] — shared across branches

    def run(self, prefix, repl=None, steer=None):
        """Decode from `prefix`. repl maps relative step -> digit STRING to
        substitute; at execution time the highest-logit reachable token decoding
        to that string is emitted (the prefix has diverged, so the planned
        token id may no longer be reachable — substitution degrades to canon
        and is recorded). Returns (ids, states, meta); states[j] =
        (reach, emitted) for invariant re-checks."""
        repl = repl or {}
        steer = self.cfg['steer'] if steer is None else steer
        cfg = self.cfg
        ids = list(prefix)
        states, degraded = [], []
        while len(ids) - len(prefix) < cfg['max_steps']:
            if self.budget[0] <= 0:
                return ids, states, dict(capped='budget', degraded=degraded)
            logits = self.step_logits_fn(ids)
            reach = self.reach_fn(logits, ids)
            step = len(ids) - len(prefix)
            emit = None
            if step in repl:
                cand = [t for t in reach if cfg['decode'](t).strip() == repl[step]]
                if cand:
                    emit = max(cand, key=lambda t: float(logits[t]))
                else:
                    degraded.append(step)
            if emit is None:
                emit = self.canon_fn(logits, reach)
                if steer and cfg['decode'](emit).strip() in cfg['markers']:
                    alt = [t for t in reach if cfg['decode'](t).strip() not in cfg['markers']]
                    if alt:
                        emit = max(alt, key=lambda t: float(logits[t]))
            assert emit in reach, 'INVARIANT: emitted token outside reachable set'
            ids.append(int(emit))
            self.budget[0] -= 1
            dig = {int(t): float(logits[t]) for t in reach if cfg['is_digit'](t)}
            states.append((set(reach), int(emit), dig))
            if emit in cfg['term_ids']:
                return ids, states, dict(capped=None, degraded=degraded)
        return ids, states, dict(capped='steps', degraded=degraded)


def canonical_forks(states, cfg):
    """Fork candidates along a canonical trajectory: digit steps whose reach
    holds a different digit token. Grouped by decoded digit string; each
    mention keeps (logit, token) candidates for target selection."""
    groups = {}
    for step, (reach, emit, dig) in enumerate(states):
        if not cfg['is_digit'](emit):
            continue
        cands = [(lg, int(t)) for t, lg in dig.items() if t != emit]
        if not cands:
            continue
        d = cfg['decode'](emit).strip()
        g = groups.setdefault(d, dict(digit=d, mentions=[]))
        g['mentions'].append(dict(step=step, canon=int(emit), cands=cands))
    return groups


def order_groups(groups, prompt, cfg, max_forks):
    """Heuristic order: >=2 reachable mentions first (internally consistent
    flips), later mention first (closer to \\boxed -> less room downstream for
    self-correction); operand-digit groups flagged. max_forks caps the total
    mentions considered (kept in order, later groups drop their tail)."""
    ops = operand_digits(prompt)

    def score(g):
        return (cfg['multi_weight'] * (len(g['mentions']) >= 2),
                cfg['late_weight'] * max(m['step'] for m in g['mentions']),
                g['digit'])
    out = []
    for g in sorted(groups.values(), key=score, reverse=True):
        g = dict(g)
        g['operand'] = g['digit'] in ops
        out.append(g)
    kept = 0
    for g in out:
        take = max(0, min(len(g['mentions']), max_forks - kept))
        g['mentions'] = sorted(g['mentions'], key=lambda m: m['step'])[:take] or None
        kept += take
    return [g for g in out if g['mentions']]


def tree_attack(step_logits_fn, reach_fn, canon_fn, base_ids, correct, prompt,
                budget, cfg, answer_fn):
    """Budgeted admissible tree search. Returns dict(ok=..., group=..., dprime=...,
    plan=..., step=..., answer=..., tried=[...], tokens=...). ok=True means a
    concluded admissible trajectory answered != correct. Per-item ASR from this
    is a LOWER bound (more budget can only add hits)."""
    stepper = Stepper(step_logits_fn, reach_fn, canon_fn, cfg, [budget])
    ids, states, meta = stepper.run(base_ids)
    tried = []
    canon_ans = answer_fn(_text(cfg, ids, base_ids))
    if canon_ans is None or canon_ans != correct:
        return dict(ok=False, skipped=True, reason='canonical-answer',
                    answer=canon_ans, tokens=budget - stepper.budget[0], tried=tried)
    groups = order_groups(canonical_forks(states, cfg), prompt, cfg, cfg['max_forks'])
    for g in groups[:cfg['max_groups']]:
        mentions = [m for m in g['mentions'] if m['cands']]
        if not mentions:
            continue
        lg, dprime_id = max((c for m in mentions for c in m['cands']),
                            key=lambda c: c[0])          # d' = highest-logit wrong digit
        dprime = cfg['decode'](dprime_id).strip()
        for plan, steps in (('A', [m['step'] for m in mentions]),
                            ('B', [max(m['step'] for m in mentions)])):
            if stepper.budget[0] <= 0:
                return dict(ok=False, skipped=False, reason='budget', dprime=dprime,
                            tokens=budget - stepper.budget[0], tried=tried)
            tids, tstates, tmeta = stepper.run(base_ids, repl={s: dprime for s in steps})
            ans = answer_fn(_text(cfg, tids, base_ids))
            tried.append(dict(group=g['digit'], dprime=dprime, plan=plan,
                              answer=ans, capped=tmeta['capped'],
                              degraded=len(tmeta['degraded'])))
            if ans is not None and ans != correct:
                return dict(ok=True, group=g['digit'], dprime=dprime, plan=plan,
                            step=steps[0], answer=ans, operand=g['operand'],
                            tokens=budget - stepper.budget[0], tried=tried)
            if plan == 'A' and len(mentions) < 2:
                break                            # single mention: B duplicates A
    return dict(ok=False, skipped=False, reason='exhausted',
                tokens=budget - stepper.budget[0], tried=tried)


def _text(cfg, ids, base_ids):
    return ''.join(cfg['decode'](t) for t in ids[len(base_ids):])
