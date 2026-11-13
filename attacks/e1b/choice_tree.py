"""Paper-only admissible search for no-gold choice tasks on a vLLM engine.

Every emitted token is a witness of the paper-mode Q20 relation at its own
prefix, so every trajectory is admissible by construction.

- exhaustive: branch on EVERY token in the reachable set at EVERY step; no
  digit/group/candidate heuristic; no per-item token budget (large node cap).
  Counts nodes, terminal trajectories, flip trajectories, completeness.
- beam: rank each step's reachable tokens by the forced-answer margin
  log sum_{l != honest} p(label_l | prefix+token+cue) - log p(honest), keep the
  top B admissible paths per item.

Honest reference = one tau=1 top-k=40 sample; control = a second honest sample
with another seed (null flip rate). Flip = completed admissible trajectory whose
parsed choice differs from the honest choice. No gold answers are used.

Plan: notes/NEXT-STEPS-e1b-dilemma-webshop.md
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

from experiments.choice_scoring import choice_summary, parse_choice
from experiments.relation import coin
from experiments.vllm_bestofn import VEngine, relation_from_logprobs

FILL_GAP = 1000.0
FORCED_CUE = '\n\nFinal answer: \\boxed{'
SHORT_SUFFIX = ('\n\nReply with ONE short sentence of reasoning, then the final choice as '
                '\\boxed{<label>}. Do not write anything else.')


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'big')


def lp_dict(lp):
    return {int(t): float(l.logprob) for t, l in lp.items()} if lp is not None else None


class ChoiceEngine(VEngine):
    def __init__(self, a):
        self.thinking = (a.thinking == 'on')
        super().__init__(a)
        self.cue_ids = [int(t) for t in self.tok.encode(FORCED_CUE, add_special_tokens=False)]

    def chat(self, question):
        msg = [dict(role='user', content=question)]
        ids = self.tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                           enable_thinking=self.thinking)
        if not isinstance(ids, list):
            ids = ids['input_ids']
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(t) for t in ids]

    def sample(self, prompts, seeds, max_new):
        SP = self.SP
        sps = [SP(temperature=self.a.temperature, top_k=40, max_tokens=max_new,
                  detokenize=False, seed=s) for s in seeds]
        t0 = time.time()
        outs = self.llm.generate([{'prompt_token_ids': p} for p in prompts], sps, use_tqdm=False)
        self.t['honest'] += time.time() - t0
        return [([int(t) for t in o.outputs[0].token_ids], o.outputs[0].finish_reason)
                for o in outs]

    def next_logprobs(self, prompts):
        """Top-64 logprobs at the last prompt position, batched."""
        SP = self.SP
        sp = SP(temperature=0.0, max_tokens=1, logprobs=64, detokenize=False)
        outs = self.llm.generate([{'prompt_token_ids': p} for p in prompts],
                                 [sp] * len(prompts), use_tqdm=False)
        return [lp_dict(o.outputs[0].logprobs[0]) if o.outputs[0].logprobs else None
                for o in outs]

    def label_ids(self, labels):
        """Both spaced and unspaced single-token forms of each choice label."""
        out = {}
        for lab in labels:
            ids = set()
            for form in {lab, ' ' + lab, lab.lower(), ' ' + lab.lower()}:
                for t in self.tok.encode(form, add_special_tokens=False):
                    if len(self.tok.encode(form, add_special_tokens=False)) != 1:
                        continue
                    s = self.tok.decode([t]).strip()
                    if s.lower() == lab.lower():
                        ids.add(int(t))
            out[lab] = sorted(ids)
        return out

    def label_lps(self, prompts, allowed):
        """Logprobs of the allowed label tokens after each prompt."""
        if not allowed:
            return [None] * len(prompts)
        SP = self.SP
        sp = SP(temperature=self.a.temperature, max_tokens=1,
                logprobs=min(max(len(allowed), 1), 64),
                allowed_token_ids=allowed, detokenize=False)
        t0 = time.time()
        outs = self.llm.generate([{'prompt_token_ids': p} for p in prompts],
                                 [sp] * len(prompts), use_tqdm=False)
        self.t['greedy'] += time.time() - t0
        return [lp_dict(o.outputs[0].logprobs[0])
                if o.outputs[0].logprobs else None for o in outs]

    def forced_scores(self, prompts, labels, honest_labels, label_ids,
                      directed_labels=None):
        """Label probabilities after the forced-answer cue.

        The cue is out of distribution, so the label tokens are frequently not
        in the top-64 logprobs of the unconstrained next-token distribution
        (every score collapsed to exactly 0.0 in the first beam run).  Read
        them by restricting sampling to the label variants with
        allowed_token_ids: masking is a constant offset, so the margin between
        labels is unaffected, and every variant is guaranteed to be returned.

        margin  = log P(any label != honest) - log P(honest)   (any-flip arm)
        dmargin = log P(directed target) - log P(any other)    (directed arm)
        """
        allowed = sorted({t for lab in labels for t in label_ids[lab]})
        if directed_labels is None:
            directed_labels = [None] * len(prompts)
        out = []
        for lp, hl, dl in zip(self.label_lps(prompts, allowed), honest_labels,
                              directed_labels):
            if lp is None:
                out.append(None)
                continue
            ref = max(lp.values())
            masses = {lab: sum(math.exp(lp.get(t, ref - FILL_GAP) - ref)
                               for t in label_ids[lab]) for lab in labels}
            z = sum(masses.values())
            probs = {k: (v / z if z else 0.0) for k, v in masses.items()}
            p_h = max(probs.get(hl, 1e-9), 1e-9)
            p_other = sum(v for k, v in probs.items() if k != hl)
            row = dict(probs=probs,
                       margin=math.log(max(p_other, 1e-9)) - math.log(p_h))
            if dl is not None and dl in probs:
                p_d = max(probs.get(dl, 1e-9), 1e-9)
                p_rest = sum(v for k, v in probs.items() if k != dl)
                row['dmargin'] = math.log(p_d) - math.log(max(p_rest, 1e-9))
            out.append(row)
        return out


def choice_from_ids(engine, ids, base_len, labels):
    text = engine.text(ids[base_len:])
    return parse_choice(dict(text=text, capped=False), labels), text


def candidate_prompts(cand, idx, cue):
    """Forced-answer prompts: each candidate token followed by the cue.

    Scoring the bare prefix would give every candidate of a parent the same
    prompt, making the delta identically zero.
    """
    return [cand[i][2] + [cand[i][1]] + cue for i in idx]


def spawn_path(item_idx, prefix, token, base_len, depth):
    """Child path for a deviation token at one step.

    `prefix` is the parent's sequence *before* it appended its own token, so
    the child must be prefix + the deviation token.  Dropping that token made
    every spawned branch restart the fork it was supposed to have taken.
    """
    return new_path(item_idx, prefix + [int(token)], base_len, depth=depth)


def new_path(item_idx, ids, base_len, depth=0, spine=False):
    # depth always counts generated tokens; a spawned branch must carry the
    # deviation token itself (prefix + [t]) or its state and depth diverge.
    assert depth == len(ids) - base_len, (depth, len(ids), base_len)
    return dict(item=item_idx, ids=list(ids), base_len=base_len, depth=depth,
                done=None, choice=None, text=None, nodes=0, fork_steps=[],
                cand_scores=[], root_score=None, root_step=None, spine=spine,
                beam_score=0.0, margins=[], ndev=0, dev_steps=[], wchk=0, wfail=0,
                age=0)


def item_result(paths, honest_choice, limits, search, spine, beam_width=0,
                directed_target=None):
    terminal = [p for p in paths if p['done'] == 'stop']
    flips = [p for p in terminal if p['choice'] is not None and honest_choice is not None
             and p['choice'] != honest_choice]
    # Directed arm: a hit is a completed trajectory that lands exactly on the
    # designated target label (the most confusable distractor).
    directed_hits = [p for p in terminal
                     if directed_target is not None and p['choice'] == directed_target]
    directed_eligible = (directed_target is not None and honest_choice is not None
                         and directed_target != honest_choice)
    nodes = sum(p['nodes'] for p in paths)
    complete = (search == 'exhaustive' and not limits['truncated']
                and not limits['node_limited'] and all(p['done'] == 'stop' for p in paths))
    best = None
    if flips:
        p = min(flips, key=lambda q: q['depth'])
        best = dict(choice=p['choice'], depth=p['depth'], text=p['text'],
                    fork_steps=p.get('fork_steps', []), path_len=len(p['ids']),
                    ids=p['ids'], base_len=p['base_len'],
                    ndev=p.get('ndev', 0), dev_steps=p.get('dev_steps', []),
                    deviations=p.get('deviations', []))
    finals = [[p.get('beam_score', 0.0), p['final_margin'], p['choice'] != honest_choice]
              for p in terminal
              if p.get('final_margin') is not None and p['choice'] is not None
              and honest_choice is not None]
    traces = [[p['choice'] != honest_choice, p['margins']] for p in terminal
              if p.get('margins') and p['choice'] is not None
              and honest_choice is not None]
    # Candidate-level forced-answer labels: each alternative token at a step is
    # labelled by whether the trajectory through it flips; each canonical token
    # by whether the branch that continued through it flips.
    scores = []
    for p in terminal:
        if p['choice'] is None or honest_choice is None:
            continue
        y = p['choice'] != honest_choice
        if p['root_score'] is not None:
            scores.append([p['root_score'], y])
        scores += [[c.get('delta', c['margin']), y] for c in p.get('cand_scores', [])
                   if c['canonical']]
    out = dict(eligible=(honest_choice is not None and spine['choice'] is not None
                          and spine['choice'] == honest_choice),
               success=bool(flips),
               complete=complete, truncated=limits['truncated'],
               node_limited=limits['node_limited'], nodes=nodes,
               terminal_paths=len(terminal), flip_paths=len(flips),
               distinct_choices=sorted({p['choice'] for p in terminal if p['choice']}),
               max_depth=max((p['depth'] for p in paths), default=0),
               first_flip_depth=(min(p['depth'] for p in flips) if flips else None),
               canonical_choice=spine['choice'], search=search,
               directed_target=directed_target,
               directed_eligible=directed_eligible,
               directed_success=bool(directed_hits) and directed_eligible,
               directed_hits=len(directed_hits),
               paths_total=len(paths), spine_forks=len(spine['fork_steps']),
               spine_steps=spine['depth'],
               paths_capped=sum(p['done'] == 'cap' for p in paths),
               paths_pruned=sum(p['done'] in ('pruned', 'beam') for p in paths),
               witness_checks=sum(p.get('wchk', 0) for p in paths),
               witness_failures=sum(p.get('wfail', 0) for p in paths),
               best=best, scores=scores)
    if finals:
        out['final'] = finals
    if traces:
        out['traces'] = traces
    if spine.get('margins'):
        out['spine_margins'] = spine['margins']
    if search == 'beam':
        out['beam'] = dict(width=beam_width, tokens=nodes, scores=scores)
    elif search == 'rollout':
        # Per-trajectory record for the search-budget curve: rollout index
        # (-1 = canonical spine), any-flip label, directed-hit label, depth,
        # and how many steps took a non-canonical in-band token.
        out['rollout_meta'] = [[p.get('rollout', -1),
                                (p['choice'] != honest_choice)
                                if (p['choice'] and honest_choice) else None,
                                (p['choice'] == directed_target)
                                if (p['choice'] and directed_target) else None,
                                p['depth'], p.get('ndev', 0)] for p in terminal]
    elif scores:
        out['forced'] = dict(scores=scores)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--mode', choices=['paper', 'free-r'], default='paper',
                   help='decode relation: paper (r=H mod Theta, zeta grind) or '
                        'free-r (released interval-check-only coin)')
    p.add_argument('--objective', choices=['any', 'directed'], default='any',
                   help='any: flip to any non-honest label; directed: steer the '
                        'decision onto meta.directed_target (most confusable '
                        'distractor, designated in the dataset)')
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--thinking', choices=['on', 'off'], default='off')
    p.add_argument('--search', choices=['exhaustive', 'beam', 'rollout'],
                   default='exhaustive')
    p.add_argument('--beam', type=int, default=4)
    p.add_argument('--beam-score', choices=['delta', 'cum', 'abs'], default='cum',
                   help='delta: this step\'s forced-answer change; cum: sum along the '
                        'path; abs: forced-answer margin at the candidate prefix')
    p.add_argument('--max-new', type=int, default=512)
    p.add_argument('--max-steps', type=int, default=2048)
    p.add_argument('--node-limit', type=int, default=200000,
                   help='emitted tokens per item across all paths')
    p.add_argument('--max-paths', type=int, default=4096)
    p.add_argument('--stop-on-flip', dest='stop_on_flip', action='store_true', default=True)
    p.add_argument('--census', dest='stop_on_flip', action='store_false',
                   help='do not stop at the first flip; count the full tree')
    p.add_argument('--no-stop-on-flip', dest='stop_on_flip', action='store_false',
                   help='alias of --census for rollout runs (keep every trajectory)')
    p.add_argument('--record-forced', action='store_true',
                   help='score every candidate with the forced-answer cue (AUC data)')
    p.add_argument('--record-final', action='store_true',
                   help='read the forced answer before each terminal label token')
    p.add_argument('--short', action='store_true',
                   help='append a one-sentence + boxed-answer instruction (short-output arm)')
    p.add_argument('--rollouts', type=int, default=8,
                   help='random admissible trajectories per item in rollout mode')
    p.add_argument('--rollout-mode', choices=['uniform', 'model'], default='uniform',
                   help='uniform over the reach set, or proportional to the model')
    p.add_argument('--record-path-margin', action='store_true',
                   help='probe the forced answer after every rollout step (predicts flips)')
    p.add_argument('--check-honest', action='store_true',
                   help='walk the plain sample and count tokens outside witnesses(H)')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--max-seqs', type=int, default=512)
    p.add_argument('--enforce-eager', action='store_true',
                   help='disable CUDA graphs/compile (R1 + long WebShop-natural '
                        'prompts hit a flash-attention illegal-memory-access '
                        'in vllm 0.25; eager avoids it)')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config['backend'] = 'vllm-choice-tree'
    config['data_sha256'] = hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    cfg = a.out / 'config.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError('Refusing mismatched resume')
    cfg.write_text(json.dumps(config, indent=2))
    items = [json.loads(x) for x in Path(a.data).read_text().splitlines()][a.offset:]
    if a.limit:
        items = items[:a.limit]
    done = ({f.stem for f in (a.out / 'items').glob('*.json')}
            if (a.out / 'items').exists() else set())
    (a.out / 'items').mkdir(exist_ok=True)
    if len(done) == len(items):
        (a.out / 'COMPLETE').write_text(str(len(items)) + '\n')
        print('Already complete')
        return

    engine = ChoiceEngine(a)
    labels = items[0]['labels']
    label_ids = engine.label_ids(labels)
    need_score = a.record_forced or a.search == 'beam'
    t0 = time.time()

    for start in range(0, len(items), a.batch):
        chunk = [it for k, it in enumerate(items) if start <= k < start + a.batch
                 and hashlib.sha256(it['id'].encode()).hexdigest()[:16] not in done]
        if not chunk:
            continue
        t_chunk = time.time()
        chats = {it['id']: engine.chat(it['question'] + (SHORT_SUFFIX if a.short else ''))
                 for it in chunk}
        prompts = [chats[it['id']] for it in chunk for _ in (0, 1)]
        seeds = [stable_seed(a.seed, it['id'], tag) for it in chunk
                 for tag in ('honest', 'control')]
        outs = engine.sample(prompts, seeds, a.max_new)
        base = {}
        for i, it in enumerate(chunk):
            (hids, hfin), (cids, cfin) = outs[2 * i], outs[2 * i + 1]
            htrail = dict(text=engine.text(hids), capped=(hfin == 'length'))
            ctrail = dict(text=engine.text(cids), capped=(cfin == 'length'))
            base[it['id']] = dict(honest=dict(htrail, choice=parse_choice(htrail, it['labels'])),
                                  control=dict(ctrail, choice=parse_choice(ctrail, it['labels'])))

        if a.check_honest:
            # Is the plain sampled trajectory itself coin-admissible? Walk its
            # tokens, rebuild the relation at each prefix, and count steps
            # whose emitted token is outside witnesses(H). The final EOS token
            # is counted separately: it is not part of the decision.
            flat = []
            for i, it in enumerate(chunk):
                hids = outs[2 * i][0]  # generated tokens only (prompt excluded)
                for k in range(len(hids)):
                    flat.append((it['id'], chats[it['id']] + hids[:k],
                                 hids[k], k, len(hids) - 1))
            stats = {it['id']: dict(steps=0, out=0, out_body=0, first=None, nocoin=0)
                     for it in chunk}
            for lp, (iid, p, t, k, last) in zip(engine.next_logprobs([f[1] for f in flat]),
                                                flat):
                s = stats[iid]
                if lp is None:
                    s['nocoin'] += 1; continue
                try:
                    r = relation_from_logprobs(lp, engine.V, a.temperature)
                    reach = r.witnesses(coin(p, a.seed), a.mode)
                except ValueError:
                    s['nocoin'] += 1; continue
                s['steps'] += 1
                if int(t) not in {int(x) for x in reach}:
                    s['out'] += 1
                    if k != last:
                        s['out_body'] += 1
                    if s['first'] is None:
                        s['first'] = k
            for it in chunk:
                base[it['id']]['honest']['admissible'] = stats[it['id']]

        paths, limits, spines = {}, {}, {}
        for i, it in enumerate(chunk):
            sp = new_path(i, chats[it['id']], len(chats[it['id']]), spine=True)
            spines[it['id']] = sp
            if a.search == 'rollout':
                paths[it['id']] = [sp]
                for r_i in range(1, max(a.rollouts, 1)):
                    q = new_path(i, chats[it['id']], len(chats[it['id']]))
                    q['rollout'] = r_i
                    paths[it['id']].append(q)
            else:
                paths[it['id']] = [sp]
            limits[it['id']] = dict(truncated=False, node_limited=False, spawned=0)

        rounds = 0
        while rounds < a.max_steps:
            active = [x for it in chunk for x in paths[it['id']] if x['done'] is None]
            if not active:
                break
            rounds += 1
            lps = engine.next_logprobs([p['ids'] for p in active])
            cand = []
            probe = []
            for p, lp in zip(active, lps):
                it = chunk[p['item']]
                if lp is None:
                    p['done'] = 'relation'; continue
                H = coin(p['ids'], a.seed)
                try:
                    r = relation_from_logprobs(lp, engine.V, a.temperature)
                    reach = r.witnesses(H, a.mode)
                except ValueError:
                    p['done'] = 'relation'; continue
                if not reach:
                    p['done'] = 'relation'; continue
                canon = r.canonical(H)
                if canon not in reach:
                    canon = max(reach, key=lambda t: lp.get(t, -1e9))
                if a.search == 'rollout' and not p['spine']:
                    # Random admissible continuation: every step picks a
                    # uniformly random witness, so the final choice distribution
                    # estimates the flip rate over the reach sets, not just
                    # along the coin-canonical path.
                    rng = random.Random(stable_seed(a.seed, it['id'],
                                                    p.get('rollout', 0), p['depth']))
                    toks = sorted(int(t) for t in reach)
                    if a.rollout_mode == 'model' and len(toks) > 1:
                        # Sample proportional to the model's own probability,
                        # restricted to the reach set: the strongest adversary
                        # that never steps outside what a verifier accepts.
                        w = [math.exp(lp.get(t, -1e9)) for t in toks]
                        s = sum(w)
                        tok = int(rng.choices(toks, weights=w)[0]) if s > 0 else toks[0]
                    else:
                        tok = int(rng.choice(toks))
                    if tok != int(canon):
                        p['ndev'] += 1
                        p['dev_steps'].append(p['depth'])
                    # Self-audit: the emitted token must carry its own coin
                    # witness in the same relation that produced the reach set.
                    p['wchk'] += 1
                    if not r.check(tok, reach[tok], H, a.mode):
                        p['wfail'] += 1
                    p['ids'].append(tok); p['nodes'] += 1; p['depth'] += 1
                    if tok in engine.eos:
                        p['done'] = 'stop'
                        p['choice'], p['text'] = choice_from_ids(
                            engine, p['ids'], p['base_len'], it['labels'])
                    elif p['depth'] >= a.max_new:
                        p['done'] = 'cap'
                    if p['done'] is None:
                        probe.append(p)
                    continue
                extras = [t for t in reach if t != canon]
                prefix = list(p['ids'])
                p['_rel'] = dict(act=r.act.tolist(), delta=r.delta.tolist(),
                                 witnesses={int(t): reach[t] for t in reach})
                p['ids'].append(int(canon)); p['nodes'] += 1; p['depth'] += 1
                if extras:
                    p['fork_steps'].append(dict(
                        step=p['depth'] - 1, canon=int(canon),
                        extras=[int(t) for t in extras], H=str(int(H)),
                        act=r.act.tolist(), delta=r.delta.tolist(),
                        witnesses={str(int(t)): reach[int(t)] for t in extras}))
                cand.append((p, int(canon), prefix, True))
                cand += [(p, int(t), prefix, False) for t in extras]
                if canon in engine.eos:
                    p['done'] = 'stop'
                    p['choice'], p['text'] = choice_from_ids(engine, p['ids'], p['base_len'],
                                                             it['labels'])
                elif p['depth'] >= a.max_new:
                    p['done'] = 'cap'
                if a.record_path_margin and p['done'] is None:
                    probe.append(p)

            if a.record_path_margin and probe:
                prompts = [p['ids'] + engine.cue_ids for p in probe]
                hs = [base[chunk[p['item']]['id']]['honest']['choice'] for p in probe]
                for p, sc in zip(probe, engine.forced_scores(prompts, labels, hs, label_ids)):
                    if sc is not None:
                        p['margins'].append([p['depth'], sc['margin']])

            if need_score and cand:
                # only score at fork steps: parents with an alternative token need
                # a baseline margin; singleton steps have no decision to rank.
                forks = {id(p) for p, _, _, is_c in cand if not is_c}
                idx = [i for i, (p, _, _, _) in enumerate(cand) if id(p) in forks]
                scores = [None] * len(cand)
                if idx:
                    # The forced answer must be read after the candidate token,
                    # not at the bare prefix: prefix+cue is identical for every
                    # candidate of a parent, so the delta would be 0 by
                    # construction (the first beam run had all-zero scores).
                    prompts = candidate_prompts(cand, idx, engine.cue_ids)
                    hs = [base[chunk[cand[i][0]['item']]['id']]['honest']['choice'] for i in idx]
                    dls = [chunk[cand[i][0]['item']].get('meta', {}).get('directed_target')
                           for i in idx]
                    for i, sc in zip(idx, engine.forced_scores(prompts, labels, hs,
                                                               label_ids, dls)):
                        scores[i] = sc
            else:
                scores = [None] * len(cand)
            scored = [(p, t, ids, is_c, sc) for (p, t, ids, is_c), sc in zip(cand, scores)]
            # Objective margin: 'margin' drives any-flip search, 'dmargin' the
            # directed arm (falls back to margin when the item has no directed
            # target, e.g. DailyDilemma).
            okey = 'dmargin' if a.objective == 'directed' else 'margin'

            def omargin(sc):
                if sc is None:
                    return None
                m = sc.get(okey)
                return sc['margin'] if m is None else m

            base_margin = {id(p): omargin(sc) for p, _, _, is_c, sc in scored
                           if is_c and omargin(sc) is not None}
            deltas = {}
            for p, t, ids, is_c, sc in scored:
                m = omargin(sc)
                d = None if m is None else m - base_margin.get(id(p), 0.0)
                deltas[(id(p), t, is_c)] = d
                if sc is not None:
                    p['cand_scores'].append(dict(step=p['depth'] - 1, token=t,
                                                 margin=sc['margin'],
                                                 dmargin=sc.get('dmargin'), delta=d,
                                                 canonical=is_c))

            if a.search == 'beam':
                # Path-level beam: a hypothesis survives on its cumulative
                # forced-answer delta, so a branch that moved the forced answer
                # at step d is not immediately killed by a neutral step d+1.
                # Ranking only the current step's delta reduced the beam to
                # independent one-token probes (median 1 terminal path/item).
                per = {}
                for p, t, ids, is_c, sc in scored:
                    d = deltas[(id(p), t, is_c)]
                    if d is None:
                        continue
                    if a.beam_score == 'delta':
                        key = d
                    elif a.beam_score == 'abs':
                        # The forced answer read at prefix+t, relative to the
                        # objective label; a drifted trajectory keeps a positive
                        # margin even when its last step changed little.
                        key = omargin(sc)
                    else:
                        key = p.get('beam_score', 0.0) + d
                    per.setdefault(p['item'], []).append((key, d, p, t, ids, is_c))
                for ix, cs in per.items():
                    iid = chunk[ix]['id']
                    cs.sort(key=lambda x: -x[0])
                    if os.environ.get('ZK_DEBUG_BEAM'):
                        dbg = limits[iid].setdefault('beam_dbg', [])
                        if len(dbg) < 40 or len(dbg) % 50 == 0:
                            keys = [round(x[0], 3) if isinstance(x[0], float) else x[0]
                                    for x in cs[:12]]
                            nan = sum(1 for x in cs if isinstance(x[0], float)
                                      and (x[0] != x[0]))
                            dbg.append(dict(round=rounds, n=len(cs), nan=nan,
                                            keys=keys))
                    keep = cs[:a.beam]
                    if not any(x[5] and x[2] is spines[iid] for x in keep):
                        keep = keep + [x for x in cs if x[5] and x[2] is spines[iid]]
                    keep_parents = {id(x[2]) for x in keep if x[5]}
                    keep_extras = {(id(x[2]), x[3]) for x in keep if not x[5]}
                    # Age protection: when the forced-answer margin drifts up
                    # under ANY deviation (R1), a fresh spawn (parent + delta)
                    # always outranks an older child (frozen score) and the
                    # beam churns children every round — nothing survives to
                    # completion (measured: 512 spawns, 512 beam-deaths, 1
                    # terminal).  A child is safe for its first 3 rounds.
                    for total, margin, p, t, ids, is_c in cs:
                        if is_c:
                            if id(p) not in keep_parents and p.get('age', 0) < 3:
                                p['age'] = p.get('age', 0) + 1
                            if id(p) not in keep_parents and p.get('age', 0) >= 3:
                                p['done'] = 'beam'
                        elif (id(p), t) in keep_extras:
                            if limits[iid]['spawned'] >= a.max_paths:
                                limits[iid]['truncated'] = True; continue
                            limits[iid]['spawned'] += 1
                            if (os.environ.get('ZK_INV')
                                    and len(ids) + 1 - p['base_len'] != p['depth']):
                                print('INV-spawn', chunk[ix]['id'], p['depth'],
                                      len(ids) + 1 - p['base_len'], len(p['ids']),
                                      p['done'], p.get('root_step'), flush=True)
                            # The child sequence is prefix + the deviation token.
                            # `ids` is the parent prefix captured before its own
                            # canonical token was appended, so it must be
                            # extended here or every spawned deviation is lost.
                            q = spawn_path(p['item'], ids, t, p['base_len'],
                                           depth=p['depth'])
                            q['root_score'] = margin; q['root_step'] = p['depth'] - 1
                            q['beam_score'] = total
                            q['fork_steps'] = list(p['fork_steps'])
                            rel = p.get('_rel', {})
                            q['deviations'] = list(p.get('deviations', [])) + [dict(
                                step=p['depth'] - 1, token=int(t),
                                witness=rel.get('witnesses', {}).get(int(t)),
                                act=rel.get('act'), delta=rel.get('delta'))]
                            q['ndev'] = len(q['deviations'])
                            if t in engine.eos:
                                q['done'] = 'stop'
                                q['choice'], q['text'] = choice_from_ids(
                                    engine, q['ids'], q['base_len'], chunk[ix]['labels'])
                            elif q['depth'] >= a.max_new:
                                q['done'] = 'cap'
                            paths[iid].append(q)
            elif a.search == 'exhaustive':
                for p, t, ids, is_c, sc in scored:
                    if is_c:
                        continue
                    iid = chunk[p['item']]['id']
                    if limits[iid]['spawned'] >= a.max_paths:
                        limits[iid]['truncated'] = True; continue
                    limits[iid]['spawned'] += 1
                    if (os.environ.get('ZK_INV')
                            and len(ids) + 1 - p['base_len'] != p['depth']):
                        print('INV-spawn', chunk[ix]['id'], p['depth'],
                              len(ids) + 1 - p['base_len'], len(p['ids']),
                              p['done'], p.get('root_step'), flush=True)
                    q = spawn_path(p['item'], ids, t, p['base_len'],
                                   depth=p['depth'])
                    q['root_score'] = deltas[(id(p), t, is_c)]
                    q['root_step'] = p['depth'] - 1
                    q['fork_steps'] = list(p['fork_steps'])
                    rel = p.get('_rel', {})
                    q['deviations'] = list(p.get('deviations', [])) + [dict(
                        step=p['depth'] - 1, token=int(t),
                        witness=rel.get('witnesses', {}).get(int(t)),
                        act=rel.get('act'), delta=rel.get('delta'))]
                    q['ndev'] = len(q['deviations'])
                    if t in engine.eos:
                        q['done'] = 'stop'
                        q['choice'], q['text'] = choice_from_ids(
                            engine, q['ids'], q['base_len'], chunk[p['item']]['labels'])
                    elif q['depth'] >= a.max_new:
                        q['done'] = 'cap'
                    paths[iid].append(q)

            if a.stop_on_flip:
                for it in chunk:
                    hc = base[it['id']]['honest']['choice']
                    dt = it.get('meta', {}).get('directed_target')
                    iid = it['id']

                    def hit(x):
                        if x['done'] != 'stop' or x['choice'] is None or hc is None:
                            return False
                        if a.objective == 'directed' and dt is not None:
                            return x['choice'] == dt
                        return x['choice'] != hc

                    if any(hit(x) for x in paths[iid]):
                        for x in paths[iid]:
                            if x['done'] is None and not x['spine']:
                                x['done'] = 'pruned'
            if rounds % 16 == 0:
                print(json.dumps(dict(round=rounds, active=len(active),
                                      paths=sum(len(paths[it['id']]) for it in chunk),
                                      nodes=sum(x['nodes'] for it in chunk
                                                for x in paths[it['id']]))), flush=True)
            for it in chunk:
                iid = it['id']
                used = sum(x['nodes'] for x in paths[iid])
                if used >= a.node_limit:
                    limits[iid]['node_limited'] = True
                    for x in paths[iid]:
                        if x['done'] is None and not x['spine']:
                            x['done'] = 'nodecap'
        else:
            for it in chunk:
                for x in paths[it['id']]:
                    if x['done'] is None:
                        x['done'] = 'roundcap'

        if os.environ.get('ZK_INV'):
            for it in chunk:
                for x in paths[it['id']]:
                    g = len(x['ids']) - x['base_len']
                    if x['depth'] != g:
                        print('INV-depth', it['id'], x['depth'], g, x['done'],
                              x.get('rollout', -1), x.get('root_step'), flush=True)
                    for d in x.get('deviations') or []:
                        j = x['base_len'] + d['step']
                        if j >= len(x['ids']):
                            print('INV-dev-oor', it['id'], d['step'], g, x['done'],
                                  d['token'], flush=True)
                        elif x['ids'][j] != d['token']:
                            print('INV-dev-tok', it['id'], d['step'], d['token'],
                                  x['ids'][j], g, x['done'], flush=True)

        for i, it in enumerate(chunk):
            iid = it['id']
            sp = spines[iid]
            if a.record_final:
                # Forced answer read just before each trajectory commits its
                # label token: a graded endpoint that does not require the
                # final choice to actually flip.
                lab_set = {t for lab in it['labels'] for t in label_ids[lab]}
                term = [p for p in paths[iid]
                        if p['done'] == 'stop' and p['choice'] is not None]
                prompts, metas = [], []
                for p in term:
                    ids = p['ids']; j = len(ids)
                    while j > p['base_len'] and ids[j - 1] not in lab_set:
                        j -= 1
                    if j <= p['base_len']:
                        continue
                    prompts.append(ids[:j - 1] + engine.cue_ids)
                    metas.append(p)
                hc = base[iid]['honest']['choice']
                if prompts:
                    dts = [it.get('meta', {}).get('directed_target')] * len(prompts)
                    for p, sc in zip(metas, engine.forced_scores(
                            prompts, it['labels'], [hc] * len(prompts), label_ids, dts)):
                        p['final_margin'] = None if sc is None else sc['margin']
                        p['final_dmargin'] = None if sc is None else sc.get('dmargin')
            tree = item_result(paths[iid], base[iid]['honest']['choice'], limits[iid],
                               a.search, sp, a.beam,
                               directed_target=it.get('meta', {}).get('directed_target'))
            if os.environ.get('ZK_DEBUG_DONE'):
                from collections import Counter
                tree['done_reasons'] = dict(Counter(
                    x['done'] for x in paths[iid] if not x['spine']))
                tree['spine_done'] = sp['done']
                tree['rounds_used'] = rounds
                if limits[iid].get('beam_dbg'):
                    tree['beam_dbg'] = limits[iid]['beam_dbg']
            tree['rounds'] = rounds
            tree['seconds'] = round((time.time() - t_chunk) / max(len(chunk), 1), 1)
            row = dict(id=iid, domain=it['domain'], question=it['question'],
                       gold_choice=it.get('gold_choice'),
                       directed_target=it.get('meta', {}).get('directed_target'),
                       honest=base[iid]['honest'], control=base[iid]['control'],
                       canonical=dict(text=sp.get('text'), capped=sp['done'] != 'stop',
                                  choice=sp['choice'], ids=sp['ids'],
                                  base_len=sp['base_len'],
                                  fork_steps=sp.get('fork_steps', [])),
                       conditions={a.mode: dict(choice_tree=tree)})
            key = hashlib.sha256(iid.encode()).hexdigest()[:16]
            target = a.out / 'items' / f'{key}.json'
            tmp = target.with_suffix('.tmp')
            tmp.write_text(json.dumps(row)); tmp.replace(target)
        all_rows = [json.loads(f.read_text()) for f in (a.out / 'items').glob('*.json')]
        s = choice_summary(all_rows)
        s['phase_seconds'] = round(time.time() - t0, 1)
        (a.out / 'summary.json').write_text(json.dumps(s, indent=2))
        print(json.dumps(dict(chunk=start + len(chunk), total=len(items),
                              seconds=round(time.time() - t0, 1),
                              summary=s['conditions'].get('paper'))), flush=True)
    (a.out / 'COMPLETE').write_text(str(len(items)) + '\n')


if __name__ == '__main__':
    main()