"""Attention-lever choice search: sign-matrix steering + decode tolerance on vLLM.

Arms (vs the sampling-only arms of choice_tree.py, which run with config 0):
  attn-only    decode is strictly canonical (r = H mod Theta on the CURRENT
               logits); the attacker's only freedom is the per-(layer,head)
               attention row-mass sign matrix.  Menu at a step =
               { canonical(H, logits(C)) : C in {0, S*} }.
  attn+decode  union of the full paper witness set under logits(0) and
               logits(S*) — both freedoms together.

S* is a per-item sign matrix, layer-granularity coordinate ascent maximising
the run objective (any-flip margin / directed margin) at the spine's
forced-answer decision point.  Row-uniform signs are a subset of the
admissible per-row zeta choices, so every configuration used here is
admissible; effects are lower bounds.

Cache protocol (prefix cache is config-blind):
  * menu probes [prefix] under C recompute only the last row (kv_cache_manager
    guarantees >=1 recomputed token) and write it back under C; after all C
    probes the runner re-canonicalises [prefix] under 0 before the spine
    continues;
  * a fork token t belongs to exactly ONE config (dedup): [prefix+t...] under
    that config writes t's row once, and every later read is consistent with
    the committed fork semantics.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments import attn_slack_vllm as slack
from experiments.choice_scoring import choice_summary, parse_choice
from experiments.choice_tree import (ChoiceEngine, choice_from_ids, item_result,
                                     lp_dict, new_path, null_walks, spawn_path,
                                     stable_seed)
from experiments.relation import coin
from experiments.vllm_bestofn import relation_from_logprobs


def canon_choice(engine, ids, base_len, labels):
    return choice_from_ids(engine, ids, base_len, labels)


def slack_generate(engine, prompts, sp, signs, rows='uniform'):
    """One generate() call under one sign matrix.

    rows='uniform'  -> matrix active at every position (row-uniform);
    rows='last'     -> matrix active ONLY at each prompt's last (decision)
                       row — the per-position relaxation: one buffer write
                       covers a heterogeneous batch because every prompt's
                       decision row gathers its own position column."""
    if signs is None:
        slack.set_signs(None)
    elif rows == 'last':
        slack.set_signs(signs, positions=[len(p) - 1 for p in prompts])
    else:
        slack.set_signs(signs)
    try:
        return engine.llm.generate([{'prompt_token_ids': p} for p in prompts],
                                   [sp] * len(prompts), use_tqdm=False)
    finally:
        slack.set_signs(None)


def next_lps(engine, prompts, signs=None, rows='uniform'):
    sp = engine.SP(temperature=0.0, max_tokens=1, logprobs=64, detokenize=False)
    outs = slack_generate(engine, prompts, sp, signs, rows)
    return [lp_dict(o.outputs[0].logprobs[0]) if o.outputs[0].logprobs else None
            for o in outs]


def walk_spines(engine, chats, a, items):
    """Phase A: canonical spine per item, batched rounds (config 0)."""
    paths = {it['id']: new_path(i, chats[it['id']], len(chats[it['id']]), spine=True)
             for i, it in enumerate(items)}
    by_id = {it['id']: it for it in items}
    rounds = 0
    while any(p['done'] is None for p in paths.values()) and rounds < a.max_steps:
        rounds += 1
        act = [(iid, p) for iid, p in paths.items() if p['done'] is None]
        lps = next_lps(engine, [p['ids'] for _, p in act])
        for (iid, p), lp in zip(act, lps):
            if lp is None:
                p['done'] = 'relation'; continue
            it = by_id[iid]
            H = coin(p['ids'], a.seed)
            try:
                r = relation_from_logprobs(lp, engine.V, a.temperature)
                tok = r.canonical(H)
            except ValueError:
                p['done'] = 'relation'; continue
            p['ids'].append(int(tok)); p['nodes'] += 1; p['depth'] += 1
            if tok in engine.eos:
                p['done'] = 'stop'
                p['choice'], p['text'] = canon_choice(engine, p['ids'], p['base_len'],
                                                      it['labels'])
            elif p['depth'] >= a.max_new:
                p['done'] = 'cap'
    return paths


def objective_margin(engine, spine_ids, labels, honest, directed, label_ids,
                     objective):
    """Forced-answer objective at the spine decision point (config 0)."""
    sc = engine.forced_scores([spine_ids + engine.cue_ids], labels, [honest],
                              label_ids, [directed or None])
    if not sc or sc[0] is None:
        return None
    return sc[0].get('dmargin') if (objective == 'directed' and directed) \
        else sc[0]['margin']


def optimise_signs(engine, spine_ids, labels, honest, directed, label_ids, a,
                   n_layers, n_heads):
    """Layer-granularity coordinate ascent on the chosen objective.

    objective 'margin': the forced-answer (directed) margin at [spine+cue] —
    signs that move the decision.
    objective 'reach':  |witnesses(H, logits(S))| at the last spine step —
    signs that maximise the single-step reachable token count (the fork-menu
    size the decode relation admits under the perturbed computation).
    Each trial is one generate under the trial matrix (row-uniform signs).
    Returns (S*, value).
    """
    H = coin(spine_ids, a.seed)

    def value(S):
        if a.opt_objective == 'reach':
            lp = next_lps(engine, [spine_ids], signs=S, rows=a.sign_rows)[0]
            if lp is None:
                return None
            try:
                r = relation_from_logprobs(lp, engine.V, a.temperature)
                return float(len(r.witnesses(H, a.mode)))
            except ValueError:
                return None
        # margin objective: in 'last' mode the matrix is active only at the
        # cue's final (decision) row
        cue = spine_ids + engine.cue_ids
        if a.sign_rows == 'last':
            slack.set_signs(S, positions=[len(cue) - 1])
        else:
            slack.set_signs(S)
        try:
            sc = engine.forced_scores([cue], labels,
                                      [honest], label_ids,
                                      [directed or None])[0]
        finally:
            slack.set_signs(None)
        if sc is None:
            return None
        return sc.get('dmargin') if (a.objective == 'directed' and directed) \
            else sc['margin']

    S = np.zeros((n_layers, n_heads), dtype=int)
    best = value(S)
    if best is None:
        return None, None
    order = list(range(n_layers))
    if a.opt_order == 'rev':
        order.reverse()
    for li in order:
        cand_best, cand_sign = best, 0
        for s in (1, -1):
            S[li, :] = s
            v = value(S)
            if v is not None and v > cand_best:
                cand_best, cand_sign = v, s
        S[li, :] = cand_sign
        best = cand_best
    return S, best


def canonical_witness(r, H):
    """Decode-free witness fields for r.canonical(H) (attn-only forks)."""
    total = int(r.honest_mass.sum())
    cdf = r.honest_mass.cumsum()
    j = int(np.searchsorted(cdf, H % total, side='right')) if total else 0
    j = min(j, len(cdf) - 1)
    return dict(shift=int(r.honest_shift), total=total, r=int(H % total),
                prefix=int(cdf[j - 1]) if j else 0, next_prefix=int(cdf[j]),
                canonical=True)


def beam_search(engine, chunk, chats, base, a, paths, limits, spines, signs_by_item,
                label_ids, labels):
    """Phase C: canonical spine + config-keyed fork menus, beam over paths."""
    by_id = {it['id']: it for it in chunk}
    rounds = 0
    while rounds < a.max_steps:
        active = [x for it in chunk for x in paths[it['id']] if x['done'] is None]
        if not active:
            break
        rounds += 1
        # (1) S*-menu logits first (contaminates the last row of each probed
        #     prefix); (2) the canonical call recomputes it under 0, which both
        #     restores the cache and yields this round's canonical relation.
        groups = {}
        for p in active:
            iid = chunk[p['item']]['id']
            S = signs_by_item.get(iid)
            if S is not None and p.get('_pending_lp') is None:
                groups.setdefault(id(S), (S, []))[1].append(p)
        slps = {}
        for _, (S, ps) in groups.items():
            for p, lp in zip(ps, next_lps(engine, [p['ids'] for p in ps], signs=S,
                                           rows=a.sign_rows)):
                slps[id(p)] = (S, lp)
        need = [p for p in active if p.get('_pending_lp') is None]
        clps = next_lps(engine, [p['ids'] for p in need])
        clps = dict(zip(map(id, need), clps))

        cand = []  # (path, token, prefix, is_canonical, config, lp_for_relation)
        for p in active:
            it = chunk[p['item']]
            iid = it['id']
            pend = p.pop('_pending_lp', None)
            lp = pend if pend is not None else clps.get(id(p))
            if lp is None:
                p['done'] = 'relation'; continue
            H = coin(p['ids'], a.seed)
            try:
                r = relation_from_logprobs(lp, engine.V, a.temperature)
            except ValueError:
                p['done'] = 'relation'; continue
            try:
                canon = r.canonical(H)
            except Exception:
                p['done'] = 'relation'; continue
            # menu under config 0: decode-tolerance witnesses (paper/free-r)
            menu = {}
            if a.arm in ('attn+decode', 'sampling'):
                try:
                    menu = {int(t): w for t, w in r.witnesses(H, a.mode).items()}
                except ValueError:
                    menu = {}
            menu.setdefault(int(canon), canonical_witness(r, H))
            # menu under S*: each entry carries its own perturbed relation
            # state so the model-free audit can replay it (fork 'attn' list).
            attn_extras = {}
            got = slps.get(id(p))
            if got is not None and a.arm != 'sampling':
                S, slp = got
                if slp is not None:
                    try:
                        r2 = relation_from_logprobs(slp, engine.V, a.temperature)
                        c2 = int(r2.canonical(H))
                        if a.arm == 'attn-only':
                            if c2 not in menu:
                                attn_extras[c2] = (canonical_witness(r2, H),
                                                   r2.act.tolist(),
                                                   r2.delta.tolist(), S)
                        else:
                            for t, w in r2.witnesses(H, a.mode).items():
                                if int(t) not in menu:
                                    attn_extras[int(t)] = (w, r2.act.tolist(),
                                                           r2.delta.tolist(), S)
                    except ValueError:
                        pass
            prefix = list(p['ids'])
            p['_rel0'] = dict(act=r.act.tolist(), delta=r.delta.tolist())
            canon_tok = int(canon)
            p['ids'].append(canon_tok); p['nodes'] += 1; p['depth'] += 1
            extras = [(t, w, None) for t, w in menu.items() if t != canon_tok]
            extras += [(t, w, S) for t, (w, _act, _dl, S) in attn_extras.items()]
            if extras:
                p['fork_steps'].append(dict(
                    step=p['depth'] - 1, canon=canon_tok,
                    extras=[int(t) for t, _, _ in extras], H=str(int(H)),
                    act=r.act.tolist(), delta=r.delta.tolist(),
                    witnesses={str(int(t)): w for t, w, S in extras
                               if S is None},
                    attn=[dict(token=int(t), witness=w, act=act, delta=dl)
                          for t, (w, act, dl, _S) in attn_extras.items()]))
            cand.append((p, canon_tok, prefix, True, None, None))
            for t, w, S in extras:
                cand.append((p, int(t), prefix, False, S,
                             attn_extras.get(int(t), (None, None, None, None))[1:3]))
            if canon_tok in engine.eos:
                p['done'] = 'stop'
                p['choice'], p['text'] = canon_choice(engine, p['ids'], p['base_len'],
                                                      it['labels'])
            elif p['depth'] >= a.max_new:
                p['done'] = 'cap'

        # fork scoring, grouped by config (0 + each S*)
        if cand:
            forks = {id(p) for c in cand if not c[3] for p in [c[0]]}
            idx = [i for i, c in enumerate(cand) if id(c[0]) in forks]
            scored = [None] * len(cand)
            byconf = {}
            for i in idx:
                p, t, prefix, _, S, _rp = cand[i]
                byconf.setdefault(id(S) if S is not None else 0, (S, []))[1].append(i)
            for _, (S, ixs) in byconf.items():
                prompts = [cand[i][2] + [cand[i][1]] + engine.cue_ids for i in ixs]
                hs = [base[chunk[cand[i][0]['item']]['id']]['honest']['choice']
                      for i in ixs]
                dts = [by_id[chunk[cand[i][0]['item']]['id']].get('meta', {}).get(
                    'directed_target') for i in ixs]
                # score each candidate under ITS fork's config (this was
                # previously missed: S* candidates were scored canonically)
                if S is None:
                    slack.set_signs(None)
                elif a.sign_rows == 'last':
                    slack.set_signs(S, positions=[len(x) - 1 for x in prompts])
                else:
                    slack.set_signs(S)
                try:
                    scs = engine.forced_scores(prompts, labels, hs, label_ids, dts)
                finally:
                    slack.set_signs(None)
                for i, sc in zip(ixs, scs):
                    scored[i] = sc
            okey = 'dmargin' if a.objective == 'directed' else 'margin'

            def omargin(sc):
                if sc is None:
                    return None
                m = sc.get(okey)
                return sc['margin'] if m is None else m

            base_m = {}
            deltas = {}
            abskey = {}
            for i, c in enumerate(cand):
                m = omargin(scored[i])
                if c[3] and m is not None:
                    base_m[id(c[0])] = m
            for i, (p, t, prefix, is_c, S, _rp) in enumerate(cand):
                m = omargin(scored[i])
                d = None if m is None else m - base_m.get(id(p), 0.0)
                deltas[(id(p), t)] = d
                if m is not None and not is_c:
                    abskey[(id(p), t)] = m
                if scored[i] is not None and not is_c:
                    p.setdefault('cand_scores', []).append(
                        dict(step=p['depth'] - 1, token=t,
                             margin=scored[i]['margin'],
                             dmargin=scored[i].get('dmargin'), delta=d,
                             canonical=False, config='S*' if S is not None else '0'))

            per = {}
            for (p, t, prefix, is_c, S, _rp) in cand:
                d = deltas.get((id(p), t))
                if is_c:
                    key = p.get('beam_score', 0.0)
                elif d is None:
                    continue
                elif a.beam_score == 'delta':
                    key = d
                elif a.beam_score == 'abs':
                    # own drift state at prefix+token+cue — the candidate's
                    # objective margin, not the parent's cumulative score
                    key = abskey.get((id(p), t))
                    if key is None:
                        continue
                else:
                    key = p.get('beam_score', 0.0) + d
                per.setdefault(p['item'], []).append((key, p, t, prefix, is_c, S))
            for ix, cs in per.items():
                iid = chunk[ix]['id']
                cs.sort(key=lambda x: -x[0])
                keep = cs[:a.beam]
                if not any(x[4] and x[1] is spines[iid] for x in keep):
                    keep = keep + [x for x in cs if x[4] and x[1] is spines[iid]]
                keep_parents = {id(x[1]) for x in keep if x[4]}
                keep_extras = {(id(x[1]), x[2]) for x in keep if not x[4]}
                spawns = []
                # Age protection against beam churn (see choice_tree.py).
                for key, p, t, prefix, is_c, S in cs:
                    if is_c:
                        if id(p) not in keep_parents and p.get('age', 0) < 3:
                            p['age'] = p.get('age', 0) + 1
                        if id(p) not in keep_parents and p.get('age', 0) >= 3:
                            p['done'] = 'beam'
                        continue
                    if (id(p), t) not in keep_extras:
                        continue
                    if limits[iid]['spawned'] >= a.max_paths:
                        limits[iid]['truncated'] = True; continue
                    limits[iid]['spawned'] += 1
                    q = spawn_path(p['item'], prefix, t, p['base_len'],
                                   depth=p['depth'])
                    # config-0 forks audit against the canonical relation,
                    # S* forks against their own perturbed relation state.
                    w, f_act, f_delta = None, p.get('_rel0', {}).get('act'), \
                        p.get('_rel0', {}).get('delta')
                    for fs in p.get('fork_steps', []):
                        if fs['step'] != p['depth'] - 1:
                            continue
                        if S is None and fs.get('witnesses', {}).get(str(int(t))):
                            w = fs['witnesses'][str(int(t))]
                            break
                        for e in fs.get('attn') or []:
                            if int(e['token']) == int(t):
                                w, f_act, f_delta = e['witness'], e['act'], e['delta']
                                break
                    q['root_score'] = deltas.get((id(p), t))
                    q['root_step'] = p['depth'] - 1
                    q['beam_score'] = key
                    q['fork_steps'] = list(p['fork_steps'])
                    q['deviations'] = list(p.get('deviations', [])) + [dict(
                        step=p['depth'] - 1, token=int(t), witness=w,
                        act=f_act, delta=f_delta,
                        config='S*' if S is not None else '0')]
                    q['ndev'] = len(q['deviations'])
                    if t in engine.eos:
                        q['done'] = 'stop'
                        q['choice'], q['text'] = canon_choice(
                            engine, q['ids'], q['base_len'], chunk[ix]['labels'])
                    elif q['depth'] >= a.max_new:
                        q['done'] = 'cap'
                    else:
                        spawns.append((q, S))
                    paths[iid].append(q)
                # batched: each spawn call computes the fork token's row under
                # the fork's config and returns the child's next-step logits.
                if spawns:
                    byS = {}
                    for q, S in spawns:
                        byS.setdefault(id(S) if S is not None else 0,
                                       (S, []))[1].append(q)
                    for _, (S, qs) in byS.items():
                        for q, lp1 in zip(qs, next_lps(engine, [q['ids'] for q in qs],
                                                       signs=S, rows=a.sign_rows)):
                            if lp1 is None:
                                q['done'] = 'relation'
                            else:
                                q['_pending_lp'] = lp1

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
        for it in chunk:
            iid = it['id']
            if sum(x['nodes'] for x in paths[iid]) >= a.node_limit:
                limits[iid]['node_limited'] = True
                for x in paths[iid]:
                    if x['done'] is None and not x['spine']:
                        x['done'] = 'nodecap'
        if rounds % 16 == 0:
            print(json.dumps(dict(round=rounds, active=len(active),
                                  paths=sum(len(paths[it['id']]) for it in chunk),
                                  nodes=sum(x['nodes'] for it in chunk
                                            for x in paths[it['id']]))), flush=True)
    else:
        for it in chunk:
            for x in paths[it['id']]:
                if x['done'] is None:
                    x['done'] = 'roundcap'
    return rounds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--thinking', choices=['on', 'off'], default='off')
    p.add_argument('--arm', choices=['attn-only', 'attn+decode', 'sampling'],
                   default='attn-only')
    p.add_argument('--mode', choices=['paper', 'free-r'], default='paper')
    p.add_argument('--objective', choices=['any', 'directed'], default='any')
    p.add_argument('--null-k', type=int, default=5,
                   help='canonical null decodes per item (coin-key-only, §2); '
                        '0 disables the null arm')
    p.add_argument('--beam', type=int, default=8)
    p.add_argument('--beam-score', choices=['delta', 'cum', 'abs'], default='cum',
                   help='abs ranks each path by its own forced-answer margin '
                        '(no cumulative drift; immune to the fork-churn '
                        'pathology on R1 — see choice_tree.py)')
    p.add_argument('--sign-scope', choices=['item', 'chunk'], default='item',
                   help='optimise one sign matrix per item, or share one per '
                        'chunk (cheaper rounds, weaker attack)')
    p.add_argument('--sign-rows', choices=['uniform', 'last'], default='uniform',
                   help="uniform: sign matrix active at every computed row "
                        "(original); last: active ONLY at each probe's "
                        "decision row — the per-position relaxation")
    p.add_argument('--opt-order', choices=['fwd', 'rev'], default='fwd')
    p.add_argument('--opt-objective', choices=['margin', 'reach'], default='margin',
                   help="what S* maximises: decision-point forced-answer margin, "
                        "or the single-step reachable token count under the "
                        "perturbed relation (fork-menu size)")
    p.add_argument('--max-new', type=int, default=1024)
    p.add_argument('--max-steps', type=int, default=2048)
    p.add_argument('--node-limit', type=int, default=40000)
    p.add_argument('--max-paths', type=int, default=4096)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--max-seqs', type=int, default=512)
    p.add_argument('--tensor-parallel', type=int, default=1,
                   help='vLLM tensor-parallel size (2 for Qwen3-32B fp32)')
    p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16',
                   help='engine dtype; float32 is the circuit-faithful setting '
                        'for the +-0.68% attention lever (bf16 cannot resolve '
                        'it — measured shifts 0 vs 0.0151)')
    p.add_argument('--enforce-eager', action='store_true', default=True,
                   help='required for the attention patch (compile bakes the '
                        'sign buffer); default on for this runner')
    p.add_argument('--no-enforce-eager', dest='enforce_eager', action='store_false')
    p.add_argument('--stop-on-flip', dest='stop_on_flip', action='store_true',
                   default=True)
    p.add_argument('--census', dest='stop_on_flip', action='store_false',
                   help='do not stop at the first flip; count the full tree')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config['backend'] = 'vllm-choice-tree-attn'
    config['rho'] = slack.RHO
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

    slack.install()
    engine = ChoiceEngine(a)
    n_attached = slack.attach(engine.llm)
    labels = items[0]['labels']
    label_ids = engine.label_ids(labels)
    n_layers = engine.llm.llm_engine.model_config.hf_config.num_hidden_layers
    n_heads = engine.llm.llm_engine.model_config.hf_config.num_attention_heads
    print(f'[attn] arm={a.arm} scope={a.sign_scope} L={n_layers} H={n_heads} '
          f'rho={slack.RHO:.6f} attached={n_attached}', flush=True)
    t0 = time.time()

    for start in range(0, len(items), a.batch):
        chunk = [it for k, it in enumerate(items) if start <= k < start + a.batch
                 and hashlib.sha256(it['id'].encode()).hexdigest()[:16] not in done]
        if not chunk:
            continue
        t_chunk = time.time()
        chats = {it['id']: engine.chat(it['question']) for it in chunk}
        prompts = [chats[it['id']] for it in chunk for _ in (0, 1)]
        seeds = [stable_seed(a.seed, it['id'], tag) for it in chunk
                 for tag in ('honest', 'control')]
        outs = engine.sample(prompts, seeds, a.max_new)
        base = {}
        for i, it in enumerate(chunk):
            (hids, hfin), (cids, cfin) = outs[2 * i], outs[2 * i + 1]
            htrail = dict(text=engine.text(hids), capped=(hfin == 'length'))
            ctrail = dict(text=engine.text(cids), capped=(cfin == 'length'))
            base[it['id']] = dict(honest=dict(htrail, choice=parse_choice(htrail,
                                                                          it['labels'])),
                                  control=dict(ctrail, choice=parse_choice(ctrail,
                                                                            it['labels'])))
        # §2 null arm: K canonical re-decodes (coin-key-only changes) under a
        # canonical sign buffer, before Phase B — shared chat prefix cache-hot,
        # and the sign state is still all-zero at this point in the chunk.
        slack.set_signs(None)
        null_raw = null_walks(engine, chats, chunk, a) if a.null_k else None
        # Phase A: canonical spines (config 0)
        spines0 = walk_spines(engine, chats, a, chunk)
        # Phase B: per-item (or per-chunk) sign matrices at the decision point
        signs_by_item = {}
        for i, it in enumerate(chunk):
            iid = it['id']
            sp = spines0[iid]
            hc = base[iid]['honest']['choice']
            if sp['done'] != 'stop' or sp['choice'] is None or hc is None:
                continue
            dt = it.get('meta', {}).get('directed_target')
            S, v = optimise_signs(engine, sp['ids'], it['labels'], hc, dt,
                                  label_ids, a, n_layers, n_heads)
            if S is not None and S.any():
                signs_by_item[iid] = S
        if a.sign_scope == 'chunk':
            # one shared matrix: the per-item argmax by objective value
            shared = {}
            for iid, S in signs_by_item.items():
                shared[id(S)] = S
            if shared:
                S0 = next(iter(shared.values()))
                signs_by_item = {iid: S0 for iid in signs_by_item}
        # Phase C: beam over paths with config-keyed menus
        paths, limits, spines = {}, {}, {}
        for i, it in enumerate(chunk):
            sp = new_path(i, chats[it['id']], len(chats[it['id']]), spine=True)
            spines[it['id']] = sp
            paths[it['id']] = [sp]
            limits[it['id']] = dict(truncated=False, node_limited=False, spawned=0)
        rounds_used = beam_search(engine, chunk, chats, base, a, paths, limits,
                                  spines, signs_by_item, label_ids, labels)
        for i, it in enumerate(chunk):
            iid = it['id']
            sp = spines[iid]
            tree = item_result(paths[iid], base[iid]['honest']['choice'],
                               limits[iid], 'beam', sp, a.beam,
                               directed_target=it.get('meta', {}).get('directed_target'))
            S = signs_by_item.get(iid)
            tree['signs'] = ([[int(li), int(s)] for li, row in
                              enumerate(S) for s in {int(x) for x in row}
                              if s != 0] if S is not None else None)
            tree['rounds'] = rounds_used
            tree['seconds'] = round((time.time() - t_chunk) / max(len(chunk), 1), 1)
            row = dict(id=iid, domain=it['domain'], question=it['question'],
                       gold_choice=it.get('gold_choice'),
                       directed_target=it.get('meta', {}).get('directed_target'),
                       honest=base[iid]['honest'], control=base[iid]['control'],
                       canonical=dict(text=sp.get('text'), capped=sp['done'] != 'stop',
                                      choice=sp['choice'], ids=sp['ids'],
                                      base_len=sp['base_len'],
                                      fork_steps=sp.get('fork_steps', [])),
                       conditions={f'{a.arm}|{a.mode}|{a.objective}':
                                   dict(choice_tree=tree)})
            if null_raw is not None:
                c0 = sp['choice']
                dt = it.get('meta', {}).get('directed_target')
                na = dict(null_raw[iid])
                na['c0'] = c0
                na['directed_target'] = dt
                # any-null = E[q_i]: per-decision flip fraction, NOT any-of-K.
                # A capped/unparsed decode cannot be certified equal to c0, so
                # it counts as != c0; undefined when c0 itself does not parse
                # (same denominator convention as choice_tree.py).
                na['any_frac'] = (sum(1 for c in na['choices'] if c != c0)
                                  / len(na['choices'])) if c0 is not None else None
                na['dir_frac'] = (sum(1 for c in na['choices']
                                      if dt is not None and c == dt)
                                  / len(na['choices']))
                na['all_stable'] = bool(c0 is not None
                                        and all(c == c0 for c in na['choices']))
                row['null_arm'] = na
            key = hashlib.sha256(iid.encode()).hexdigest()[:16]
            target = a.out / 'items' / f'{key}.json'
            tmp = target.with_suffix('.tmp')
            tmp.write_text(json.dumps(row)); tmp.replace(target)
        all_rows = [json.loads(f.read_text()) for f in (a.out / 'items').glob('*.json')]
        s = choice_summary(all_rows)
        s['phase_seconds'] = round(time.time() - t0, 1)
        s['engine_seconds'] = {k: round(v, 1) for k, v in engine.t.items()}
        (a.out / 'summary.json').write_text(json.dumps(s, indent=2))
        print(json.dumps(dict(chunk=start + len(chunk), total=len(items),
                              seconds=round(time.time() - t0, 1),
                              summary=next(iter(s['conditions'].values())))),
              flush=True)
    (a.out / 'COMPLETE').write_text(str(len(items)) + '\n')


if __name__ == '__main__':
    main()
