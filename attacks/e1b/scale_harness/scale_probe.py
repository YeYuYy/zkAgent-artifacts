"""Scaling probe: per-model attention-lever magnitude + decode reachable set.

Model-size campaign Phase A (attn & sample surfaces measured independently):
  - sample surface: per-step |witnesses(paper)| and |witnesses(free-r)| along
    the canonical spine of Mind2Web decision rounds (no attack, measurement);
  - attn surface:   top1-top2 margin shift at [prefix+cue] under coherent+ and
    random per-(layer,head) sign matrices, row-uniform AND last-row-only
    (per-position) application.

Runs on vLLM (any model the installed vLLM supports; Qwen3.5 needs the
py312-vllm35 tree via PYTHONPATH). fp32 + eager for attention semantics.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from experiments import attn_slack_vllm as slack
from experiments.choice_tree import ChoiceEngine, lp_dict, stable_seed
from experiments.relation import coin
from experiments.vllm_bestofn import relation_from_logprobs

CUE = '\n\nFinal answer: \\boxed{'


class ProbeEngine(ChoiceEngine):
    pass


def next_lps(engine, prompts, signs=None, rows='uniform'):
    sp = engine.SP(temperature=0.0, max_tokens=1, logprobs=64, detokenize=False)
    if signs is None:
        slack.set_signs(None)
    elif rows == 'last':
        slack.set_signs(signs, positions=[len(p) - 1 for p in prompts])
    else:
        slack.set_signs(signs)
    try:
        outs = engine.llm.generate([{'prompt_token_ids': p} for p in prompts],
                                   [sp] * len(prompts), use_tqdm=False)
    finally:
        slack.set_signs(None)
    return [lp_dict(o.outputs[0].logprobs[0]) if o.outputs[0].logprobs else None
            for o in outs]


def margins(lp):
    if lp is None or len(lp) < 2:
        return None
    top = sorted(lp.items(), key=lambda kv: -kv[1])[:2]
    return top[0][1] - top[1][1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--probe-every', type=int, default=12)
    p.add_argument('--rand-patterns', type=int, default=4)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--thinking', choices=['on', 'off'], default='off')
    p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='float32')
    p.add_argument('--enforce-eager', action='store_true', default=True)
    p.add_argument('--max-seqs', type=int, default=64)
    p.add_argument('--max-model-len', type=int, default=8192)
    a = p.parse_args()

    slack.install()
    engine = ProbeEngine(a)
    n_attached = slack.attach(engine.llm)
    items = [json.loads(x) for x in Path(a.data).read_text().splitlines()][:a.limit]
    labels = items[0]['labels']
    label_ids = engine.label_ids(labels)
    try:
        n_layers = engine.llm.llm_engine.model_config.hf_config.num_hidden_layers
    except Exception:
        n_layers = engine.llm.llm_engine.model_config.hf_config.text_config.num_hidden_layers
    try:
        n_heads = engine.llm.llm_engine.model_config.hf_config.num_attention_heads
    except Exception:
        n_heads = engine.llm.llm_engine.model_config.hf_config.text_config.num_attention_heads
    print(f'[probe] L={n_layers} H={n_heads} attached={n_attached} '
          f'dtype={a.dtype}', flush=True)
    rng = np.random.default_rng(7)
    S_rands = [rng.integers(-1, 2, size=(n_layers, n_heads))
               for _ in range(a.rand_patterns)]
    S_coh = np.ones((n_layers, n_heads), dtype=int)

    out = dict(model=a.model, n_layers=n_layers, n_heads=n_heads,
               dtype=a.dtype, items=len(items), reach_paper=[], reach_free=[],
               reach_paper_attn=[], margin_shift={})
    for name in ('coherent+', 'random', 'random_last'):
        out['margin_shift'][name] = []
    t0 = time.time()

    chats = {it['id']: engine.chat(it['question']) for it in items}
    spines = {it['id']: list(chats[it['id']]) for it in items}
    active = {it['id']: True for it in items}
    steps_done = 0
    while steps_done < a.steps and any(active.values()):
        steps_done += 1
        act = [(iid, ids) for iid, ids in spines.items() if active[iid]]
        lps = next_lps(engine, [ids for _, ids in act])
        for (iid, ids), lp in zip(act, lps):
            if lp is None:
                active[iid] = False
                continue
            H = coin(ids, a.seed)
            try:
                r = relation_from_logprobs(lp, engine.V, a.temperature)
                out['reach_paper'].append(len(r.witnesses(H, 'paper')))
                out['reach_free'].append(len(r.witnesses(H, 'free-r')))
            except ValueError:
                pass
            tok = None
            try:
                tok = r.canonical(H)
            except Exception:
                active[iid] = False
                continue
            if tok is None:
                active[iid] = False
                continue
            spines[iid] = ids + [int(tok)]
            if tok in engine.eos:
                active[iid] = False
        # attention probes at the current decision point
        if steps_done % a.probe_every == 0:
            probe_ids = {it['id']: spines[it['id']] for it in items}
            for it in items:
                ids = probe_ids[it['id']]
                cue_ids = engine.tok.encode(CUE, add_special_tokens=False)
                prompt = ids + cue_ids
                base = margins(next_lps(engine, [prompt])[0])
                if base is None:
                    continue
                probes = [('coherent+', S_coh, 'uniform')]
                probes += [(f'random{k}', S_rands[k], 'uniform')
                           for k in range(a.rand_patterns)]
                probes += [(f'random_last{k}', S_rands[k], 'last')
                           for k in range(min(2, a.rand_patterns))]
                for name, S, rows in probes:
                    m = margins(next_lps(engine, [prompt], signs=S, rows=rows)[0])
                    if m is not None:
                        key = 'random' if name.startswith('random') and rows == 'uniform' \
                            else 'random_last' if name.startswith('random') else 'coherent+'
                        out['margin_shift'][key].append(m - base)
                # reachable under one random attn config (attn+sample union view)
                lp_a = next_lps(engine, [ids], signs=S_rands[0])[0]
                if lp_a is not None:
                    try:
                        r2 = relation_from_logprobs(lp_a, engine.V, a.temperature)
                        out['reach_paper_attn'].append(len(r2.witnesses(coin(ids, a.seed), 'paper')))
                    except ValueError:
                        pass
    stats = {}
    for k, v in out['margin_shift'].items():
        av = [abs(x) for x in v if x == x]
        stats[k] = dict(n=len(av), median=statistics.median(av) if av else None,
                        p90=sorted(av)[int(0.9 * len(av))] if av else None,
                        max=max(av) if av else None)
    out['reach'] = dict(
        paper=dict(n=len(out['reach_paper']),
                   mean=statistics.mean(out['reach_paper']) if out['reach_paper'] else None,
                   median=statistics.median(out['reach_paper']) if out['reach_paper'] else None),
        free=dict(n=len(out['reach_free']),
                  mean=statistics.mean(out['reach_free']) if out['reach_free'] else None,
                  median=statistics.median(out['reach_free']) if out['reach_free'] else None),
        paper_attn=dict(n=len(out['reach_paper_attn']),
                        mean=statistics.mean(out['reach_paper_attn']) if out['reach_paper_attn'] else None))
    out['margin_stats'] = stats
    out['seconds'] = round(time.time() - t0, 1)
    out.pop('reach_paper'); out.pop('reach_free'); out.pop('reach_paper_attn')
    out.pop('margin_shift')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
