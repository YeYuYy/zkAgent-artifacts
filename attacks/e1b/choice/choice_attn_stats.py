"""Attention-axis summary: paper reach vs paper reach + attention perturbations.

Reads runs from experiments/choice_tree_hf.py (same item schema as the vLLM
runner). Reports the no-gold eligibility/ASR numbers next to the reach
expansion: how many steps were probed, how many extra tokens the three
per-layer attention scalings added, and how often they created a new fork.

    python3 -m experiments.choice_attn_stats RUN_DIR [RUN_DIR ...]
"""
import argparse
import json
import statistics
from pathlib import Path

from experiments.choice_report import summarize


def attn_stats(run):
    rows = [json.loads(f.read_text()) for f in sorted((run / 'items').glob('*.json'))]
    trees = [r['conditions']['paper']['choice_tree'] for r in rows]
    stats = [t.get('attn') or {} for t in trees]
    items_extra = sum(1 for s in stats if s.get('extra_tokens'))
    tokens = [s.get('extra_tokens', 0) for s in stats]
    forks = [s.get('extra_forks', 0) for s in stats]
    probed = [s.get('steps_probed', 0) for s in stats]
    return dict(
        items_with_extra=items_extra,
        extra_tokens_total=sum(tokens),
        extra_tokens_median=statistics.median(tokens) if tokens else None,
        extra_tokens_max=max(tokens) if tokens else None,
        extra_forks_total=sum(forks),
        extra_forks_median=statistics.median(forks) if forks else None,
        steps_probed_total=sum(probed),
        extra_per_100_probes=(100 * sum(tokens) / sum(probed)) if sum(probed) else None,
        flips_among_extra=sum(1 for r, s in zip(rows, stats)
                              if s.get('extra_tokens')
                              and r['conditions']['paper']['choice_tree'].get('success')),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    a = p.parse_args()
    out = []
    for run in a.runs:
        s = summarize(run)
        out.append(dict(run=run.name, **{k: s[k] for k in (
            'n', 'honest_parse', 'null_flip', 'canon_agree', 'eligible', 'flips',
            'asr', 'asr_stable', 'stable_eligible', 'terminal_paths', 'nodes_median')},
            **attn_stats(run)))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()