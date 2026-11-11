"""Summarize choice-tree runs without re-reading the model.

No-gold eligibility (the plan's definition): honest choice parses AND the
canonical trajectory parses AND the two agree. ASR = eligible items with at
least one completed admissible flip trajectory. Null flip rate = control honest
sample differs from the honest sample.

    python3 -m experiments.choice_report RUN_DIR [RUN_DIR ...] [--out report.md]
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from experiments.choice_scoring import _pooled_auc


def feasible_paths(row):
    """log10 of the product of per-step reach sizes along the canonical spine.

    The spine stores one fork entry per step with |reach| > 1; every other
    step has reach size 1. This is the exact number of admissible
    trajectories through the reach sets, independent of search truncation.
    """
    fs = row.get('canonical', {}).get('fork_steps') or []
    return sum(math.log10(len(f.get('extras', [])) + 1) for f in fs)


def summarize(run):
    rows = [json.loads(f.read_text()) for f in sorted((run / 'items').glob('*.json'))]
    rows = [r for r in rows if 'conditions' in r]
    n = len(rows)
    honest = [r.get('honest', {}).get('choice') for r in rows]
    control = [r.get('control', {}).get('choice') for r in rows]
    canon = [r.get('canonical', {}).get('choice') for r in rows]
    parsed = [h is not None for h in honest]
    null = [c is not None and h is not None and c != h for c, h in zip(control, honest)]
    agree = [c is not None and h is not None and c == h for c, h in zip(control, honest)]
    elig = [p and c is not None and c == h for p, c, h in zip(parsed, canon, honest)]
    trees = [r['conditions']['paper']['choice_tree'] for r in rows if 'paper' in r['conditions']]
    flips = [t.get('success') for t in trees]
    nodes = [t.get('nodes', 0) for t in trees]
    secs = [t.get('seconds', 0) for t in trees]
    scores = [s for t in trees for s in (t.get('scores') or [])]
    beam = [t.get('beam') for t in trees if t.get('beam')]
    feasible = [feasible_paths(r) for r in rows]
    final = [f for t in trees for f in (t.get('final') or [])]
    fork_sizes = [len(f.get('extras', [])) + 1
                  for r in rows for f in (r.get('canonical', {}).get('fork_steps') or [])]
    steps = [t.get('spine_steps', 0) for t in trees]
    forks = [t.get('spine_forks', 0) for t in trees]
    elig_term = sum(t.get('terminal_paths', 0)
                    for e, t in zip(elig, trees) if e)
    elig_flip = sum(t.get('flip_paths', 0)
                    for e, t in zip(elig, trees) if e)
    stable = [e and c == h and h is not None
              for e, c, h in zip(elig, control, honest)]
    return dict(
        run=str(run), n=n,
        honest_parse=sum(parsed) / n if n else None,
        null_flip=sum(null) / sum(parsed) if sum(parsed) else None,
        control_stable=sum(a for p, a in zip(parsed, agree) if p) / sum(parsed)
        if sum(parsed) else None,
        canon_parse=sum(c is not None for c in canon) / n if n else None,
        canon_agree=sum(c is not None and c == h for c, h in zip(canon, honest))
        / sum(parsed) if sum(parsed) else None,
        eligible=sum(elig), flips=sum(1 for e, f in zip(elig, flips) if e and f),
        asr=(sum(1 for e, f in zip(elig, flips) if e and f) / sum(elig)) if sum(elig) else None,
        stable_eligible=sum(stable),
        flips_stable=sum(1 for st, f in zip(stable, flips) if st and f),
        asr_stable=(sum(1 for st, f in zip(stable, flips) if st and f) / sum(stable))
        if sum(stable) else None,
        flip_paths=sum(t.get('flip_paths', 0) for t in trees),
        terminal_paths=sum(t.get('terminal_paths', 0) for t in trees),
        path_flip_rate=(sum(t.get('flip_paths', 0) for t in trees)
                        / sum(t.get('terminal_paths', 0) for t in trees))
        if sum(t.get('terminal_paths', 0) for t in trees) else None,
        eligible_path_flip_rate=(elig_flip / elig_term) if elig_term else None,
        nodes_median=statistics.median(nodes) if nodes else None,
        nodes_max=max(nodes) if nodes else None,
        seconds_median=statistics.median(secs) if secs else None,
        complete=sum(bool(t.get('complete')) for t in trees),
        beam_width=max((b.get('width', 0) for b in beam), default=None),
        auc=_pooled_auc(beam) if beam else (_pooled_auc([dict(scores=scores)]) if scores else None),
        final_auc=_pooled_auc([dict(scores=[[m, y] for _, m, y in final])]) if final else None,
        log10_paths_median=statistics.median(feasible) if feasible else None,
        log10_paths_max=max(feasible) if feasible else None,
        searched_paths=sum(t.get('paths_total', 0) for t in trees),
        spine_steps_median=statistics.median(steps) if steps else None,
        fork_rate=(sum(forks) / sum(steps)) if sum(steps) else None,
        reach_at_fork_median=statistics.median(fork_sizes) if fork_sizes else None,
        reach_hist={k: fork_sizes.count(k) for k in sorted(set(fork_sizes))},
        choice_dist={k: sum(1 for h in honest if h == k) for k in sorted({h for h in honest
                                                                        if h is not None})},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    out = [summarize(r) for r in a.runs]
    print(json.dumps(out, indent=2))
    if a.out:
        lines = ['| run | n | honest | null | canon | agree | elig | flips | ASR | ASR-st | AUC | seconds |',
                 '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for s in out:
            def f(x, nd=3):
                return '—' if x is None else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))
            lines.append(f"| {Path(s['run']).name} | {s['n']} | {f(s['honest_parse'])} | "
                         f"{f(s['null_flip'])} | {f(s['canon_parse'])} | {f(s['canon_agree'])} | "
                         f"{s['eligible']} | {s['flips']} | {f(s['asr'])} | {f(s['asr_stable'])} | "
                         f"{f(s['auc'])} | {f(s['seconds_median'], 1)} |")
        a.out.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()