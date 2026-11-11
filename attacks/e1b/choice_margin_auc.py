"""How early does the forced-answer margin predict a trajectory's decision?

Reads runs produced with --search rollout --record-path-margin. Each terminal
trajectory stores [(depth, margin), ...] from probing `prefix + cue` after every
step; the canonical spine stores the same trace, so both the absolute margin
and the delta against the canonical continuation at the same depth can be
scored. Label = whether the trajectory's final choice differs from honest.

    python3 -m experiments.choice_margin_auc RUN_DIR [--out out.json]
"""
import argparse
import json
from pathlib import Path

from experiments.choice_scoring import _auc

DEPTH_EDGES = {'0-5': (0, 5), '5-10': (5, 10), '10-20': (10, 20), '20-40': (20, 40),
               '40-80': (40, 80), '80+': (80, 10 ** 9)}
REL_EDGES = {'last1': (0, 1), 'last2-5': (1, 5), 'last5-10': (5, 10),
             'last10-20': (10, 20), 'last20+': (20, 10 ** 9)}


def collected(run):
    """(abs_margin, delta, depth, distance_from_end, label) per measurement."""
    out = []
    for f in sorted((run / 'items').glob('*.json')):
        r = json.loads(f.read_text())
        t = r['conditions']['paper']['choice_tree']
        spine = {d: m for d, m in (t.get('spine_margins') or [])}
        for y, tr in t.get('traces') or []:
            if tr == (t.get('spine_margins') or []):
                continue  # the canonical spine is the reference, not a sample
            for i, (d, m) in enumerate(tr):
                out.append((m, m - spine[d] if d in spine else None,
                            d, len(tr) - 1 - i, y))
    return out


def pool(rows, col, key, edges):
    out = {name: ([], []) for name in edges}
    for r in rows:
        v = r[col]
        if v is None:
            continue
        k = key(r)
        for name, (lo, hi) in edges.items():
            if lo <= k < hi:
                out[name][0 if r[4] else 1].append(v)
                break
    return {name: dict(auc=_auc(pos, neg), pos=len(pos), neg=len(neg))
            for name, (pos, neg) in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    rows = collected(a.run)
    res = dict(run=str(a.run), probes=len(rows),
               flip_probes=sum(1 for r in rows if r[4]),
               overall_abs=pool(rows, 0, lambda r: 0, {'all': (0, 1)}),
               overall_delta=pool(rows, 1, lambda r: 0, {'all': (0, 1)}),
               by_depth_abs=pool(rows, 0, lambda r: r[2], DEPTH_EDGES),
               by_depth_delta=pool(rows, 1, lambda r: r[2], DEPTH_EDGES),
               by_end_abs=pool(rows, 0, lambda r: r[3], REL_EDGES),
               by_end_delta=pool(rows, 1, lambda r: r[3], REL_EDGES))
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()