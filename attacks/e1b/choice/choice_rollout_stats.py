"""Search-budget curve and deviation statistics for rollout runs.

Rollout runs store one row per completed trajectory: [rollout index (-1 =
canonical spine), flip label, depth, ndev]. ASR(k) = fraction of eligible items
with a flip among the first k random trajectories (the spine is always present
and, by eligibility, never flips). ndev = number of steps that took a
non-canonical in-band token, i.e. how far the trajectory diverged from the coin
path.

Runs must be produced without --stop-on-flip (pass --census), otherwise a flip
prunes the remaining trajectories and ASR(k) is censored.

    python3 -m experiments.choice_rollout_stats RUN_DIR [RUN_DIR ...]
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def load(run):
    rows = [json.loads(f.read_text()) for f in (run / 'items').glob('*.json')]
    out = []
    for r in rows:
        t = r['conditions']['paper']['choice_tree']
        h = r['honest'].get('choice')
        elig = bool(h and r.get('canonical', {}).get('choice') == h)
        meta = [m for m in (t.get('rollout_meta') or []) if m[0] >= 0]
        out.append(dict(id=r['id'], elig=elig, meta=meta,
                        fs=sum(math.log10(len(f.get('extras', [])) + 1)
                               for f in (r.get('canonical', {}).get('fork_steps') or []))))
    return out


def budget_curve(items, ks):
    elig = [it for it in items if it['elig']]
    curve = {}
    for k in ks:
        hit = sum(any(y for i, y, _, _ in it['meta'] if y is not None and i <= k)
                  for it in elig)
        curve[k] = hit / len(elig) if elig else None
    return curve, len(elig)


def dev_stats(items):
    """Flip rate by ndev bucket and ndev distribution of flipped trajectories."""
    buckets = {'0': 0, '1': 0, '2': 0, '3-5': 0, '6+': 0}
    flips = {'0': 0, '1': 0, '2': 0, '3-5': 0, '6+': 0}
    flip_ndev, stay_ndev = [], []
    for it in items:
        if not it['elig']:
            continue
        for _, y, _, nd in it['meta']:
            if y is None:
                continue
            key = '0' if nd == 0 else '1' if nd == 1 else '2' if nd == 2 else (
                '3-5' if nd <= 5 else '6+')
            buckets[key] += 1
            flips[key] += int(y)
            (flip_ndev if y else stay_ndev).append(nd)
    return dict(rate_by_ndev={k: (flips[k] / buckets[k] if buckets[k] else None)
                             for k in buckets},
                count_by_ndev=buckets,
                flip_ndev_median=(statistics.median(flip_ndev) if flip_ndev else None),
                stay_ndev_median=(statistics.median(stay_ndev) if stay_ndev else None),
                flip_frac_zero_ndev=(sum(1 for x in flip_ndev if x == 0) / len(flip_ndev)
                                     if flip_ndev else None))


def expected_flippers(items):
    """Uniform-over-reach rollout: p_hat * feasible paths = expected number of
    flipping admissible trajectories per item (log10)."""
    out = []
    for it in items:
        if not it['elig'] or not it['meta']:
            continue
        p = sum(1 for _, y, _, _ in it['meta'] if y) / len(it['meta'])
        if p > 0:
            out.append(math.log10(p) + it['fs'])
    return dict(items=len(out),
                median_log10=(statistics.median(out) if out else None),
                min_log10=(min(out) if out else None),
                max_log10=(max(out) if out else None))


def summarize(run):
    items = load(run)
    ks = list(range(1, 9))
    curve, elig = budget_curve(items, ks)
    return dict(run=str(run), n=len(items), eligible=elig, asr_by_budget=curve,
                **dev_stats(items), expected_flippers=expected_flippers(items))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    a = p.parse_args()
    print(json.dumps([summarize(r) for r in a.runs], indent=2))


if __name__ == '__main__':
    main()