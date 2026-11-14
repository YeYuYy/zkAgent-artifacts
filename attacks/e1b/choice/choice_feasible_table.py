"""Per-question search-space size, for the "enumerate the feasible trajectories"
requirement: the product of per-step reach sizes along the canonical spine (an
order-of-magnitude width estimate, not an exact count and not a bound -- completed
census trees can be larger or smaller), the number of trajectories the search
actually explored/finished, and whether it flipped.

    python3 -m experiments.choice_feasible_table RUN_DIR [...] --out table.jsonl

One JSON object per (run, item):
  id, eligible, log10_spine_product, spine_steps, forks, searched_paths,
  terminal_paths, flips, flip_paths, nodes
"""
import argparse
import json
from pathlib import Path

from experiments.choice_report import feasible_paths


def rows(run):
    for f in sorted((run / 'items').glob('*.json')):
        r = json.loads(f.read_text())
        t = r['conditions']['paper']['choice_tree']
        h = r.get('honest', {}).get('choice')
        c = r.get('canonical', {}).get('choice')
        yield dict(run=run.name, id=r['id'], eligible=bool(h and c == h),
                   log10_spine_product=round(feasible_paths(r), 4),
                   spine_steps=t.get('spine_steps'), forks=t.get('spine_forks'),
                   searched_paths=t.get('paths_total'),
                   terminal_paths=t.get('terminal_paths'), flips=t.get('success'),
                   flip_paths=t.get('flip_paths'), nodes=t.get('nodes'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    out = [x for run in a.runs for x in rows(run)]
    text = '\n'.join(json.dumps(x, sort_keys=True) for x in out)
    if a.out:
        a.out.write_text(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()