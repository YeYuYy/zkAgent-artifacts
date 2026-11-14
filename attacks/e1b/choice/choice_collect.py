"""Collect E1b-choice run summaries into one comparison table.

Usage: python3 -m experiments.choice_collect <run-dir> [<run-dir> ...]
Each run dir must contain summary.json (from choice_tree / choice_tree_attn).
Prints one row per run with any-flip / directed ASR and baselines.
"""
import json
import sys
from pathlib import Path


def row(path):
    s = json.loads((path / 'summary.json').read_text())
    conds = s.get('conditions') or {}
    name = f'{path.parent.name}/{path.name}'
    out = dict(run=name, n=s.get('n'), parsed=s.get('honest_parse_rate'),
               null=s.get('null_flip_rate'),
               stable=s.get('stable_rate'))
    for mode, c in sorted(conds.items()):
        d = (c or {})
        # summary.json stores the per-mode dict flat; per-item rows nest it
        # under 'choice_tree' — accept either shape.
        if 'eligible' not in d:
            d = d.get('choice_tree') or {}
        if not d:
            continue
        out.setdefault('mode', mode)
        out.update(eligible=d.get('eligible'), flips=d.get('flips'),
                   asr=d.get('asr_eligible'),
                   asr_stable=d.get('asr_stable'),
                   nodes_med=d.get('nodes_median'),
                   secs_med=d.get('seconds_median'),
                   directed_elig=d.get('directed_eligible'),
                   directed_hits=d.get('directed_hits'),
                   asr_directed=d.get('asr_directed'),
                   null_directed=d.get('null_directed'),
                   rollout_directed=d.get('rollout_directed'))
        if 'rollout' in path.name:
            out['rollout_arm'] = 'yes'
    return out


def main():
    keys = ['run', 'mode', 'n', 'parsed', 'eligible', 'flips', 'asr', 'asr_stable',
            'directed_elig', 'directed_hits', 'asr_directed', 'null',
            'null_directed', 'rollout_directed', 'nodes_med', 'secs_med']
    rows = [row(Path(p)) for p in sys.argv[1:]]
    w = {k: max(len(k), max((len(str(r.get(k))) for r in rows), default=0)) for k in keys}
    print('  '.join(k.ljust(w[k]) for k in keys))
    for r in rows:
        print('  '.join(str(r.get(k, '')).ljust(w[k]) for k in keys))


if __name__ == '__main__':
    main()
