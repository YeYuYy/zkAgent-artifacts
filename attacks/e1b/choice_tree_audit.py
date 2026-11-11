"""Model-free witness audit for choice-tree runs.

For every recorded deviation (a non-canonical token emitted at a fork) and every
recorded canonical fork decision, rebuild the Q20 paper relation from the saved
top-40 (act, delta) state, recompute H = coin(prefix, seed), and check:

  - each emitted deviation token has a valid witness:  Relation.check(tok, w, H)
  - each canonical fork token equals Relation.canonical(H)
  - the token actually stored in the trajectory is the one being audited
  - all alternative-token witnesses recorded at that fork also check out

No model or GPU is needed. Run:
    python3 -m experiments.choice_tree_audit $ROOT/runs-choice/<...>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.relation import ALPHA, coin, masses
from experiments.vllm_bestofn import relation_from


def relation_state(act, delta):
    """Rebuild a Relation including the honest-shift fields (`relation_from`
    only restores what witnesses()/check() need)."""
    r = relation_from(act, delta)
    mx = int(r.delta.max())
    r.honest_shift = mx + int(np.floor(np.log(np.exp(ALPHA * (r.delta - mx)).sum()) / ALPHA + .5))
    r.honest_mass = masses(r.delta, r.honest_shift)
    return r


def audit_trace(trace, seed):
    ids = [int(t) for t in trace['ids']]
    base = int(trace['base_len'])
    devs = {int(d['step']): d for d in (trace.get('deviations') or [])}
    checked, failures = 0, []

    def check(ok, what, step=None):
        nonlocal checked
        checked += 1
        if not ok:
            failures.append(dict(step=step, check=what))

    for step, d in sorted(devs.items()):
        if d.get('witness') is None or d.get('act') is None:
            failures.append(dict(step=step, check='missing-witness'))
            continue
        if base + step >= len(ids):
            failures.append(dict(step=step, check='step-out-of-range'))
            continue
        H = coin(ids[:base + step], seed)
        r = relation_state(d['act'], d['delta'])
        tok = int(d['token'])
        check(bool(r.check(tok, d['witness'], H, 'paper')), 'deviation-witness', step)
        check(ids[base + step] == tok, 'deviation-token-match', step)
        if int(d['witness'].get('shift', 1 << 60)) > r.upper:
            failures.append(dict(step=step, check='shift-out-of-band'))

    for fk in trace.get('fork_steps') or []:
        step = int(fk['step'])
        if base + step >= len(ids) or fk.get('act') is None:
            failures.append(dict(step=step, check='bad-fork-record'))
            continue
        H = coin(ids[:base + step], seed)
        r = relation_state(fk['act'], fk['delta'])
        if step not in devs:
            check(r.canonical(H) == int(fk['canon']), 'canonical-token', step)
            check(ids[base + step] == int(fk['canon']), 'canonical-emitted', step)
        for t, w in (fk.get('witnesses') or {}).items():
            check(bool(r.check(int(t), w, H, 'paper')), 'alt-witness', step)
        for e in fk.get('attn') or []:
            r2 = relation_state(e['act'], e['delta'])
            check(bool(r2.check(int(e['token']), e['witness'], H, 'paper')),
                  'attn-witness', step)
    return dict(checked=checked, failures=failures)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    cfg = json.loads((a.run / 'config.json').read_text())
    seed = int(cfg['seed'])
    rows = [json.loads(f.read_text()) for f in sorted((a.run / 'items').glob('*.json'))]
    totals = dict(items=0, traces=0, checked=0, failures=0, flips=0)
    detail = []
    for row in rows:
        totals['items'] += 1
        tree = ((row.get('conditions') or {}).get('paper') or {}).get('choice_tree') or {}
        totals['flips'] += int(bool(tree.get('success')))
        traces = [('canonical', row.get('canonical') or {})]
        if tree.get('best'):
            traces.append(('best', tree['best']))
        for name, tr in traces:
            if not tr.get('ids'):
                continue
            totals['traces'] += 1
            r = audit_trace(tr, seed)
            totals['checked'] += r['checked']
            totals['failures'] += len(r['failures'])
            for f in r['failures']:
                detail.append(dict(id=row['id'], trace=name, **f))
    report = dict(run=str(a.run), **totals, detail=detail[:100])
    out = a.out or (a.run / 'audit.json')
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if totals['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()