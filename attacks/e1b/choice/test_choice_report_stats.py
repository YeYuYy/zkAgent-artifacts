"""Offline tests for the §2/§8 protocol statistics and the null-coin salting.

Run: python3 -I tests/test_choice_report_stats.py  (from the repo root).
No torch / vLLM required.
"""
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.choice_report import paired_bootstrap_ci, summarize, wilson
from experiments.choice_tree import null_salts, stable_seed
from experiments.relation import coin


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def test_wilson():
    # Ground truth by inversion: each endpoint p0 must satisfy
    # |(phat - p0)/sqrt(p0(1-p0)/n)| == z95 (checked numerically below).
    lo, hi = wilson(7, 8)
    phat, n, z = 7 / 8, 8, 1.959963984540054
    for p0 in (lo, hi):
        zsc = (phat - p0) / math.sqrt(p0 * (1 - p0) / n)
        assert abs(zsc - z) < 1e-8 or abs(zsc + z) < 1e-8, (p0, zsc)
    assert close(lo, 0.5291, 5e-4) and close(hi, 0.9776, 5e-4), (lo, hi)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and close(hi, 0.2775, 5e-4), (lo, hi)
    assert wilson(0, 0) is None
    # symmetry / monotonicity spot checks
    assert wilson(5, 10)[0] < wilson(6, 10)[0] < wilson(7, 10)[0]
    print('wilson OK', wilson(7, 8))


def test_paired_bootstrap():
    xs = [1.0] * 6 + [0.0] * 4
    ys = [0.0] * 10
    ci = paired_bootstrap_ci(xs, ys, draws=4000)
    mean = sum(x - y for x, y in zip(xs, ys)) / len(xs)
    assert ci[0] <= mean <= ci[1], (ci, mean)
    assert ci[0] > 0.1, ci          # clearly excludes zero at 6/10 vs 0/10
    again = paired_bootstrap_ci(xs, ys, draws=4000)
    assert ci == again              # fixed seed -> deterministic
    # perfect pairs -> zero-width interval
    ci0 = paired_bootstrap_ci([0.5] * 8, [0.5] * 8, draws=200)
    assert close(ci0[0], 0.0) and close(ci0[1], 0.0), ci0
    assert paired_bootstrap_ci([], []) is None
    print('paired bootstrap OK', ci)


def test_null_salts():
    a = null_salts(20261111, 'item-1', 5)
    b = null_salts(20261111, 'item-1', 5)
    assert a == b, 'salts must be deterministic'
    assert len(set(a)) == 5, 'salts must differ per decode'
    assert all(0 <= s < (1 << 62) for s in a), '62-bit domain (vLLM C-long)'
    assert null_salts(20261111, 'item-2', 5) != a, 'salts must differ per item'
    # The null coin must actually move with the key: same prefix, different
    # salts give different coins, and all differ from the attack's coin salt.
    prefix = [1, 2, 3, 4, 5]
    base = coin(prefix, 20261111)
    hs = [coin(prefix, s) for s in a]
    assert len(set(hs)) == 5 and base not in hs
    # stability of the whole chain
    assert null_salts(20261111, 'item-1', 5)[0] == stable_seed(20261111, 'item-1', 'null', 0)
    print('null_salts OK', [hex(s)[:10] for s in a])


def _row(iid, c0, distinct, na=None):
    tree = dict(success=any(c != '1' for c in distinct) and c0 is not None,
                distinct_choices=list(distinct), nodes=10, terminal_paths=1,
                flip_paths=1 if any(c != '1' for c in distinct) else 0,
                seconds=1.0, complete=True, spine_steps=5, spine_forks=2,
                paths_total=1, first_flip_depth=None, eligible=c0 is not None)
    return dict(id=iid, honest=dict(choice=c0), canonical=dict(choice=c0,
                fork_steps=[dict(extras=[9])]),
                conditions=dict(paper=dict(choice_tree=tree)), **(na or {}))


def test_summarize_fullscale():
    # 4 c0-eligible items, null arms with K=5; 2 items whose c0 does not parse
    # (any_frac undefined there, matching the harness).
    def na(choices, k=5):
        c0 = '1'
        return dict(null_arm=dict(k=k, choices=choices,
                                  any_frac=sum(1 for c in choices if c != c0) / k,
                                  dir_frac=sum(1 for c in choices if c == '3') / k,
                                  all_stable=all(c == c0 for c in choices),
                                  c0=c0, witness_failures=0))
    rows = [
        _row('i1', '1', ['2'], na(['1'] * 5)),              # flip, q=0
        _row('i2', '1', ['2'], na(['2', '2', '1', '1', '1'])),  # flip, q>0
        _row('i3', '1', ['1'], na(['3', '1', '1', '1', '1'])),  # no flip, q>0
        _row('i4', '1', ['1'], na(['1'] * 5)),              # no flip, q=0
        _row('i5', None, ['2'], na([None] * 5)),            # c0 unparseable
        _row('i6', None, ['1'], dict(null_arm=dict(k=5, choices=[None] * 5,
                                                   any_frac=None, dir_frac=0.0,
                                                   all_stable=False, witness_failures=0))),
    ]
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        (run / 'items').mkdir()
        for r in rows:
            (run / 'items' / f"{r['id']}.json").write_text(json.dumps(r))
        s = summarize(run)
    assert s['n'] == 6
    assert s['eligible_c0'] == 4 and s['flips_c0'] == 2
    assert close(s['asr_c0'], 0.5), s['asr_c0']
    assert close(s['null_any'], 3 / 20), s['null_any']       # pooled decodes
    assert close(s['null_dir'], 1 / 20), s['null_dir']
    assert close(s['asr_minus_null'], 0.5 - 0.15), s['asr_minus_null']
    ci = s['asr_minus_null_ci']
    assert ci and ci[0] < 0.35 < ci[1]
    assert close(s['stability_p_q0'], 0.5), s['stability_p_q0']
    assert close(s['asr_q0'], 0.5), s['asr_q0']               # i1 flips, i4 not
    assert s['null_any_ci'] is not None and s['asr_c0_ci'] is not None
    # old-style fields still populated
    assert s['eligible'] == 4 and s['null_flip'] == 0.0
    print('summarize full-scale OK', {k: s[k] for k in
          ('asr_c0', 'null_any', 'asr_minus_null', 'stability_p_q0', 'asr_q0')})


def test_summarize_legacy_rows():
    # Pre-§2 runs (no null_arm) must not break and must yield null_k=None.
    rows = [_row('i1', '1', ['2']), _row('i2', '1', ['1'])]
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        (run / 'items').mkdir()
        for r in rows:
            (run / 'items' / f"{r['id']}.json").write_text(json.dumps(r))
        s = summarize(run)
    assert s['null_k'] is None and s['null_any'] is None
    assert s['asr_c0_ci'] is not None and close(s['asr_c0'], 0.5)
    print('summarize legacy OK')


if __name__ == '__main__':
    test_wilson()
    test_paired_bootstrap()
    test_null_salts()
    test_summarize_fullscale()
    test_summarize_legacy_rows()
    print('ALL PASS')
