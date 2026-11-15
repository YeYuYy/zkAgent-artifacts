"""Summarize choice-tree runs without re-reading the model.

No-gold eligibility (the plan's definition): honest choice parses AND the
canonical trajectory parses AND the two agree. ASR = eligible items with at
least one completed admissible flip trajectory. Null flip rate = control honest
sample differs from the honest sample.

Full-scale protocol (Next-step-full-scale.md §2/§8), reported when the run
carries a null_arm: eligibility is c0-based (canonical decision parses), ASR
counts any terminal admissible path whose parsed choice != c0, null is the
per-decision flip rate of the K coin-key-only canonical re-decodes, both with
Wilson 95% CIs, ASR - null with a per-item paired bootstrap CI, plus the
stability small table (P(q-hat=0) and ASR0 on the all-stable subset).

    python3 -m experiments.choice_report RUN_DIR [RUN_DIR ...] [--out report.md]
"""
import argparse
import json
import math
import random
import statistics
from pathlib import Path

from experiments.choice_scoring import _pooled_auc

Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """Wilson score 95% interval for k successes in n trials; None if n == 0."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def paired_bootstrap_ci(xs, ys, draws=10000, seed=20261116, alpha=0.05):
    """Percentile CI for mean(xs) - mean(ys) over per-item paired samples."""
    if len(xs) != len(ys) or not xs:
        return None
    diffs = [x - y for x, y in zip(xs, ys)]
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(draws):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(math.ceil(alpha / 2 * draws)) - 1]
    hi = means[min(int(math.ceil((1 - alpha / 2) * draws)) - 1, draws - 1)]
    return (lo, hi)


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
    # --- Full-scale protocol (§2/§8): c0-based eligibility, coin-key null ---
    # c0 = the canonical spine's own decision; eligible iff it parses (no
    # honesty filter — instability is accounted by the null, not by dropping
    # items).  ASR(c0) = any terminal admissible path parsed to a choice != c0.
    nulls = [r.get('null_arm') if isinstance(r.get('null_arm'), dict) else None
             for r in rows]
    elig_c0 = [c is not None for c in canon]
    flip_c0 = [e and any(ch != c0 for ch in (t.get('distinct_choices') or []))
               for e, c0, t in zip(elig_c0, canon, trees)]
    any_na = [na.get('any_frac') if na else None for na in nulls]
    dir_na = [na.get('dir_frac') if na else None for na in nulls]
    stable0 = [bool(na.get('all_stable')) if na else None for na in nulls]
    k_dec = sum((na.get('k') or 0) for na, e in zip(nulls, elig_c0)
                if na and e and na.get('any_frac') is not None)
    k_any = sum(round((na.get('any_frac') or 0.0) * na.get('k', 0))
                for na, e in zip(nulls, elig_c0)
                if na and e and na.get('any_frac') is not None)
    # directed nulls only count where the target differs from c0 (protocol §2:
    # directed eligibility requires target != c0); otherwise landing on c0
    # would count as a directed hit while being no flip at all.
    k_dir_dec = sum((na.get('k') or 0)
                    for na, e, c0 in zip(nulls, elig_c0, canon)
                    if na and e and na.get('dir_frac') is not None
                    and na.get('directed_target') is not None
                    and na['directed_target'] != c0)
    k_dir = sum(round((na.get('dir_frac') or 0.0) * na.get('k', 0))
                for na, e, c0 in zip(nulls, elig_c0, canon)
                if na and e and na.get('dir_frac') is not None
                and na.get('directed_target') is not None
                and na['directed_target'] != c0)
    pair = [(1.0 if f else 0.0, y) for f, y, e in zip(flip_c0, any_na, elig_c0)
            if e and y is not None]
    # q-hat=0 subset: defined only on c0-eligible items with a null arm (the
    # ASR denominator); None elsewhere so it never inflates the denominator.
    q0 = [bool(s) if (s is not None and e) else None
          for e, s in zip(elig_c0, stable0)]
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
        # §2/§8 protocol columns (always computed; null columns need null_arm)
        eligible_c0=sum(elig_c0),
        flips_c0=sum(flip_c0),
        asr_c0=(sum(flip_c0) / sum(elig_c0)) if sum(elig_c0) else None,
        asr_c0_ci=wilson(sum(flip_c0), sum(elig_c0)),
        null_k=(nulls[0].get('k') if any(nulls) else None),
        null_any=(k_any / k_dec) if k_dec else None,
        null_any_ci=wilson(k_any, k_dec),
        null_dir=(k_dir / k_dir_dec) if k_dir_dec else None,
        null_dir_ci=wilson(k_dir, k_dir_dec),
        asr_minus_null=((sum(x for x, _ in pair) / len(pair)
                         - sum(y for _, y in pair) / len(pair)) if pair else None),
        asr_minus_null_ci=(paired_bootstrap_ci([x for x, _ in pair],
                                                [y for _, y in pair])
                           if pair else None),
        stability_p_q0=(sum(1 for q in q0 if q) / sum(1 for q in q0 if q is not None)
                        if any(q is not None for q in q0) else None),
        asr_q0=(sum(1 for f, q in zip(flip_c0, q0) if f and q) / sum(1 for q in q0 if q))
        if any(q for q in q0) else None,
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
        # Full-scale protocol table (§2/§8): ASR (Wilson CI) / null (Wilson CI)
        # with a paired-bootstrap ASR-null column and the stability small table.
        if any(s.get('null_k') for s in out):
            def ci(c, nd=3):
                return '—' if c is None else f'[{c[0]:.{nd}f}, {c[1]:.{nd}f}]'
            lines += ['', '## Full-scale protocol (c0 eligibility, coin-key null)',
                      '| run | n | c0-elig | ASR (95% CI) | null any (95% CI) | '
                      'null dir (95% CI) | ASR-null (95% CI) | P(q̂=0) | ASR₀ |',
                      '|---|---|---|---|---|---|---|---|---|']
            for s in out:
                lines.append(
                    f"| {Path(s['run']).name} | {s['n']} | {s['eligible_c0']} | "
                    f"{f(s['asr_c0'])} {ci(s['asr_c0_ci'])} | "
                    f"{f(s['null_any'])} {ci(s['null_any_ci'])} | "
                    f"{f(s['null_dir'])} {ci(s['null_dir_ci'])} | "
                    f"{f(s['asr_minus_null'])} {ci(s['asr_minus_null_ci'])} | "
                    f"{f(s['stability_p_q0'])} | {f(s['asr_q0'])} |")
        a.out.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()