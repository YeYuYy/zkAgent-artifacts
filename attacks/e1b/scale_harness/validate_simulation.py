"""Validate the Python Q20 decode-relation simulation against the REAL
zkAgent prover+verifier (attacks/validate_simulation.cpp, built from the
released artifact).

For each logit vector the Python Relation predicts, for every in-band
committed shift zeta:
  paper mode:  r = H mod Theta routes to exactly one token;
  free-r mode: any token with mass>0 and any r inside its interval.
Negatives: mass-0 tokens, r outside the interval, out-of-band shifts.

The C++ driver runs the REAL proveTopKBatch + verifyZkDecodeProof for each
(token, zeta, r) case; the simulation is faithful iff its predictions match
the driver's ACCEPT/REJECT/PROVER_FAIL exactly.

Usage: python3 tests/validate_simulation.py [--gpt2 vector_count]
Random logits by default; --gpt2 feeds real GPT-2 124M next-token logit
vectors (needs torch+transformers, downloads openai-community/gpt2).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.relation import EPS, THETA, Relation, coin  # noqa: E402

H = int.from_bytes(b'\x5d\xee\xce\x66\xd1\x23\x45\x67', 'big')  # fixed prefix hash


def cases_for(logits_q, n_paper=6, n_freer=6, n_neg=6, seed=0):
    """(zeta, token, r, expected) cases for one quantized logit vector."""
    r = Relation(logits_q.astype(np.float64) * 7.952219286828541e-10, 1.0)
    shifts, matrix, totals = r.states()
    rng = np.random.default_rng(seed)
    out = []
    in_band = [(int(z), int(t)) for z, t in zip(shifts, totals)]

    # paper mode: coin H mod Theta -> one token per state
    idx = np.linspace(0, len(in_band) - 1, min(n_paper, len(in_band))).astype(int)
    for j, i in enumerate(np.unique(idx)):
        z, Th = in_band[i]
        m = matrix[i]
        cdf = np.cumsum(m)
        rr = H % Th
        tok = int(r.act[int(np.searchsorted(cdf, rr, side='right'))])
        out.append((z, tok, rr, True))

    # free-r mode: interval edges + middles for mass>0 tokens
    picked = 0
    for i in rng.permutation(len(in_band)):
        if picked >= n_freer:
            break
        z, Th = in_band[i]
        m = matrix[i]
        cdf = np.cumsum(m)
        cols = np.flatnonzero(m)
        if not len(cols):
            continue
        col = int(cols[rng.integers(len(cols))])
        pre = int(cdf[col - 1]) if col else 0
        nxt = int(cdf[col])
        for rr in {pre, nxt - 1, (pre + nxt) // 2}:
            out.append((z, int(r.act[col]), int(rr), True))
        picked += 1

    # negatives: (a) token with zero mass at an in-band shift, in-range r;
    #            (b) in-band shift, valid token, r OUTSIDE its interval;
    #            (c) OUT-OF-BAND shift, valid (token, r).
    for i in rng.permutation(len(in_band)):
        z, Th = in_band[i]
        m = matrix[i]
        cdf = np.cumsum(m)
        zero = [c for c in range(len(m)) if m[c] == 0]
        if zero:
            col = int(zero[rng.integers(len(zero))])
            tok = int(r.act[col]) if m.any() else col
            # non-top-k token (mass forced 0) with an in-range r
            out.append((z, tok, int(Th // 2), False))
            break
    for i in rng.permutation(len(in_band)):
        z, Th = in_band[i]
        m = matrix[i]
        cols = np.flatnonzero(m)
        if not len(cols):
            continue
        col = int(cols[0])
        pre = int(np.cumsum(m)[col - 1]) if col else 0
        tok = int(r.act[col])
        rr = (pre - 1) % Th  # strictly below the token's interval
        out.append((z, tok, int(rr), False))
        break
    # out-of-band shift: states() collapses constant regions, so +-1 past a
    # breakpoint can still be in-band — step several table-widths (2^23)
    # beyond the ends so the total genuinely leaves [theta-eps, theta+eps].
    z_lo, z_hi = int(shifts[0]), int(shifts[-1])
    m0 = matrix[0]
    c0 = int(np.flatnonzero(m0)[0]) if m0.any() else 0
    pre = int(np.cumsum(m0)[c0 - 1]) if c0 else 0
    for z in (z_lo - (1 << 25), z_hi + (1 << 25)):
        out.append((z, int(r.act[c0]), pre, False))

    return out[:n_paper + 3 * n_freer + n_neg]


def run_driver(vectors, cases, binary):
    lines = [f'{len(vectors[0])} 40', str(len(vectors))]
    for v in vectors:
        lines.append(' '.join(map(str, v.astype(np.int64))))
    lines.append(str(len(cases)))
    for vi, z, t, rr in cases:
        lines.append(f'{vi} {z} {t} {rr}')
    p = subprocess.run([binary], input='\n'.join(lines) + '\n', text=True,
                       capture_output=True, cwd=Path(binary).parent)
    if p.returncode != 0:
        print(p.stderr[-2000:])
        raise SystemExit(f'driver rc={p.returncode}')
    res = []
    for line in p.stdout.splitlines():
        f = line.split()
        if len(f) >= 5:
            res.append((int(f[0]), f[4], ' '.join(f[5:])))
    return res, p.stdout


GPT2_LOCAL = str(Path(__file__).resolve().parents[1] / 'cache/next/gpt2')


def gpt2_vectors(n, seed=0):
    """Real GPT-2 124M next-token quantized logit vectors from web text."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    src = GPT2_LOCAL if (Path(GPT2_LOCAL) / 'model.safetensors').is_file() \
        else 'openai-community/gpt2'
    tok = GPT2Tokenizer.from_pretrained(src)
    model = GPT2LMHeadModel.from_pretrained(src)
    model.eval()
    text = ('The quick brown fox jumps over the lazy dog. In a shocking finding, '
            'scientist discovered a herd of unicorns living in a remote valley. '
            'To compute the total revenue, first multiply the unit price by the '
            'quantity sold, then subtract the costs of shipping and taxes. The '
            'customer asked for a refund because the product arrived damaged.')
    ids = tok(text, return_tensors='pt').input_ids[0]
    alpha = 7.952219286828541e-10
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(ids) - 1, size=n, replace=False)
    out = []
    with torch.no_grad():
        logits = model(ids.unsqueeze(0)).logits[0]
    for j in picks:
        q = np.floor(logits[j].numpy() / alpha + 0.5).astype(np.int64)
        # Scale into the magnitude regime the released artifact's prover is
        # exercised in (E1a used +-2^17): divide by a power of two so the
        # integer grid is preserved; both simulation and driver get the SAME
        # scaled vector, so this validates the relation machinery on
        # GPT-2-shaped inputs, not the deployment quantization constant.
        scale = 1 << max(0, int(np.ceil(np.log2(max(1, np.abs(q).max() / (1 << 17))))))
        out.append((q // scale).astype(np.int64))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vectors', type=int, default=2)
    ap.add_argument('--gpt2', action='store_true')
    ap.add_argument('--gpt2-count', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20260421)
    ap.add_argument('--binary', default=str(Path(__file__).resolve().parents[1] /
                                            'zkAgent-artifacts/validate_simulation'))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    if a.gpt2:
        vectors = gpt2_vectors(a.gpt2_count)
    else:
        vectors = [rng.integers(-(1 << 17), 1 << 17, size=256) for _ in range(a.vectors)]
    cases = []
    for vi, v in enumerate(vectors):
        for z, t, rr, exp in cases_for(np.asarray(v), seed=vi):
            cases.append((vi, z, t, rr, exp))
    print(f'{len(vectors)} vectors, {len(cases)} cases '
          f'({sum(1 for c in cases if c[4])} positive / '
          f'{sum(1 for c in cases if not c[4])} negative)')
    res, stdout = run_driver([np.asarray(v) for v in vectors],
                             [(vi, z, t, rr) for vi, z, t, rr, _ in cases], a.binary)
    mism = 0
    for i, ((vi, z, t, rr, exp), (ci, verdict, msg)) in enumerate(zip(cases, res)):
        got = verdict == 'ACCEPT'
        # prover-side failure on a NEGATIVE case is also a correct rejection
        ok = (got == exp) or (not exp and verdict == 'PROVER_FAIL')
        if not ok:
            mism += 1
            print(f'MISMATCH case {i}: v{vi} zeta={z} token={t} r={rr} '
                  f'expected={"accept" if exp else "reject"} got={verdict} ({msg})')
    print(f'{len(cases) - mism}/{len(cases)} match'
          f'{" — SIMULATION FAITHFUL" if mism == 0 else " — MISMATCHES ABOVE"}')
    return 1 if mism else 0


if __name__ == '__main__':
    raise SystemExit(main())
