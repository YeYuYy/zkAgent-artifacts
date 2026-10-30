# E1a — zkAgent's tolerance-checked normalizers are set-valued

This directory verifies, against the **released artifact** (GPT-2 124M, `Q = 20`,
commit `af36422`), the decision-integrity gap described in *Decision Integrity
for Verifiable LLM Inference*.

## Claim under test

zkAgent proves the whole transcript including decoding, but both its softmax
normalizers commit a **prover-chosen shift `ζ`** and only range-check the
resulting mass against a public band `[θ − ε, θ + ε]`:

* **Attention** (`src/model/attention_softmax.cpp`, every head/layer/position):
  the released verifier message is *"attention row mass exceeds its public
  tolerance"*; the row output is divided by the committed `ζ`-dependent mass.
* **Decode, top-k/top-p** (`src/zkdecode/…`): the sampling coin is
  `r = H(tr_<t) mod Θ`, where `Θ` is the committed total mass. The verifier
  (`check_decode_sampling_claims`) only checks `Θ ∈ [θ − ε, θ + ε]`.

Because `Θ` (resp. the attention row mass) is a deterministic function of the
committed `ζ`, **each admissible `ζ` is a different accepted outcome for the same
committed input**. The proof relation is therefore a *set*, not a function:
`|H(x)| > 1`, so decision integrity fails even though every transcript verifies.

This contradicts zkAgent's own statement that *"because r is a function of the
prefix alone, the prover cannot re-roll it after observing the logit and thus
cannot bias the sampled token."*

## What the PoCs do

Nothing in the **verifier** is changed. The only two source edits are
prover-side test instruments that let a caller override the committed shift `ζ`;
the verifier never reads them and honest behavior is unchanged when they are
`nullptr`:

| File | Instrument |
|---|---|
| `src/zkdecode/zkdecode_exp_digits.{hpp,cpp}` | `decode_exp_digits_shift_override` |
| `src/model/attention_softmax.{hpp,cpp}` | `attention_softmax_zeta_override` |

* **`setvalued_decode.cpp`** — runs the **real prover + real verifier**
  (`proveTopKBatch` / `verifyZkDecodeProof`) twice for the *same* committed
  logits and *same* prefix hash `H`, with two different committed `ζ`. It reports
  that both proofs are accepted, that they commit to different `Θ`, and that the
  fixed coin `r = H mod Θ` is routed to different tokens (a genuine decision
  flip). It also enumerates how many distinct in-band `Θ` / reachable tokens
  exist.
* **`setvalued_attention.cpp`** — calls the released acceptance predicate
  `verify_attention_softmax_row` and shows it accepts many distinct `ζ` for one
  score row, each a different verified attention-row output.

## Build

```bash
# from the repository root, after the normal build (see top-level README)
./build.sh Release                       # builds cmake-build-release/src/libgpt_lib.a

LIB=cmake-build-release
g++ -std=c++14 -O2 -fopenmp -mcmodel=large -Isrc -I3rd -I3rd/mcl/include \
    attacks/setvalued_decode.cpp  $LIB/src/libgpt_lib.a $LIB/lib/libmcl.a \
    -lgmp -lcrypto -o setvalued_decode
g++ -std=c++14 -O2 -fopenmp -mcmodel=large -Isrc -I3rd -I3rd/mcl/include \
    attacks/setvalued_attention.cpp $LIB/src/libgpt_lib.a $LIB/lib/libmcl.a \
    -lgmp -lcrypto -o setvalued_attention
```

## Run

```bash
./setvalued_decode   --vocab 256 --k 40 --seed 1 --backend logup-gkr   # or --backend shout
./setvalued_attention --seed 1 --row-len 8
```

`--backend` is accepted but irrelevant to the gap: the band check lives in
`check_decode_sampling_claims`, which both lookup backends share.

## Results (this artifact, `Q = 20`)

Band: `θ = 2^20 = 1048576`, `ε = 7088` (the released
`decode_exp_digits_recommended_epsilon` default for `k = 40`), so
`ε/θ = 0.68%` — matching the ~0.7% per-row slack in the writeup.

**Decode, end to end.** For one committed logit vector and one fixed prefix
hash, the unmodified verifier accepts two transcripts that commit to different
`Θ` and sample different tokens. Example (`seed 1, vocab 256, k 40`):

```
committed zeta=4638932187 Theta=1051296 token=5  -> verifier ACCEPTS
committed zeta=4638946287 Theta=1049184 token=12 -> verifier ACCEPTS
```

**Robustness.** 24/24 configurations demonstrate a flip — every
`seed ∈ {1,2,7}` × `(vocab,k) ∈ {(256,40),(512,40),(1024,32),(128,20)}` × both
backends. Per configuration, 19–34 distinct tokens are reachable for a single
fixed coin at a coarse `ζ` step (the full band holds ~`2ε = 14176` integer `Θ`,
so this is a lower bound).

**Attention.** `verify_attention_softmax_row` accepts thousands of `ζ` per row,
each a distinct verified output, for every `seed ∈ {0,1,2,7,99}` and
`row_len ∈ {4,8,16}`.

## Scope / caveats

* The gap is specific to **sampling** decode (top-k/top-p) and to **attention**
  (all modes). **Greedy** decode is *not* steerable this way: the released
  verifier forces `θ = ε = 0` for greedy (`zkdecode_verifier.cpp`), so there is
  no mass band — `"greedy parameters contain sampling fields"`.
* `k` is capped at the released `max_active_count = 40`; `k > 40` is rejected by
  the honest witness builder and is not a valid instance.
* These PoCs target the C++ proof artifact only. The quantization/PPL
  experiments (paper Table 9 / E4) are unrelated to this attack and need
  PyTorch, which is out of scope here.
