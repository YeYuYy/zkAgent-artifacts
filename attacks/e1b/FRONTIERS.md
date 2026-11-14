# Frontiers — paper-ready results (as of 2026-11-15)

This file collects only results intended for the final paper: audited campaigns on
7B–27B models, n = 119–150, all attacks run under the **paper design** decode model
(`r = H(tr_<t) mod Θ`, the prover can only grind the committed shift ζ so that
Θ ∈ [θ ± ε]). Preliminary material (1.5B / GPT-2 smokes, random-logit yields,
bug-era numbers) is excluded — list at the end. Numbers and full tables live in
`SUMMARY.md` (section references inline); the chronological progression is in
`TIMELINE.md`.

> Scope note: the released implementation has a separate, stronger flaw (the sampling
> coin `r` is never bound to the prefix hash — "free-r"). Those results are
> implementation-specific and are **not** part of the scientific claim set here; see
> `SUMMARY.md` §1/§E1b-8B if needed.

Conventions: **eligible** = items the model answers stably (gold tasks: honestly
correct; choice tasks: canonical == honest); every ASR is a **search lower bound**
(increasing the budget can only raise it); all injection witnesses were independently
re-audited against the relation (8M+ `Relation.check` / replay checks across
campaigns, 0 failures).

---

## C0. Structural claims (the paper's backbone)

1. **zkAgent's two tolerance-checked normalizers make the proof relation set-valued**
   (E1a, real prover + real verifier, end-to-end): for the same committed logits and
   the same prefix coin, the unmodified verifier accepts two contradictory proofs —
 decode-coin re-roll accepted in 24/24 configurations, attention-row shifts in 15/15.
   The "coins cannot be re-rolled" security statement is false for the protocol's own
   relation.
2. **The Q20 relation simulation used throughout the campaigns is exactly the real
   verifier's acceptance set** (E1b-simval, on the artifact-supported GPT-2 model):
   54/54 random-logit, 54/54 scaled-real-logit, and 27/27 *unscaled* real-logit
   (±1.9e11, bound 2^46) cases match the real `proveTopKBatch`/`verifyZkDecodeProof`
   path token-for-token; deep-tail masses zero out identically; all negative controls
   rejected. Every "reachable set" number below is the real accept set, not an
   approximation.

## C1. Billing: the paper design's defense fails at 7B/8B (and any learnable value function unbounds it)

Qwen3-8B / R1-Distill-7B, GSM8K test n=150 (vLLM step-loop campaign; HF replica agrees).

| Metric (GSM8K, τ=1, paper mode) | Qwen3-8B | R1-Distill-7B |
|---|---|---|
| Reachable tokens per step (paper ζ-grind) | 1.8 | 4.9 (p90 = 12) |
| Billing attack hits the 4096 cap | 44% | **81–87%** |
| Injection rate (lexical opener policy) | 2.3% | 19.4% |
| Median inflation | 1.74× | 2.46× (lower bound, censored) |
| + 1-step-lookahead **value probe** | injection 2.3 → **22.7%** (10×) | injection 19.4 → **61.6%**, uncensored inflation **3.64×** (vs 1.0× lexical) |

- **A weak probe suffices**: the probe itself generalizes poorly (held-out R² ≤ 0.22)
  yet as a per-candidate scoring function it turns the whole reachable set into
  injection points — the attacker needs *any* learnable value function, not a good one.
- **MATH-500 (τ=1)**: R1 paper mode 98% cap-hit at 25% injection — unbounded on
  longer-CoT data too.
- **Temperature**: paper-mode injection falls with sharper τ (R1 18.9% → 2.0% at
  τ=0.6) but cap-hit only moves 81% → 68%.
- Contrast with the 1.5B-era picture ("paper inflation modest, ~1.1–1.2×, never hits
  cap"): at 7B/8B Qwen3's reflection loops amplify each injected opener into a full
  re-derivation, and R1's wider paper-reachable set admits ~8.5× more injections.

## C2. Decisions on gold-answer tasks: robust — but behaviorally, not structurally

R1-Distill-7B / Qwen3-8B × GSM8K / MATH-500 L4–5, n=150; **strict** = every token of
the flipped trajectory carries a witness.

| Attack (paper mode) | R1-Distill-7B | Qwen3-8B |
|---|---|---|
| Greedy lookahead (vLLM campaign, corrected scoring) | 8.5% (τ=1); 3.8–5.2% (τ≤0.8); MATH 2.4% | 0 |
| Best-of-N ladder @48 (hash-chain re-roll, sampled) | 1.6% lenient / **0% strict** | 0 |
| + steering, + attention forks, τ ladder, MATH-500 | lenient ≤3.9% / **strict 0 everywhere** | 0 (one 0.9% flip under attn+steer combined) |
| **Admissible tree search** (every emitted token inside its reachable set, strict by construction) | **1/131 = 0.8%** — the first fully-admissible paper-design flip at 7B; MATH 0/77 | 0 |

- The strict-audited immunity is **robust to every lever we have** (budget 1→48,
  self-correction suppression, attention forks, τ ∈ {0.6, 0.8, 1.0}, dataset switch)
  — but **not absolute**: constrained search inside the admissible set finds a flip
  that the earlier sample-continuation-then-audit protocol cannot.
- Why it holds: **model self-correction, not the relation**. Forensics: under the
  paper model only 2.5% of digit steps carry a reachable replacement digit
  (Qwen3-8B), the final-answer region is essentially locked, and completed rollouts
  self-correct 12/12 even at depth 3072. R1 (weak self-correction) is flippable
  lenient; Qwen3 (strong) is not — a family effect at equal scale, not a size effect.
- Per-item outcomes are chaotic across serving backends (HF vs vLLM share only 4
  flipped items); the structural conclusions are backend-independent.

## C3. Headline result: on gold-free agentic choice tasks, decision soundness collapses — under the paper design alone

On preference/choice tasks every honest decision is itself an attackable decision, so
the "honest-correct" eligibility filter of C2 disappears. All runs below are
**paper-relation only** (forced-answer beam, B=8; null = honest re-sample with a
different seed).

**Static choice datasets (Qwen3-8B):**

| Dataset | eligible | ASR | Random admissible trajectories | null |
|---|---|---|---|---|
| DailyDilemma short / natural | 139 / 132 | **31.6% / 81.1%** | 13.7% / 23.0% | honest sampling itself leaves 1.2–2.4% of tokens outside the reachable set |
| WebShop-choice (static proxy) short / natural | 140 / 125 | **35.0% / 85.6%** | 12.9% / 27.5% | ditto |

**Agent decision rounds at scale (natural, paper, beam8; directed = a designated
most-confusable / highest-cost distractor fixed in the data, never the gold):**

| Model · dataset | any ASR | directed ASR | null (any / dir) |
|---|---|---|---|
| Qwen3-8B · Mind2Web (70+ sites, 4-way element choice) | 75.9% (abs-scoring 91.9%) | **61.5%** — all 67 flips land on the designated target | 13.4% / 1.8% |
| Qwen3-8B · τ-bench (31 airline + 88 retail action choice) | **96.2%** (stable subset 96.5%) | **87.7%** (cancel wrong booking, $100 overpay, …) | 42.9% / 10.8% |
| R1-Distill-7B · Mind2Web | 33.3% ≈ null 29.0% (its decisions are unstable) | 39.0% | 29.0% / 7.8% |
| R1-Distill-7B · τ-bench | 46.7% | 40.0% | 22.6% / 11.4% |

- **Framing for the paper**: the paper design's decision immunity *holds on gold
  math/QA tasks and fails on preference and agentic choice tasks* — and the latter are
  exactly the trajectories zkAgent claims to protect. Directed ASR shows flips are
  aimable at a chosen high-impact action, not random drift.
- τ-bench's high null (42.9%) reflects genuine sampling variance in action choice;
  ASR still more than doubles it, and directed 87.7% vs null-dir 10.8% is unambiguous.
- Exhaustive enumeration is infeasible (median fork product 10^2.58 short / 10^19.5
  natural per item), so all numbers are lower bounds; Δ-guided beam dominates
  unguided enumeration at equal budget (finds flips at 437 vs >16,886 nodes).

## C4. The attention tolerance: an independent second lever, with corrected fp32 semantics

- **Magnitude (7B/8B, mid-CoT probes)**: adversarial per-layer sign choices move the
  top1−top2 margin by median 0.13–0.25, max 1.1–1.25; 1.3% (Qwen3-8B) / 2.8% (R1-7B)
  of steps flippable by attention alone; flip threshold ≈ margin 0.25. (On GPT-2 124M
  — the artifact's own model — threshold 0.30, 3/16 steps: validation measurement only.)
  Under the paper decode model this **enlarges the reachable set** (GPT-2: 22 → 37,
  +68%; R1-7B paper fork slots +61%).
- **Circuit semantics vs engine precision (methodology result for the evaluation
  section)**: the ±ε/θ ≈ ±0.676% row-mass band sits at bf16's ulp boundary — HF-bf16
  shifts are *exactly* 0, vLLM-bf16 shifts are kernel rounding luck. **bf16 engines
  cannot faithfully simulate this lever**; the Q20 circuit's ~1e-4 relative resolution
  resolves it fine. fp32 engines measure median shift 0.0151, matching the HF fp32
  reference 0.0127 (pearson 0.358, n=24, noise-limited positive signal across kernel
  stacks). All pre-fp32 bf16 attention ASRs must be read as "that engine + patch".
- **fp32 re-measure (WebShop natural, n=50, attn-only, any)**: **41.3%** (row-uniform
  signs; bf16 had measured 24.8% — bf16 *underestimated*); decision-row-only signs
  still 29.5% (null 12.0%) — per-position freedom is real but secondary.
- **Composition with decode (bf16 engine, labeled as such)**: R1-Distill-7B
  attn+decode (abs) reaches **96.4% / 95.1%** (any/directed) — 2.4× over either lever
  alone (attn-only 38.6/17.9%, sampling 40.4/50.8%), +67pp over null. On Qwen3-8B the
  hybrid (84.9/72.2%) only matches sampling (84.5/85.0%): the amplification is
  model-dependent. Attention forks union the decode witness sets under perturbed
  logits — the fork menu itself changes.
- **Structural boundary**: Qwen3.5's hybrid attention (24/32 GDN layers, bf16-only
  kernels) zeroes the lever under the available kernel stack — a documented honest
  boundary, not a defense claim; the decode surface is unaffected.

## C5. Size / generation scaling: the attack surface shrinks with size; generation dominates size

Mind2Web natural, n=150, paper mode, beam8, identical protocol across models.

| Model | any ASR | directed ASR | null (any / dir) | paper reachable/step (mean) |
|---|---|---|---|---|
| Qwen3.5-4B | **82.4%** | 71.6% | 19.2% / 2.0% | 3.84 |
| Qwen3.5-9B | 80.6% | **79.1%** | 13.8% / 2.6% | 2.88 |
| Qwen3.5-27B | 65.7% | 50.4% | 4.1% / 0.8% | — |
| Qwen3-8B (previous generation, control) | 75.9% | 61.5% | 13.4% / 2.6% | 1.74 |

- **Monotone shrink with size** (4B→9B→27B: 82→81→66 any, 72→79→50 directed):
  sharper distributions leave fewer reachable tokens. The null shrinks too
  (19%→4%) — but when aimed at, 27B still flips 50–66%.
- **Generation > size**: both Qwen3.5 4B and 9B exceed Qwen3-8B on reachable set and
  ASR (paper reach 3.84/2.88 vs 1.74 — new-generation architecture/calibration widens
  the sampling-tolerance surface ~3×). The R1-vs-Qwen3 contrast in C2–C4 is likewise
  a family effect at equal scale. Any "bigger model ⇒ more attackable" intuition is
  refuted within this family.

## C6. Paper-ready claim set

1. The proof relations are **set-valued** (E1a, end-to-end) and the campaigns'
   relation simulation **is** the real verifier's acceptance set (simval) — so every
   result below is about the actual protocol, not an approximation.
2. **Billing soundness fails under the paper design** at 7B/8B (81–98% cap-hit), and
   any learnable per-candidate value function — however weak — makes it unbounded
   (injection ×3–10, 3.64× uncensored inflation).
3. **Decision soundness on gold-answer tasks is behaviorally robust** (strict ASR 0
   across budget/steering/attention/τ/dataset at 7B/8B; the binding constraint is
   model self-correction) **but not structurally absolute** — admissible tree search
   finds fully-witnessed flips (0.8%).
4. **Decision soundness collapses on gold-free choice tasks** under the paper design
   alone: 31.6–96.2% ASR across DailyDilemma / WebShop / Mind2Web / τ-bench, with
   directed aiming at high-cost actions (87.7% on τ-bench).
5. The **attention tolerance is an independent second lever** — quantified magnitude,
   fp32-correct semantics (bf16 engines cannot resolve it), 2.4× composition gains,
   and a structural boundary on hybrid-attention architectures.
6. **The attack surface shrinks monotonically with model size; generation dominates
   size** — defense-by-scaling is not supported within a family.

**Global honest boundaries (attach to every number):**
- WebShop-choice / Mind2Web / τ-bench are **static decision proxies** (no interactive
  environment); R1 WebShop completion ~55% (1536-step truncation); R1 Mind2Web
  ASR ≈ null (its choices are unstable — reported, not claimed).
- All ASRs are **search lower bounds** (beam widths, budget caps, single-point S*
  optimization); the attention band is a first-order two-pole model with
  row-uniform / decision-row signs — a subset of the admissible set, hence a lower
  bound; the hybrid numbers come from a bf16 engine and are labeled as engine
  behavior pending full fp32 re-runs.
- Model coverage: Qwen3-8B, R1-Distill-7B, Qwen3.5-{4B, 9B, 27B}; n = 119–150;
  decode relation: Q20 (validated bit-exact against the C++ oracle and the real
  verifier).

---

## Explicitly excluded from the paper (kept only in TIMELINE / reports)

- All 1.5B / GPT-2 smoke ASRs (E1b-4 greedy, E1b-5 3-item smoke; the GPT-2 attention
  3/16-step measurement is cited only as artifact-model validation).
- Random-logit yields (~33/step — known overestimate).
- Beam/census numbers from the spawn-bug era (0.7% / 0% — invalidated, superseded).
- bf16-engine attention ASRs presented as circuit-semantics claims (re-labeled as
  engine behavior after the fp32 correction).
- **All free-r (released-implementation) numbers** — implementation-specific, out of
  scope for the paper's claims; archived in `SUMMARY.md`.
- Superseded early conclusions ("paper decision immunity is absolute", "attention
  correlation pearson −0.09", …).
