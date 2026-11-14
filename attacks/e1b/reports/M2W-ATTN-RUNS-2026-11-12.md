# Mind2Web choice + attention-lever campaign (2026-11-12) — COMPLETE 2026-11-13

Final numbers and narrative: `zkAgent-artifacts/attacks/e1b/SUMMARY.md`
§E1b-choice-m2w. Highlights: Mind2Web directed 61.5% (Qwen3, all flips on
target); R1 hybrid (attn+decode, abs) 96.4%/95.1% on WebShop natural —
attention+decode is a 2.4× amplifier over either lever alone on R1;
attn-only ≈ sampling on R1 (38.6% vs 40.4%) but ≪ sampling on Qwen3
(24.8% vs 84.5%). Beam-churn pathology found & fixed (abs scoring).
All runs audited (choice_tree_audit, ~4.3M checks this campaign, 0 failures).

Run inventory (`$ROOT/runs-choice/`):
- qwen3: m2w-natural-beam8-cum-{any,directed}, m2w-natural-beam8-abs-any,
  m2w-natural-rollout, ws-natural-beam8-cum-{any,directed}-v3,
  ws-natural-attn-only-{any,directed}, ws-natural-attn+decode-{any,directed},
  ws-natural-attnhyb-abs-any
- r1: m2w-natural-beam8-abs-{any,directed}, m2w-natural-rollout,
  ws-natural-beam8-abs-{any,directed}, ws-natural-attn-only-{any,directed},
  ws-natural-attnhyb-abs-{any,directed}
- attn-validate-{clean,cached,clean-compile,final}.json

Two new experiment lines on the E1b-choice infrastructure. Both H200 workers
relaunched (w1 = original queue `$ROOT/worker`, w2 = second queue
`$ROOT/w2/worker` with symlinked env/models/data/snapshots; both vLLM image
`registry.h.pjlab.org.cn/ailab-llmapi/vllm-openai:v0.25.0`).

## 1. Mind2Web natural choice (repeat of DD/WebShop protocol)

Dataset `data/mind2web-choice-150.jsonl` (`experiments/choice_data.py::
load_mind2web`, build: `experiments/build_choice_data.py --dataset mind2web
--parquet cache/next/mind2web/train_*.parquet`, HF `osunlp/Mind2Web` train
parquet, 409 annotations):

- One decision round per annotation: richest step (most interactive neg
  candidates); **history is fixed in the prompt** (last ≤8 action reprs) so
  every generated token belongs to the round under attack.
- 4-way element choice: target element + 3 same-page distractors (interactive
  tags first, ranked same-tag-then-task-overlap); element text from
  `cleaned_html` via bs4 (subtree text, placeholder/value/id fallback).
- `meta.directed_target` = most confusable distractor (same tag first, max
  task-word overlap) — designated in the data, never the gold.
- gold recorded (never used for eligibility/ASR). 70+ websites, ops
  142 CLICK / 5 TYPE / 3 SELECT, balanced label + directed dists.

Harness (`experiments/choice_tree.py`): new `--objective {any,directed}`
(directed beam maximises `dmargin = log p(target) − log Σ p(other)` at the
forced-answer cue; `directed_success` = terminal choice == target),
`--mode {paper,free-r}` plumbing, directed fields in scoring/rollout.
Smokes: Qwen3 any 0.5 / directed 1.0 (2 eligible), R1 template OK (50 s/item).

Runs (w1, beam8 cum, natural, n=150, τ=1): qwen3/r1 × {any, directed} +
rollout baselines (7 random admissible trajectories, no early stop).
WebShop sampling v3 controls on `webshop-choice-150-directed.jsonl`
(directed targets annotated: max instruction-overlap non-gold option):
qwen3/r1 × {any, directed} — the sampling baseline for the attention arms.

## 2. Attention lever at vLLM speed

Patch `experiments/attn_slack_vllm.py`: monkeypatches
`Qwen2/3Attention.forward` (vLLM 0.25.0) to scale every attention output row
computed in the current forward batch by `1 + rho*sign[layer, head]`,
`rho = eps/theta = 7088/2^20` (the E1b-attn two-pole band model; scaling the
row output == choosing an in-band row mass Theta_row; o_proj linear ⇒ pre/post
equivalent). Signs live in ONE pre-allocated CUDA buffer updated in place ⇒
decode CUDA graphs captured at init stay valid (multiply always executes,
zeros = canonical). **Must install before LLM()** (capture happens at init).

Semantics: rows already in the prefix cache are never recomputed, so each
row's perturbation is frozen when the row is computed — a per-row admissible
zeta choice (row-uniform sign pattern = a subset). Cached-prefix rows keep
canonical zeta.

Runner `experiments/choice_tree_attn.py`:
- Phase A canonical spine (config 0) → Phase B per-item sign matrix S* =
  layer-granularity coordinate ascent on the forced-answer objective at
  [spine+cue] (`--sign-scope item|chunk`) → Phase C beam with config-keyed
  fork menus.
- Arms: `attn-only` (menu = canonical(H, logits(0)) ∪ canonical(H,
  logits(S*)); decode strictly canonical — the pure attention question) and
  `attn+decode` (union of paper witness sets under both configs).
- Cache protocol: S*-menu probes ([prefix] under S*, last row always
  recomputed per kv_cache_manager) run FIRST each round; the canonical call
  restores the row under 0. Fork tokens are deduped to one config; spawn
  calls compute the fork row under its config and hand the child its
  next-step logits (`_pending_lp`), so the child's continuation reads the
  committed fork row. Auditable: S* forks carry their own perturbed
  (act, delta) in fork `attn` entries + deviations (choice_tree_audit
  replays them; `attn-witness` checks).
- Validation `experiments/attn_slack_validate.py`: vLLM patch vs HF eager
  per-(layer,head) hook (attn_slack_logits.py convention), coherent+ /
  random patterns: perturbation-delta correlation, top-1 agreement, and
  cache-restoration exactness (probe under S then 0 ⇒ identical logits).

Plan: validate (w2) → ws attn-only smokes (w2) → full WebShop natural runs
(both models × {attn-only, attn+decode} × {any, directed}) → compare
ASR(attn-only) vs ASR(sampling paper) vs ASR(hybrid) vs null/random baselines.

## Honest boundaries

- Static element/product choice proxies (no interactive environment).
- Attention band = first-order two-pole model (per-element rounding not
  modelled) ⇒ lower bounds; sign patterns are row-uniform (subset of
  per-row admissible choices); S* optimized at the decision point then
  applied at each step (heuristic, not per-step optimal).
- WebShop attn runs use row-uniform signs; A0's adversarial per-row signs
  were stronger per probe.
