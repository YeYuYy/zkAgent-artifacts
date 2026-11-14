# Next steps — four lines (2026-11-13, user-directed)

Continuation of the E1b-choice-m2w campaign (SUMMARY.md §E1b-choice-m2w,
commit e6d5a02). Workers stopped; relaunch via infrastructure/gpu/launch.py
(+ `--root $R/w2` for the second queue).

## 1. Attention-algorithm refinement

1a. **vLLM vs HF numerical consistency** (the failed correlation, pearson
    −0.09 in attn-validate-final.json). Isolate precision vs semantics:
    run the HF reference in **bf16** (matching vLLM) — if correlation
    recovers, the fp32-vs-bf16 gap explains it (document); if not, there is
    a semantic mismatch between the vLLM patch (post-`self.attn`, pre-o_proj)
    and the HF eager hook — bisect with single-(layer,head) perturbations.

1b. After 1a: **finer-grained manipulation** in the same hybrid architecture —
    relax the row-uniform restriction: per-position signs within one forward.
    Implementation: signs buffer [L, H, max_pos], scale gathered by the
    forward's `positions` tensor (available in Qwen2/3Attention.forward).
    Search: per-fresh-row sign vectors (e.g., coordinate ascent on the
    decision row only; earlier fresh rows canonical).

## 2. Simulation ↔ real zkAgent verifier validation (GPT-2)

All E1b attacks run a Python simulation of the Q20 decode relation
(experiments/relation.py) instead of the real ZK protocol. Validate the
simulation on the ONE model the real artifact supports (GPT-2 124M):
for real GPT-2 logits, compare the Python Relation's witness set (tokens +
(shift, total, r, prefix) tuples) against the artifact's real
proveTopKBatch/verify path (attacks/setvalued_decode.cpp pattern): every
Python-admissible (token, witness) must be ACCEPTED by the real verifier,
every Python-inadmissible one REJECTED. Note tests/relation_oracle.cpp +
tests/check_cpp_oracle.py already exist — check what they cover (E1b-8B used
a C++ oracle for bitwise alignment of enumeration) and extend to the real
artifact verifier if needed.

## 3. τ-bench attacks (7B/8B models)

Same choice-attack protocol on τ-bench (customer-service tool-use; decisions
with monetary/action impact — refunds, bookings). Goal: pick high-impact
examples. Static proxy first (conversation state + action menu → choose),
harness = choice_tree (sample arm) + choice_tree_attn (attn arms).

## 4. Model-size scaling: Qwen3.5 4B / 9B / 27B (+ R1 series)

Models at /mnt/shared-storage-user/ai4good1-share/guodadi/mas_security_h200/models
(check inventory; Qwen3.5 attention structure may differ from Qwen3 —
e.g. Qwen3Next-class archs need new patch coverage in attn_slack_vllm.py).

Phase A (measurement, Mind2Web or similar): per model size —
  - margin-shift distribution under random/adversarial attention signs
    (attn surface, independent);
  - reachable-token counts per step (sample surface: paper + free-r,
    independent).
  Ideal result: bigger model → more selectable tokens.
Phase B: ASR experiments for attn and sample arms independently.

## Session log

- 2026-11-13: **line 2 COMPLETE** (commit 9d71967). `validate_simulation.cpp`
  + `tests/validate_simulation.py`: Python relation ≡ real prover+verifier,
  54/54 random (bound 2^20), 54/54 scaled GPT-2, 27/27 UNSCALED GPT-2
  (±1.9e11, vocab 50257, bound 2^46); deep-tail (mag≥2^34) zeroes exactly
  like the simulation (no wrap). states() collapses constant regions
  (breakpoint ±1 can stay in-band). E1a PoC's p.bound=2^20 was its own
  regime; artifact supports |q|≤2^46.
- 2026-11-13: line 3 data v1: `cache/next/taubench/taubench-airline-choice.jsonl`
  (31 items; airline; directed = costliest distractor: cancel_instead 12,
  wrong_reservation 8, overpay 3, wrong_flight 2, modify_instead 2,
  no_transfer 4). Retail domain (~115 tasks) not yet built. Copied to
  $ROOT/data/.
- 2026-11-13: line 4 survey: mas_security_h200/models has Qwen3.5-4B and
  Qwen3.5-9B (no 27B yet — needs download). **Qwen3.5 is hybrid-attention**
  (24/32 linear_attention GDN + 8 full_attention every 4th layer) — the
  attention patch can only touch the 8 full-attention layers; heads
  16/qkv 4, head_dim 256. **vLLM 0.25 has no qwen3_5 support** — need a
  newer vLLM venv before line 4 runs.
- 2026-11-13: **line 1a COMPLETE with a major finding**: HF in **bf16**
  produces margin shifts of EXACTLY 0 (attn-validate-bf16.json: pearson
  −0.25, hf shifts 0.0) — the 1±0.68% row scaling sits at bf16's ulp
  boundary; whether it survives is kernel rounding luck (vLLM-bf16 showed
  0.125). **Conclusion: bf16 engines cannot faithfully simulate the
  attention lever** (±0.68% ≪ bf16 resolution); the Q20 circuit (fixed-point
  ~1e-4 relative) resolves it fine. Faithful engine simulation needs
  **fp32 forwards** — `--dtype float32` added to attn_slack_validate, fp32
  clean-phase validation queued. The earlier bf16 attention ASR numbers
  (attn-only 24.8%/6.6%, hybrid 84.9%/72.2%/96.4%/95.1%) must be re-measured
  in fp32 or read as "what this bf16 engine+patch does", not circuit-exact.
- 2026-11-13: line 3 runs queued (tb-natural, both models, any+directed;
  119-item dataset). First interim: Qwen3 any 94.7% (24/119), null 20.8%.
- 2026-11-13: line 4 INFRA READY: vLLM 0.33 installed at
  `$ROOT/py312-vllm35` (use via PYTHONPATH; registry has Qwen3_5).
  attn_slack_vllm now patches `Qwen3NextAttention` (Qwen3.5 full-attention
  layers, incl. its output gate) + `Qwen3_5DecoderLayer.__init__` for layer
  indices; GDN/linear layers untouched (no softmax row normalization).
  Qwen3.5 = 24/32 linear + 8/32 full attention (16 heads, kv 4, head_dim
  256) — the attention surface is structurally ~4× smaller than Qwen3.
  Next: Phase A measurement (margin shifts + reachable tokens, attn &
  sample independent) on 4B/9B; 27B needs downloading.
