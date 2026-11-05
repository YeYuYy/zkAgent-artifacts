# E1b-NEXT — corrected attack mechanism + at-scale experiment plan

This note records (A) the **corrected attack mechanism** for the zkAgent decision/billing
attacks, (B) the **experiment plan** it implies, and (C) the **code changes** in this branch.
It supplements `SUMMARY.md` (which holds the measured 1.5B/8B results); read that first.

> Scope: the mechanism and reference code here are validated on this machine (GPT-2 124M /
> R1-Distill-Qwen-1.5B, single GPU, smoke scale). The paper-grade runs (7B/8B × n=150) must be
> executed on the H200 vLLM harness (`/root/zk_agent`, not in this repo); this branch is the
> reference/spec to port into `experiments/{evaluate,engine,relation,vllm_runner}.py`.

---

## A. Corrected attack mechanism

zkAgent's two tolerance-checked normalizers make the proof relation **set-valued** (E1a). The
earlier E1b decision attack under-counted the paper design's power because it rolled out
**greedily** after a single-digit injection. The corrected mechanism has three levers:

### A1. Hash-chain re-roll → trajectory best-of-N (the main correction)
The sampling coin is `r = H(tr_<t) mod Θ`, a hash **chain** over the committed prefix. Choosing
any reachable token at a fork re-derives **all** downstream coins ⇒ a fresh pseudo-random
continuation drawn from the **model's own distribution**. So single-step set-valuedness amplifies
into **best-of-N over whole trajectories**: pick a reachable corruption, explore N admissible
downstream coin chains, keep any that yields a wrong, non-self-corrected answer.

- Implementation: rollout is now **sampled** over the active top-k set at temperature τ (= the
  faithful honest decode), with a best-of-N loop. `greedy` (N=1) is kept as the conservative
  baseline. ASR is **monotonic non-decreasing in N**; it approaches a ceiling set by "does a
  surviving-wrong trajectory exist" (coin-locked answer regions / universal self-correction cap it).

### A2. Self-correction suppression (`steer`, a within-reachable heuristic)
SUMMARY's forensics: flips fail because the model self-corrects ("Wait/let me recheck…"). Since
the relation is set-valued, the attacker may **ban self-correction markers while another reachable
token exists** — i.e. steer the rollout away from self-correction *without leaving the
proof-accepted set*. Faithful guard: only drop a marker when the reachable set still has an
alternative. Optional `--target-restatement`: prioritize digit forks whose greedy digit also
appears among the prompt operands (flips land in the problem-restatement region).

### A3. Attention-tolerance set amplification (`attn on`, independent fork source)
The attention softmax verifier commits a per-row shift ζ and only checks row mass
`Θ_row ∈ [θ±ε]`; the row output is `(Σ m_i v_i)/θ` with the **fixed public θ** as divisor, so a
uniform ζ shift does **not** cancel — each row output scales by `g ∈ [1−ε/θ, 1+ε/θ] = 1±0.676%`,
chosen independently per (layer, head, row). This perturbs the final logits and **moves CDF
boundaries**, so a fixed coin lands on a neighboring token ⇒ **extra decode forks**, independent
of the decode-coin freedom. Measured (GPT-2, `attn_slack_logits.py`): adversarial per-row signs
flip the greedy token at steps with honest margin ≲0.30, and enlarge the **paper** reachable set
22→37 (+68%); free-r already saturates top-k=40 so it is unaffected there.

**`attn-on` in the search is a STRAIGHTFORWARD placeholder (`attn_reach.py`, v0), to be refined:**
- per-**layer scalar** g_L (not per-(head,row)); architecture-agnostic output hook.
- per step it probes {honest, coherent+, coherent−, adversarial-sign} and takes the **union** of
  the decode-reachable sets ⇒ a **loose over-approximation** of what one committed config gives.
- cost: ~4 extra forwards (one with a backward) per augmented step.
These three simplifications are deliberate; tighten to the real per-row enumeration later.

### Composition
The levers compound: A3 manufactures more low-margin forks, A1 does best-of-N over them, A2 keeps
the corrupted branch from self-correcting. "attention alone as a standalone **decision** attack"
is **not** established (bounded ±0.68% only flips low-margin steps; answer tokens are usually
high-margin/locked; multi-step coordination untested) — it is a standalone source of
set-valuedness/billing and an amplifier of the decode attack.

---

## B. Experiment plan (paper-only go/no-go)

free-r is already settled (released-impl bug; 8B ASR 11.7–25.4%, billing unbounded) — **no new
free-r runs**. The frontier question: **is the paper design breakable?** All runs below are
`decode = paper`. Notation: lowercase **n** = number of dataset problems; capital **N** =
best-of-N rollout budget. ASR denominator = *attackable* (items answered correctly honestly), not n.

Fixed axes: models **{R1-Distill-7B, Qwen3-8B}**; datasets **{MATH-500 L4–5 (primary), GSM8K (ref)}**.

- **A0 — attention magnitude @ 8B** (cheapest; run first). `attn_slack_logits.py` on both models over
  real mid-CoT traces: adversarial margin-swing distribution + paper reachable-set increment.
  Decides whether the `attn-on` axis is alive at 8B (and informs the A3 definition).
- **D1 — budget ladder**: N ∈ {1, 8, 32}, (steer off, attn off), **n=50** → ASR-vs-N curve
  (tests monotonicity / whether Qwen3-8B's 0 is a real ceiling or a budget artifact).
- **D2 — core 2×2**: N=32, {steer off/on} × {attn off/on}, **n=150** → marginal value of steering
  and attention. (off,off)@N=32 is shared with D1 ⇒ 6 runs per (model, dataset).
- **B — billing**: {attn off/on} × τ ∈ {1.0, (0.6)}, policy=lexical, **n=150**. best-of-N does not
  apply to billing; the question is whether attn-on widens the opener-reachable set under paper.

Ordering/cost: A0 → D2 (Qwen3-8B four cells are the dearest, prioritize) → D1 ladder (n=50) →
B. GSM8K only on key cells; MATH-500 full.

---

## C. Code changes in this branch

New files:
- **`attn_reach.py`** — `AttnSlack` (per-layer attention-output scaling hook, architecture-agnostic)
  + `perturbed_logits` (honest/coherent±/adversarial) + `union_reach`. The `attn-on` placeholder (A3).
- **`attn_slack_logits.py`** — standalone attention-tolerance → logit-swing / reachable-set
  measurement (GPT-2); produced the A0-style numbers in SUMMARY §E1b-attn.
- **`decision_bestofN.py`** — corrected decision attack (A1+A2+A3): `--rollout-mode
  {greedy,sample,steer}`, `--bestof N`, `--target-restatement`, `--attn {off,on}`,
  `--dataset {custom,gsm8k,math}`, `--math-levels`. Hardened answer extraction (`\boxed`/`</think>`
  only; truncated rollouts never count as flips) + light `normalize`.

Modified:
- **`eval_at_scale.py`** — added `--attn {off,on}` (unions the attention band into the per-step
  reachable set for both billing and the legacy decision), `--math-levels`, MATH loader switched to
  **MATH-500** (`hendrycks/competition_math` is now gated). Its `decision` is the **legacy greedy**
  lookahead (prints a note); use `decision_bestofN.py` for the corrected search.
- **`SUMMARY.md`** — new §E1b-5 (hash-chain best-of-N) and §E1b-attn (attention logit swing), with
  cross-refs in §3/§4/§5.

Validated on this machine (smoke): decision {greedy, sample, steer, attn-on}, billing {attn-on},
loaders {custom, gsm8k, math=MATH-500 with level filter}. Not run here: 7B/8B × n=150 (needs the
H200 vLLM harness).

### Repro (smoke)
```bash
# attention magnitude (A0-style), GPT-2 = zkAgent artifact model
python3 attacks/e1b/attn_slack_logits.py --model openai-community/gpt2
# corrected decision attack (paper), sampled best-of-N, GSM8K
python3 attacks/e1b/decision_bestofN.py --dataset gsm8k --models paper \
    --rollout-mode sample --bestof 32 --attn on --device cuda --limit 50
# MATH-500 Level 4-5
python3 attacks/e1b/decision_bestofN.py --dataset math --math-levels 4,5 --models paper \
    --rollout-mode steer --bestof 32 --attn on --device cuda --limit 150
# billing with attention band, tau sweep
python3 attacks/e1b/eval_at_scale.py --dataset math --math-levels 4,5 --task billing \
    --models paper --attn on --temperature 1.0 --device cuda --limit 150
```
At scale (8B): same flags, ported into the H200 vLLM step-loop harness; `--device cuda`,
`--dtype bfloat16`.
