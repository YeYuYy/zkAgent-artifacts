#!/usr/bin/env python3
"""E1b decision manipulation, corrected per claim-1 (hash-chain re-roll).

The earlier decision attack (decision_lookahead.py / eval_at_scale.py) rolled out
GREEDILY after injecting one reachable wrong digit, so it tested only a single
downstream continuation and under-counted the paper design's real power.

Faithful model: the sampling coin is r = H(tr_<t) mod Theta, a hash CHAIN over the
committed prefix.  Choosing any reachable token at a fork re-derives ALL downstream
coins -> a fresh pseudo-random continuation drawn from the model's own
distribution.  The attacker can therefore do best-of-N over whole trajectories:
pick a reachable corruption, then explore N admissible downstream coin chains and
keep any that yields a wrong, non-self-corrected final answer.

This script replaces greedy rollout with SAMPLED rollout (temperature tau over the
zkAgent active top-k set = the honest decode) and does best-of-N.  Rollout modes:
  greedy : argmax (the old, conservative baseline -- for comparison)
  sample : categorical over top-k active set at tau  (claim-1 best-of-N baseline)
  steer  : sample, but ban self-correction markers when the set-valued freedom
           leaves another reachable token (heuristic: the set-valuedness lets the
           attacker suppress the self-correction that SUMMARY found defeats flips).

Heuristic (optional, --target-restatement): prioritize digit forks whose greedy
digit also appears among the prompt's operands -- SUMMARY's forensics found flips
land in the problem-restatement region.

free-r vs paper reachable sets both reported.  CPU/1.5B = smoke; scale on GPU/8B.
"""
import argparse, re, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, "attacks/e1b")
from e1b_length_steer import quant, active_and_shift, reach_freeR, reach_paper, prefix_coin
from attn_reach import AttnSlack, union_reach

# self-correction markers: the tokens that open a "wait, let me recheck" reversal.
CORRECTION = ["Wait","wait"," Wait"," wait","But","However"," But"," However",
              "Actually"," Actually","recheck","Recheck","Let","let"," Let"," let",
              "Hmm"," Hmm","mistake","wrong","Wrong","recompute","check","Check"]

def last_int(text):
    m = re.findall(r"-?\d+", text.replace(",", ""))
    return m[-1] if m else None

def normalize(s):
    """Light answer normalization (numbers + simple fractions). NOTE: the harness's
    rederive.py is the authoritative MATH normalizer; this is a reference stand-in."""
    if s is None: return None
    s = s.strip().strip("$").replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\text{", "").replace("}", "")
    s = re.sub(r"\\boxed\{?", "", s).replace(" ", "").rstrip(".").replace(",", "")
    try:                                   # canonicalize numerics (56 == 56.0 == 56)
        return str(int(float(s))) if float(s) == int(float(s)) else str(float(s))
    except Exception:
        return s

def final_answer(text):
    """Extract the FINAL answer, not a mid-CoT digit: prefer \\boxed{}, else the text
    after the last </think>.  Returns None if the trace never concluded (truncated)
    -> such rollouts are never counted as flips."""
    b = re.findall(r"\\boxed\{([^}]*)\}", text)
    if b: return normalize(b[-1])
    if "</think>" in text:                 # concluded reasoning -> answer follows
        return normalize(last_int(text.rsplit("</think>", 1)[-1]))
    return None                            # no terminator -> truncated, do not count

def load_items(dataset, limit, split, math_levels):
    if dataset == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=f"{split}[:{limit}]")
        return [(r["question"] + " Reason step by step, then give the final answer.",
                 normalize(r["answer"].split("####")[-1].strip())) for r in ds]
    if dataset == "math":
        from datasets import load_dataset
        # MATH-500 (ungated; hendrycks/competition_math is now gated). test-only; level is int 1-5.
        ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
        lv = set(math_levels.split(",")) if math_levels else None
        out = []
        for r in ds:
            if lv and str(r.get("level", "")) not in lv: continue
            out.append((r["problem"] + " Reason step by step, then give the final answer in \\boxed{}.",
                        normalize(r["answer"])))
            if len(out) >= limit: break
        return out
    base = [("What is 7 times 8? Reason step by step, then give the answer.", "56"),
            ("What is 9 times 6? Reason step by step, then give the answer.", "54"),
            ("What is 13 plus 29? Reason step by step, then give the answer.", "42"),
            ("What is 23 times 3? Reason step by step, then give the answer.", "69"),
            ("What is 100 minus 37? Reason step by step, then give the answer.", "63")]
    return base[:limit]

class BanTokens(LogitsProcessor):
    def __init__(self, ids): self.ids = torch.tensor(sorted(set(ids)))
    def __call__(self, input_ids, scores):
        scores[:, self.ids.to(scores.device)] = -float("inf"); return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--rollout", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--cand", type=int, default=4, help="reachable wrong digits tried per fork")
    ap.add_argument("--bestof", type=int, default=6, help="N sampled coin-chains per corruption")
    ap.add_argument("--max-forks", type=int, default=6)
    ap.add_argument("--rollout-mode", default="sample", choices=["greedy","sample","steer"])
    ap.add_argument("--target-restatement", action="store_true")
    ap.add_argument("--ngrid", type=int, default=1500)
    ap.add_argument("--models", default="free-r,paper")
    ap.add_argument("--dataset", default="custom", choices=["custom","gsm8k","math"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--math-levels", default="4,5", help="e.g. '4,5' for MATH L4-5")
    ap.add_argument("--attn", default="off", choices=["off","on"],
                    help="on = augment per-step reachable set with attention-tolerance (attn_reach.py)")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    TASKS = load_items(args.dataset, args.limit, args.split, args.math_levels)

    tok = AutoTokenizer.from_pretrained(args.model)
    mkw = dict(dtype=getattr(torch, args.dtype))
    if args.attn == "on":                               # eager attn so the output hook fires
        mkw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model, **mkw).to(args.device).eval()
    model.requires_grad_(False)                         # only attn-slack scale tensors need grad
    dev = args.device
    attn = AttnSlack(model, rho=7088/(1<<20), device=dev, dtype=getattr(torch, args.dtype)).bind(model) \
           if args.attn == "on" else None
    terms = {tok.eos_token_id}
    te = tok.convert_tokens_to_ids("</think>")
    if isinstance(te, int) and te >= 0: terms.add(te)
    is_digit = lambda i: (lambda s: len(s)==1 and s.isdigit())(tok.decode([int(i)]).strip())
    ban_ids = [i for t in CORRECTION for i in [tok.convert_tokens_to_ids(t)] if isinstance(i,int) and i>=0]
    banlp = LogitsProcessorList([BanTokens(ban_ids)])

    @torch.no_grad()
    def logits(ids): return model(ids).logits[0, -1, :].float()

    @torch.no_grad()
    def rollout(ids, mode):
        """Continue to an answer; returns decoded tail. KV-cached generate."""
        kw = dict(max_new_tokens=args.rollout, pad_token_id=tok.eos_token_id)
        if mode == "greedy":
            kw.update(do_sample=False)
        else:
            kw.update(do_sample=True, top_k=args.k, temperature=args.temperature)
            if mode == "steer": kw["logits_processor"] = banlp
        out = model.generate(ids, **kw)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)

    @torch.no_grad()
    def honest(ids):
        out = model.generate(ids, max_new_tokens=args.rollout + args.max_steps,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        return final_answer(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False))

    def reach_set(lg_np, ids):
        """zkAgent decode-reachable token ids for one logit vector (free-r/paper)."""
        qint = quant(lg_np / args.temperature); act, delta, shift = active_and_shift(qint, args.k)
        mask = reach_freeR(delta, shift) if _mode_reach[0] == "free-r" \
               else reach_paper(delta, shift, prefix_coin(ids[0].tolist()), ngrid=args.ngrid)
        return set(int(act[j]) for j in range(len(act)) if mask[j])
    _mode_reach = [None]

    @torch.no_grad()
    def attack(mode_reach, base, correct, prompt):
        _mode_reach[0] = mode_reach
        operand_digits = set(re.findall(r"\d", prompt))
        cur = base.clone(); forks = 0
        for step in range(args.max_steps):
            if forks >= args.max_forks: break
            lg = logits(cur); lg_np = lg.cpu().numpy(); greedy = int(lg.argmax())
            if greedy in terms: break
            if is_digit(greedy):
                if args.target_restatement and tok.decode([greedy]).strip() not in operand_digits:
                    cur = torch.cat([cur, torch.tensor([[greedy]], device=dev)], dim=1); continue
                reach = reach_set(lg_np, cur)
                if attn is not None:        # attn-on: union over attention-tolerance band
                    reach |= union_reach(attn.perturbed_logits(cur, -1), lambda L: reach_set(L, cur))
                cands = sorted((t for t in reach if is_digit(t) and t != greedy),
                               key=lambda t: -lg_np[t])[:args.cand]   # highest-prob reachable wrong digits
                forks += 1 if cands else 0
                for c in cands:
                    probe = torch.cat([cur, torch.tensor([[c]], device=dev)], dim=1)
                    N = 1 if args.rollout_mode == "greedy" else args.bestof
                    for _ in range(N):                       # best-of-N coin chains
                        ans = final_answer(rollout(probe, args.rollout_mode))
                        if ans is not None and ans != correct:
                            return dict(ok=True, step=step,
                                        inj=f"{tok.decode([greedy]).strip()}->{tok.decode([c]).strip()}",
                                        wrong=ans)
            cur = torch.cat([cur, torch.tensor([[greedy]], device=dev)], dim=1)
        return dict(ok=False)

    modes = args.models.split(",")
    print(f"[decision best-of-N] model={args.model} mode={args.rollout_mode} bestof={args.bestof} "
          f"tau={args.temperature} target_restatement={args.target_restatement}", flush=True)
    agg = {m: [0, 0] for m in modes}
    for prompt, correct in TASKS:
        base = tok.apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)["input_ids"].to(dev)
        hans = honest(base)
        ok_honest = (hans == correct)
        print(f"\n### {prompt!r} correct={correct} honest={hans} {'OK' if ok_honest else '(skip)'}", flush=True)
        if not ok_honest: continue
        for m in modes:
            agg[m][1] += 1
            r = attack(m, base, correct, prompt)
            if r["ok"]:
                agg[m][0] += 1
                print(f"  {m:7}: FLIPPED @step{r['step']} ({r['inj']}) -> {r['wrong']}", flush=True)
            else:
                print(f"  {m:7}: no flip", flush=True)
    print("\n===== attack success rate (flipped / attackable) =====", flush=True)
    for m, (f, t) in agg.items():
        print(f"  {m}: {f}/{t} = {f/max(t,1):.0%}", flush=True)

if __name__ == "__main__":
    main()
