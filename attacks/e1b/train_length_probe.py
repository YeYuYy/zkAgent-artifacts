#!/usr/bin/env python3
"""Train a remaining-length probe on hidden states (the true P2 value probe).

The proposal's P2 attacker scores each admissible branch with a remaining-length
probe on hidden states ("one extra step per branch"), instead of a full rollout.
This script builds that probe:

  1. collect: honest greedy rollouts; at each step record (last hidden state,
     remaining tokens until EOS).
  2. train:   ridge regression h -> remaining_length.
  3. report:  held-out correlation / R^2 (how usable the probe is for steering).
  4. save:    probe weights (.npz) for use as a 1-step-lookahead length scorer.

The provider can train this on unlimited data from its own model; here we use a
few prompts just to validate the pipeline. Scale with --limit / a dataset.
"""
import argparse, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def collect(model, tok, prompts, max_new, dev):
    X, Y = [], []
    for q in prompts:
        ids = tok.apply_chat_template([{"role":"user","content":q}], add_generation_prompt=True,
                return_tensors="pt", return_dict=True)["input_ids"].to(dev)
        hs_steps = []
        with torch.no_grad():
            for _ in range(max_new):
                out = model(ids, output_hidden_states=True)
                h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
                hs_steps.append(h)
                t = int(out.logits[0, -1, :].argmax())
                ids = torch.cat([ids, torch.tensor([[t]], device=dev)], dim=1)
                if t == tok.eos_token_id: break
        T = len(hs_steps)
        for i, h in enumerate(hs_steps):
            X.append(h); Y.append(T - i)          # remaining tokens
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)

def ridge_fit(X, Y, lam=10.0):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xb.T @ Y)
    return w

def predict(w, X):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    return Xb @ w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    # train/eval on possibly DIFFERENT domains -> measures cross-domain generalization
    ap.add_argument("--train-domains", default="math,diverse",
                    help="comma list from {gsm8k,math,custom,diverse}")
    ap.add_argument("--eval-domains", default="",
                    help="comma list for held-out eval; empty = 80/20 split of train domains")
    ap.add_argument("--limit", type=int, default=30, help="prompts per domain")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32","bfloat16","float16"])
    ap.add_argument("--out", default="/tmp/length_probe.npz")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()

    def domain_prompts(name, n):
        if name == "gsm8k":
            from datasets import load_dataset
            return [r["question"] for r in load_dataset("openai/gsm8k","main",split=f"train[:{n}]")]
        if name == "math":
            return [f"What is {a} times {b}? Reason step by step." for a in range(2,13) for b in range(2,13)][:n]
        if name == "diverse":
            pool = ["Explain why the sky is blue.","Write a haiku about autumn.",
                    "Summarize the plot of Romeo and Juliet.","What causes inflation in an economy?",
                    "Describe how a bicycle works.","Give three tips for better sleep.",
                    "What is the difference between a virus and bacteria?","Translate 'good morning' to French.",
                    "Outline the water cycle.","Why do leaves change color?","Name three uses of a magnet.",
                    "Describe the taste of an orange.","What is recursion in programming?",
                    "How does a refrigerator keep food cold?","Suggest a title for a mystery novel.",
                    "Explain photosynthesis simply.","What makes a good leader?","How is cheese made?",
                    "Describe a sunset over the ocean.","What is the capital of Australia?"]
            return (pool * ((n // len(pool)) + 1))[:n]
        # custom arithmetic
        return ["What is 7 times 8?","Is 91 prime?","Compute 12 + 15.","What is 9 times 6?",
                "What is 100 minus 37?","What is 15 times 4?"][:n]

    def gather(domains):
        Xs, Ys, tags = [], [], []
        for d in domains:
            t0 = time.time(); X, Y = collect(model, tok, domain_prompts(d, args.limit), args.max_new, args.device)
            print(f"[collect:{d}] {len(X)} pairs in {time.time()-t0:.0f}s", flush=True)
            Xs.append(X); Ys.append(Y); tags += [d]*len(X)
        return np.concatenate(Xs), np.concatenate(Ys), np.array(tags)

    train_doms = args.train_domains.split(",")
    Xtr, Ytr, _ = gather(train_doms)
    w = ridge_fit(Xtr, Ytr)

    def report(tag, X, Y):
        pred = predict(w, X);
        corr = float(np.corrcoef(pred, Y)[0,1]) if len(Y) > 1 else float("nan")
        ss_res = float(((Y-pred)**2).sum()); ss_tot = float(((Y-Y.mean())**2).sum())
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
        print(f"[eval:{tag}] corr={corr:.3f} R^2={r2:.3f} n={len(Y)}", flush=True)

    if args.eval_domains:
        print("=== cross-domain held-out ===", flush=True)
        for d in args.eval_domains.split(","):
            Xe, Ye, _ = gather([d]); report(d, Xe, Ye)
    else:
        n = len(Xtr); perm = np.random.RandomState(0).permutation(n); te = perm[int(0.8*n):]
        report("in-domain-80/20", Xtr[te], Ytr[te])
    np.savez(args.out, w=w)
    print(f"[saved] {args.out}  (use: eval_at_scale.py --probe {args.out} --bill-policy probe)", flush=True)

if __name__ == "__main__":
    main()
