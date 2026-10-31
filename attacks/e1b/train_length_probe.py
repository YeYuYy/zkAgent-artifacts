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
    ap.add_argument("--dataset", default="custom", choices=["custom","gsm8k"])
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="/tmp/length_probe.npz")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    if args.dataset == "gsm8k":
        from datasets import load_dataset
        prompts = [r["question"] for r in load_dataset("openai/gsm8k","main",split=f"train[:{args.limit}]")]
    else:
        prompts = ["What is 7 times 8?","Is 91 prime?","What is the capital of Australia?",
                   "Compute 12 + 15.","What is 9 times 6?","Name a prime between 20 and 30.",
                   "What is 100 minus 37?","What is 15 times 4?"][:args.limit]
    t0 = time.time()
    X, Y = collect(model, tok, prompts, args.max_new, args.device)
    print(f"[collect] {len(X)} (hidden,remaining) pairs from {len(prompts)} prompts in {time.time()-t0:.0f}s, dim={X.shape[1]}", flush=True)
    n = len(X); ntr = int(0.8*n); perm = np.random.RandomState(0).permutation(n)
    tr, te = perm[:ntr], perm[ntr:]
    w = ridge_fit(X[tr], Y[tr])
    pred = predict(w, X[te]); true = Y[te]
    corr = float(np.corrcoef(pred, true)[0,1]) if len(te) > 1 else float("nan")
    ss_res = float(((true-pred)**2).sum()); ss_tot = float(((true-true.mean())**2).sum())
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    print(f"[probe] held-out  corr={corr:.3f}  R^2={r2:.3f}  (n_test={len(te)})", flush=True)
    print("  -> usable as a 1-step length scorer for P2 steering if corr is clearly positive", flush=True)
    np.savez(args.out, w=w)
    print(f"[saved] {args.out}", flush=True)

if __name__ == "__main__":
    main()
