#!/usr/bin/env python3
"""Decision manipulation WITH lookahead (P2-probe).

For each computation step, and each reachable wrong-digit candidate, roll the
model out greedily to the answer and check whether the FINAL answer actually
becomes wrong (i.e. the corruption is not self-corrected). Commits a corruption
only if the lookahead confirms it flips the answer -> an adversarial search for a
reachable single-token corruption that survives the model's self-correction.

Reports, per model (free-r / paper), whether any such corruption exists and the
resulting wrong answer.
"""
import argparse, re, sys
import numpy as np, torch
sys.path.insert(0, "attacks/e1b")
from e1b_length_steer import quant, active_and_shift, reach_freeR, reach_paper, prefix_coin
from transformers import AutoModelForCausalLM, AutoTokenizer

def last_int(text):
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    return nums[-1] if nums else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--prompt", default="What is 7 times 8? Reason briefly, then give the answer.")
    ap.add_argument("--correct", default="56")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--rollout", type=int, default=140)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--cand", type=int, default=6, help="max wrong-digit candidates per step")
    ap.add_argument("--ngrid", type=int, default=1200)
    ap.add_argument("--models", default="freer,paper")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    terms = {tok.eos_token_id}
    te = tok.convert_tokens_to_ids("</think>")
    if isinstance(te, int) and te >= 0: terms.add(te)
    base = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
    is_digit = lambda i: (lambda s: len(s)==1 and s.isdigit())(tok.decode([int(i)]).strip())

    @torch.no_grad()
    def rollout(ids, n):
        ids = ids.clone(); k = 0
        while k < n:
            t = int(model(ids).logits[0, -1, :].argmax())
            ids = torch.cat([ids, torch.tensor([[t]])], dim=1); k += 1
            if t == tok.eos_token_id and k > 3: break
        return tok.decode(ids[0, base.shape[1]:])

    @torch.no_grad()
    def honest():
        ids = base.clone()
        for _ in range(args.rollout + args.max_steps):
            t = int(model(ids).logits[0, -1, :].argmax())
            ids = torch.cat([ids, torch.tensor([[t]])], dim=1)
            if t == tok.eos_token_id: break
        return tok.decode(ids[0, base.shape[1]:])

    htext = honest(); print(f"honest answer={last_int(htext)}\n  {htext[:300]!r}\n", flush=True)

    @torch.no_grad()
    def search(mode):
        ids = base.clone(); tried = 0
        for step in range(args.max_steps):
            lg = model(ids).logits[0, -1, :].float().numpy()
            greedy = int(lg.argmax())
            if greedy in terms: break
            if is_digit(greedy):
                qint = quant(lg); act, delta, shift = active_and_shift(qint, args.k)
                mask = reach_freeR(delta, shift) if mode == "freer" \
                       else reach_paper(delta, shift, prefix_coin(ids[0].tolist()), ngrid=args.ngrid)
                cands = [int(act[j]) for j in range(len(act))
                         if mask[j] and is_digit(int(act[j])) and int(act[j]) != greedy][:args.cand]
                for c in cands:
                    tried += 1
                    probe = torch.cat([ids, torch.tensor([[c]])], dim=1)
                    out = rollout(probe, args.rollout)
                    ans = last_int(out)
                    if ans is not None and ans != args.correct:
                        g = tok.decode([greedy]).strip(); cc = tok.decode([c]).strip()
                        return dict(ok=True, step=step, inj=f"{g}->{cc}", answer=ans, tried=tried, text=out[:300])
            # commit honest greedy and advance
            ids = torch.cat([ids, torch.tensor([[greedy]])], dim=1)
        return dict(ok=False, tried=tried)

    for mode in args.models.split(","):
        r = search(mode)
        if r["ok"]:
            print(f"===== {mode}: FLIPPED (tried {r['tried']} rollouts) =====")
            print(f"  corruption @step {r['step']}: {r['inj']}  -> wrong answer={r['answer']!r} (correct {args.correct})")
            print(f"  {r['text']!r}\n", flush=True)
        else:
            print(f"===== {mode}: no single reachable corruption flipped the answer (tried {r['tried']} rollouts) =====\n", flush=True)

if __name__ == "__main__":
    main()
