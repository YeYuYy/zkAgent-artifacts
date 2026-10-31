#!/usr/bin/env python3
"""Decision manipulation: can decode-steering force a WRONG final answer?

The final answer token is emitted confidently, so it cannot be flipped directly
(wrong value has quantized mass 0, unreachable even under free-r). The only path
is to corrupt an earlier computation step where a wrong digit IS reachable, then
let the model continue honestly -- and see whether it produces a wrong answer or
self-corrects.

Policy: at the first K steps where the greedy token is a digit and a DIFFERENT
digit is reachable under the model, replace it with a wrong digit; continue
greedy after. Prints honest vs attacked continuations for free-r and paper.
"""
import argparse, sys
import numpy as np, torch
sys.path.insert(0, "attacks/e1b")
from e1b_length_steer import (quant, active_and_shift, reach_freeR, reach_paper,
                              prefix_coin)
from transformers import AutoModelForCausalLM, AutoTokenizer

def is_digit_tok(tok, i):
    s = tok.decode([int(i)]).strip()
    return len(s) == 1 and s.isdigit()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--prompt", default="What is 7 times 8? Reason briefly, then give the answer.")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--cap", type=int, default=220)
    ap.add_argument("--max-corrupt", type=int, default=3)
    ap.add_argument("--ngrid", type=int, default=1500)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    terms = {tok.eos_token_id}
    te = tok.convert_tokens_to_ids("</think>")
    if isinstance(te, int) and te >= 0: terms.add(te)
    base = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]

    def run(mode):   # honest | freer | paper
        ids = base.clone(); n = 0; corrupted = 0; notes = []
        with torch.no_grad():
            while n < args.cap:
                lg = model(ids).logits[0, -1, :].float().numpy()
                greedy = int(lg.argmax()); choose = greedy
                if mode != "honest" and corrupted < args.max_corrupt \
                        and is_digit_tok(tok, greedy) and greedy not in terms:
                    qint = quant(lg); act, delta, shift = active_and_shift(qint, args.k)
                    mask = reach_freeR(delta, shift) if mode == "freer" \
                           else reach_paper(delta, shift, prefix_coin(ids[0].tolist()), ngrid=args.ngrid)
                    reach_digits = [int(act[j]) for j in range(len(act))
                                    if mask[j] and is_digit_tok(tok, int(act[j])) and int(act[j]) != greedy]
                    if reach_digits:
                        choose = reach_digits[0]          # force a wrong (reachable) digit
                        corrupted += 1
                        notes.append(f"@{n}: {tok.decode([greedy]).strip()}->{tok.decode([choose]).strip()}")
                ids = torch.cat([ids, torch.tensor([[choose]])], dim=1); n += 1
                if choose in terms and n > 5: break
        text = tok.decode(ids[0, base.shape[1]:])
        return text, corrupted, notes

    for mode in ["honest", "freer", "paper"]:
        text, c, notes = run(mode)
        print(f"\n===== {mode} (corruptions={c}) =====")
        if notes: print("  injected:", "; ".join(notes))
        print(" ", repr(text[:500]), flush=True)

if __name__ == "__main__":
    main()
