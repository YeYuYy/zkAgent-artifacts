#!/usr/bin/env python3
"""Dump real per-decode-step logits, quantized into zkAgent's integer domain.

integer_logit = round(real_logit / LOGIT_SCALE), matching zkDecode's
prob ∝ exp(LOGIT_SCALE · delta) vs the model's softmax ∝ exp(logit/tau), tau=1.
Writes a text file: first line "T VOCAB", then T lines of VOCAB ints.
"""
import argparse, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGIT_SCALE = 7.952219286828541e-10  # zkAgent DecodeExpDigitsConfig.logit_scale

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--greedy", action="store_true", help="greedy continuation (else sample)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids
    vocab = model.config.vocab_size
    rows = []
    with torch.no_grad():
        for step in range(args.max_new):
            out = model(ids)
            logit = out.logits[0, -1, :].float() / args.temperature
            rows.append(logit)
            if args.greedy:
                nxt = int(torch.argmax(logit))
            else:
                p = torch.softmax(logit, dim=-1)
                nxt = int(torch.multinomial(p, 1))
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
            if nxt == tok.eos_token_id:
                break
    with open(args.out, "w") as f:
        f.write(f"{len(rows)} {vocab}\n")
        for r in rows:
            q = torch.round(r.double() / LOGIT_SCALE).to(torch.int64)
            f.write(" ".join(map(str, q.tolist())) + "\n")
    print(f"wrote {len(rows)} steps x {vocab} to {args.out}", file=sys.stderr)
    print("continuation:", repr(tok.decode(ids[0, -len(rows):])), file=sys.stderr)

if __name__ == "__main__":
    main()
