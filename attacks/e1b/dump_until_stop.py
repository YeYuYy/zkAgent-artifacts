#!/usr/bin/env python3
"""Generate with the chat template until the model naturally stops, dumping
per-step integer logits (zkAgent domain) plus marking stop candidates.

Output: line1 "T VOCAB eos_id think_end_id", then T lines of VOCAB ints.
The honest greedy path is what the model chose; we also record, per step,
whether a terminator (EOS or </think>) was among the top candidates.
"""
import argparse, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
LOGIT_SCALE = 7.952219286828541e-10

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    msgs=[{"role":"user","content":args.prompt}]
    ids=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    eos=tok.eos_token_id
    think_end=tok.convert_tokens_to_ids("</think>")
    if not isinstance(think_end,int) or think_end<0: think_end=-1
    vocab=model.config.vocab_size
    rows=[]
    with torch.no_grad():
        for step in range(args.max_new):
            out=model(ids); logit=out.logits[0,-1,:].float()/args.temperature
            rows.append(logit)
            nxt=int(torch.argmax(logit))
            ids=torch.cat([ids, torch.tensor([[nxt]])],dim=1)
            if nxt==eos: break
    with open(args.out,"w") as f:
        f.write(f"{len(rows)} {vocab} {eos} {think_end}\n")
        for r in rows:
            q=torch.round(r.double()/LOGIT_SCALE).to(torch.int64)
            f.write(" ".join(map(str,q.tolist()))+"\n")
    text=tok.decode(ids[0,-len(rows):])
    print(f"wrote {len(rows)} steps; stopped_naturally={rows and int(torch.argmax(rows[-1]))==eos}", file=sys.stderr)
    print("tail:", repr(text[-200:]), file=sys.stderr)

if __name__=="__main__": main()
