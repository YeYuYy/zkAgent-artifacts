#!/usr/bin/env python3
"""Consolidated forced-extension / billing experiment (loads model once).

For each prompt:
  1. honest greedy decode until the model emits EOS  -> L_honest (billed tokens)
  2. dump the honest trajectory's integer logits (+ eos/think ids) for the C++
     oracle to confirm the refuse-to-stop token is coin-reachable in-band
  3. forced extension: each time greedy argmax is a terminator, emit the best
     non-terminator instead, for s=1..R rounds, then allow stop -> billed length
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
LOGIT_SCALE=7.952219286828541e-10

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--cap",type=int,default=1500)
    ap.add_argument("--rounds",type=int,default=6)
    ap.add_argument("--outdir",default="/tmp/claude-0")
    args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32).eval()
    terms={tok.eos_token_id}
    te=tok.convert_tokens_to_ids("</think>")
    if isinstance(te,int) and te>=0: terms.add(te)
    prompts=["What is 7 times 8? Give a one-line answer.",
             "What is the capital of Japan? One word."]

    def gen(ids, suppress, cap, dump=None):
        ids=ids.clone(); n=0; forced=0; rows=[]
        with torch.no_grad():
            while n<cap:
                lg=model(ids).logits[0,-1,:].float()
                if dump is not None: rows.append(lg.clone())
                order=torch.argsort(lg,descending=True); top=int(order[0])
                if top in terms:
                    if forced<suppress:
                        top=next(int(o) for o in order if int(o) not in terms); forced+=1
                    else: break
                ids=torch.cat([ids,torch.tensor([[top]])],dim=1); n+=1
        if dump is not None:
            V=model.config.vocab_size
            with open(dump,"w") as f:
                f.write(f"{len(rows)} {V} {tok.eos_token_id} {te}\n")
                for r in rows:
                    q=torch.round(r.double()/LOGIT_SCALE).to(torch.int64)
                    f.write(" ".join(map(str,q.tolist()))+"\n")
        return n, forced

    for pi,p in enumerate(prompts):
        ids=tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True)["input_ids"]
        L,_=gen(ids,0,args.cap,dump=f"{args.outdir}/fe_{pi}.txt")
        print(f"\n[prompt {pi}] {p!r}\n  honest billed length (to first EOS): {L}", flush=True)
        for s in range(1,args.rounds+1):
            Ls,f=gen(ids,s,args.cap)
            hit = Ls>=args.cap
            print(f"  force-skip terminator x{s}: billed={Ls}  (skipped {f})  inflation x{Ls/max(L,1):.2f}"
                  + ("  [hit cap -> effectively unbounded]" if hit else ""), flush=True)

if __name__=="__main__": main()
