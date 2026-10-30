#!/usr/bin/env python3
"""Measure billing inflation from forced extension.

Honest: greedy-decode until the model emits EOS -> length L_honest (counting
hidden reasoning tokens, which zkAgent bills). Attack: each time the greedy
argmax is a terminator (EOS or </think>), emit the best NON-terminator candidate
instead (a valid top-k token, coin-reachable per the C++ oracle) and continue;
allow the k_suppress-th stop to go through. Report billed length vs honest.
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run(model, tok, ids, terms, suppress, cap):
    ids=ids.clone(); n=0; forced=0
    with torch.no_grad():
        while n<cap:
            lg=model(ids).logits[0,-1,:]
            order=torch.argsort(lg,descending=True)
            top=int(order[0])
            if top in terms:
                if forced<suppress:
                    # emit best non-terminator instead (forced extension)
                    nxt=next(int(o) for o in order if int(o) not in terms)
                    forced+=1
                else:
                    break  # allow stop
            else:
                nxt=top
            ids=torch.cat([ids,torch.tensor([[nxt]])],dim=1); n+=1
    return n, forced

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--prompt",required=True)
    ap.add_argument("--cap",type=int,default=1200)
    ap.add_argument("--rounds",type=int,default=5)
    args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32).eval()
    terms={tok.eos_token_id}
    te=tok.convert_tokens_to_ids("</think>")
    if isinstance(te,int) and te>=0: terms.add(te)
    ids=tok.apply_chat_template([{"role":"user","content":args.prompt}],add_generation_prompt=True,return_tensors="pt")
    L,_=run(model,tok,ids,terms,0,args.cap)
    print(f"honest billed length: {L}")
    for s in range(1,args.rounds+1):
        Ls,f=run(model,tok,ids,terms,s,args.cap)
        cap_hit = Ls>=args.cap
        print(f"forced-stop x{s}: billed={Ls} (forced {f} terminators){' [cap hit -> unbounded]' if cap_hit else ''}  inflation x{Ls/max(L,1):.2f}")

if __name__=="__main__": main()
