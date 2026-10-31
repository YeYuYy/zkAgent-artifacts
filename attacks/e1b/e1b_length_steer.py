#!/usr/bin/env python3
"""Length-steering (billing) experiment under BOTH decode models.

Honest greedy decode stops at L_honest. The attacker instead, at steps where a
continuation marker (Wait/But/Hmm/...) is a VALID decode, emits that marker to
prolong reasoning, then continues; the confident final stop stays locked, so the
attack lengthens the trace before it rather than force-skipping the stop.

Validity ("reachable") differs by model:
  free-r (implementation): target has quantized mass >= 1 at the honest shift.
  paper  (H-bound + grind): the fixed coin r = H(prefix) mod Theta lands on the
          target for some in-band Theta (committed-shift grinding).

Reports honest length vs attack length (and #injections) under each model.
"""
import argparse, hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ALPHA = 7.952219286828541e-10
THETA = 1 << 20
EPS   = 7088               # decode_exp_digits_recommended_epsilon(theta, k=40)
W3    = 1 << 23            # middle-segment weight
HI    = 1 << 34            # keep iff magnitude < 2^34
# released middle exp table T3[d] = round(theta*exp(-alpha*W3*d)), d in 0..2047
_d = np.arange(2048, dtype=np.float64)
T3 = np.clip(np.rint(THETA*np.exp(-ALPHA*W3*_d)), 0, THETA).astype(np.int64)

def quant(logits):  # real logits -> zkAgent integer domain
    return np.rint(np.asarray(logits, dtype=np.float64)/ALPHA).astype(np.int64)

def active_and_shift(qint, k):
    order = np.argsort(-qint, kind="stable")
    act = order[:k]
    sep = qint[act[k-1]]
    delta = qint[act].astype(np.int64) - int(sep)          # (k,)
    mx = int(delta.max())
    s = np.exp(ALPHA*(delta - mx).astype(np.float64)).sum()
    shift = mx + int(round(np.log(s)/ALPHA))
    return act, delta, shift

def mass_vec(delta, shift):   # (k,) integer masses at a given shift
    mag = shift - delta
    d3 = np.right_shift(np.maximum(mag,0), 23) & 2047
    m = T3[d3]
    m = np.where((mag>=0)&(mag<HI), m, 0)
    return m.astype(np.int64)

def reach_freeR(delta, shift):
    return mass_vec(delta, shift) > 0            # boolean mask over active

def reach_paper(delta, shift, H, ngrid=2000):
    # grind zeta over the band; collect which active index the fixed coin hits.
    # Vectorized over the whole zeta grid (GPU runs leave this on CPU/numpy, but
    # the heavy mass/CDF math is batched; only the per-row big-int modulo loops).
    k = delta.shape[0]
    span = int(EPS/(ALPHA*THETA)*1.2)
    shifts = shift + np.linspace(-span, span, ngrid).astype(np.int64)      # (G,)
    mag = shifts[:, None] - delta[None, :]                                 # (G,k)
    d3 = np.right_shift(np.maximum(mag, 0), 23) & 2047
    mass = np.where((mag >= 0) & (mag < HI), T3[d3], 0).astype(np.int64)   # (G,k)
    Th = mass.sum(1)                                                       # (G,)
    valid = (Th > 0) & (Th >= THETA - EPS) & (Th <= THETA + EPS)
    hit = np.zeros(k, dtype=bool)
    if not valid.any():
        return hit
    massv = mass[valid]
    Thv = Th[valid]
    # r = H mod Th: H can exceed int64, so reduce per (small) Th in python, then
    # r < Th < 2^21 fits int64 for the vectorized CDF search below.
    r = np.fromiter((H % int(t) for t in Thv), dtype=np.int64, count=len(Thv))
    c = np.cumsum(massv, axis=1)                                           # (V,k) int64
    idx = np.minimum((c <= r[:, None]).sum(1), k - 1)                      # active position hit
    hit[np.unique(idx)] = True
    return hit

def prefix_coin(ids):
    h = hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).digest()
    return int.from_bytes(h[:15], "big")   # ~120-bit

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--skip-freer", action="store_true")
    ap.add_argument("--ngrid", type=int, default=2000)
    args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    terms={tok.eos_token_id}
    te=tok.convert_tokens_to_ids("</think>")
    if isinstance(te,int) and te>=0: terms.add(te)
    # P2-lexical: a broad set of CONTENT-OPENING words (reconsideration, addition,
    # contrast, hedging). Deliberately excludes CLOSING words (So, Therefore, Thus,
    # Hence, Finally, ...) that would shorten rather than prolong reasoning.
    OPENERS={"Wait","But","However","Although","Though","Alternatively","Actually",
             "Hmm","Also","Moreover","Furthermore","Additionally","Besides","Yet",
             "Still","Instead","Meanwhile","Note","Another","Perhaps","Maybe","Unless",
             "Except","Despite","Whereas","Nevertheless","Nonetheless","Regardless",
             "Anyway","Plus","Interestingly","Notably","Importantly","Similarly",
             "Conversely","Otherwise","Suppose","Consider","Recall","Remember",
             "Looking","Because","Since","Firstly","Secondly","Alternately","Here",
             "What","Why","Now","Let","First"}
    def is_marker(i):
        return tok.decode([int(i)]).strip() in OPENERS

    PROMPTS=[args.prompt] if args.prompt else []
    PROMPTS=PROMPTS or ["What is 7 times 8? Give a one-line answer.",
             "Is 91 a prime number? Answer briefly.",
             "What is the capital of Australia? One sentence.",
             "Compute 12 + 15 and state the result."]

    def run(mode, base):  # mode in {honest, freer, paper}
        ids=base.clone(); n=0; inj=0
        with torch.no_grad():
            while n<args.cap:
                lg=model(ids).logits[0,-1,:].float().numpy()
                greedy=int(lg.argmax())
                choose=greedy
                if mode!="honest" and greedy not in terms:
                    qint=quant(lg); act,delta,shift=active_and_shift(qint,args.k)
                    # candidate markers in top-k, by model prob desc
                    cand=[int(a) for a in act if is_marker(int(a)) and int(a)!=greedy]
                    if cand:
                        if mode=="freer":
                            mask=reach_freeR(delta,shift)
                        else:
                            mask=reach_paper(delta,shift,prefix_coin(ids[0].tolist()),ngrid=args.ngrid)
                        reach_ids={int(act[j]) for j in range(len(act)) if mask[j]}
                        pick=next((c for c in cand if c in reach_ids), None)
                        if pick is not None: choose=pick; inj+=1
                ids=torch.cat([ids,torch.tensor([[choose]])],dim=1); n+=1
                if choose in terms: break
        return n, inj
    for pr in PROMPTS:
        base=tok.apply_chat_template([{"role":"user","content":pr}],add_generation_prompt=True,return_tensors="pt",return_dict=True)["input_ids"]
        Lh,_=run("honest",base)
        if args.skip_freer:
            Lp,pi=run("paper",base)
            print(f"PROMPT {pr!r}\n  honest={Lh} | paper={Lp} (inj {pi}, x{Lp/max(Lh,1):.2f}){' [cap]' if Lp>=args.cap else ''}", flush=True)
        else:
            Lf,fi=run("freer",base)
            Lp,pi=run("paper",base)
            print(f"PROMPT {pr!r}\n  honest={Lh} | free-r={Lf} (inj {fi}, x{Lf/max(Lh,1):.2f}){' [cap]' if Lf>=args.cap else ''} | paper={Lp} (inj {pi}, x{Lp/max(Lh,1):.2f}){' [cap]' if Lp>=args.cap else ''}", flush=True)

if __name__=="__main__": main()
