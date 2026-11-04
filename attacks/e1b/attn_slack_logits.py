#!/usr/bin/env python3
"""E1b-attn: quantify how much the attention-softmax tolerance perturbs the
per-step decode logits (and the decode reachable set).

Mechanism (verified against src + attacks/setvalued_attention.cpp):
  zkAgent commits a per-row shift zeta and only checks the row mass
  Theta_row = sum_i m_i in [theta - eps, theta + eps]; the attention output row
  is  o = (sum_i m_i v_i) / theta  with the FIXED public constant theta as the
  divisor (NOT the committed row mass).  A uniform zeta shift therefore does NOT
  cancel: it rescales the whole output row by  g = Theta_row / theta in
  [1 - eps/theta, 1 + eps/theta] = 1 +/- 0.676%.  The two poles g = 1 -/+ rho are
  the extremes the verifier accepts for each (layer, head, query-row), each chosen
  independently by the prover.

This script takes a real GPT-2 (124M, the released zkAgent artifact model),
scales every attention output row by an admissible g, re-runs the forward pass,
and measures the swing in the final-position logits and in the zkAgent decode
reachable set.  Perturbation modes:
  coherent+/- : all rows at the same pole (1 +/- rho)
  random      : each row independently +/- rho (ensemble; report max swing)
  adversarial : per-row sign = first-order-optimal to widen the top1->top2 decode
                margin at the measured step (autograd on g around 1).

Honesty: this is the first-order "two-pole" model (uniform per-row scale from the
normalizer band).  Differential integer rounding across keys adds a second-order
per-element jitter not modeled here; so the reported logit swings are the
normalizer component only -> a lower bound on the attention slack's reach.
"""
import argparse, sys
import numpy as np, torch
import transformers.models.gpt2.modeling_gpt2 as g2
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "attacks/e1b")
from e1b_length_steer import quant, active_and_shift, reach_freeR, reach_paper, prefix_coin, ALPHA, THETA, EPS

RHO = EPS / THETA                     # 7088 / 2^20 = 0.6759%

# --- attention hook: wrap eager_attention_forward, scale output row by g -------
_ORIG = g2.eager_attention_forward
_SCALE = {}          # layer_idx -> tensor (q_seq, num_heads) of per-row factors g

def _patched(module, query, key, value, attention_mask, head_mask=None, **kw):
    out, w = _ORIG(module, query, key, value, attention_mask, head_mask, **kw)
    # out: (bsz, q_seq, num_heads, head_dim); scale per (query-row, head)
    s = _SCALE.get(module.layer_idx)
    if s is not None:
        out = out * s.to(out.dtype)[None, :, :, None]
    return out, w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. In conclusion, the answer is")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--rand-draws", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layers", default="all", help="'all' or comma list of layer idxs to perturb")
    args = ap.parse_args()

    g2.eager_attention_forward = _patched
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype=torch.float32).to(args.device).eval()
    nL = model.config.n_layer; nH = model.config.n_head
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    q = ids.shape[1]
    sel = set(range(nL)) if args.layers == "all" else {int(x) for x in args.layers.split(",")}
    print(f"[attn-slack] model={args.model} layers={nL} heads={nH} prompt_len={q} "
          f"rho=eps/theta={RHO*100:.4f}%  perturbing layers={sorted(sel)}", flush=True)

    def run(scale_fn):
        _SCALE.clear()
        for L in range(nL):
            _SCALE[L] = (scale_fn(L) if L in sel
                         else torch.ones(q, nH, device=args.device))
        with torch.no_grad():
            return model(ids).logits[0, -1, :].float().cpu().numpy()

    ones = lambda L: torch.ones(q, nH, device=args.device)
    L0 = run(ones)                                   # honest (g=1) == unmodified forward
    gp = run(lambda L: torch.full((q, nH), 1 + RHO, device=args.device))   # coherent +
    gm = run(lambda L: torch.full((q, nH), 1 - RHO, device=args.device))   # coherent -

    # random per-row sign ensemble -> max observed swing
    rng = np.random.default_rng(0); rand_best = L0.copy(); rand_maxinf = 0.0
    for _ in range(args.rand_draws):
        def rf(L, rng=rng):
            s = rng.choice([1 - RHO, 1 + RHO], size=(q, nH))
            return torch.tensor(s, dtype=torch.float32, device=args.device)
        Lr = run(rf); inf = np.abs(Lr - L0).max()
        if inf > rand_maxinf: rand_maxinf, rand_best = inf, Lr

    # adversarial per-row sign: widen top1->top2 margin via autograd on g around 1
    t1, t2 = np.argsort(-L0)[:2]
    gvars = {L: torch.ones(q, nH, device=args.device, requires_grad=True) for L in sel}
    _SCALE.clear()
    for L in range(nL): _SCALE[L] = gvars[L] if L in sel else torch.ones(q, nH, device=args.device)
    lg = model(ids).logits[0, -1, :]
    obj = lg[t1] - lg[t2]                            # attacker shrinks the honest margin
    obj.backward()
    adv = run(lambda L: (1 - RHO * torch.sign(gvars[L].grad)).detach())   # minimize margin

    # ---- report logit swing ----
    def swing(Lp, name):
        d = Lp - L0
        m1 = (L0[t1] - L0[t2]); m2 = (Lp[t1] - Lp[t2])
        top = Lp.argmax()
        print(f"  {name:11} dLinf={np.abs(d).max():.4f}  dL2={np.linalg.norm(d):.4f}  "
              f"top1_margin {m1:.4f}->{m2:.4f} (d={m2-m1:+.4f})  "
              f"top1 {'KEPT' if top==t1 else 'CHANGED %d->%d'%(t1,top)}", flush=True)
    print(f"\nhonest top1={t1} ({tok.decode([int(t1)])!r}) top2={t2} ({tok.decode([int(t2)])!r}) "
          f"margin={L0[t1]-L0[t2]:.4f}\n[logit swing at the decode step, vocab={len(L0)}]", flush=True)
    swing(gp, "coherent+"); swing(gm, "coherent-")
    swing(rand_best, f"random(x{args.rand_draws})"); swing(adv, "adversarial")

    # ---- per-step sweep: how much decode margin can attention slack overcome? ----
    # For each query position p, target its own honest top1->top2 margin and find
    # the first-order-optimal per-row signs; a single forward then gives the swing.
    print(f"\n[per-step sweep: adversarial attention-slack margin swing over {q} positions]", flush=True)
    base = model(ids).logits[0].float().detach().cpu().numpy()      # (q, vocab)
    flips = 0; swings = []; overcome = []
    for p in range(q):
        tp1, tp2 = np.argsort(-base[p])[:2]
        gv = {L: torch.ones(q, nH, device=args.device, requires_grad=True) for L in sel}
        _SCALE.clear()
        for L in range(nL): _SCALE[L] = gv[L] if L in sel else torch.ones(q, nH, device=args.device)
        lgp = model(ids).logits[0, p, :]
        (lgp[tp1] - lgp[tp2]).backward()
        # forward once with the first-order-optimal per-row signs, read position p:
        _SCALE.clear()
        for L in range(nL):
            _SCALE[L] = (1 - RHO * torch.sign(gv[L].grad)).detach() if L in sel else torch.ones(q, nH, device=args.device)
        with torch.no_grad():
            lp = model(ids).logits[0, p, :].float().cpu().numpy()
        m0 = base[p][tp1] - base[p][tp2]; m1 = lp[tp1] - lp[tp2]
        swings.append(abs(m1 - m0)); fl = lp.argmax() != tp1
        flips += int(fl)
        if fl: overcome.append(m0)
    sw = np.array(swings)
    print(f"  flippable steps: {flips}/{q}   margin swing |dm|: "
          f"median={np.median(sw):.4f} max={sw.max():.4f}", flush=True)
    if overcome:
        print(f"  honest margins the attack overcame (flipped): max={max(overcome):.4f} "
              f"(i.e. any decode step with top1-top2 gap <= ~{max(overcome):.3f} is flippable by attn slack alone)", flush=True)

    # ---- decode reachable-set impact (does attention slack add decode forks?) ----
    def reach_sets(logits):
        qi = quant(logits); act, delta, shift = active_and_shift(qi, args.k)
        fr = act[reach_freeR(delta, shift)]
        pa = act[reach_paper(delta, shift, prefix_coin(ids[0].tolist()))]
        return set(int(x) for x in fr), set(int(x) for x in pa), int(act[0])
    fr0, pa0, g0 = reach_sets(L0)
    print(f"\n[decode reachable set @ step; honest free-r={len(fr0)} paper={len(pa0)} tokens]", flush=True)
    for Lp, name in [(gp, "coherent+"), (gm, "coherent-"), (rand_best, "random"), (adv, "adversarial")]:
        frp, pap, gp_ = reach_sets(Lp)
        print(f"  {name:11} free-r +{len(frp-fr0)}/-{len(fr0-frp)} (|{len(frp)}|)  "
              f"paper +{len(pap-pa0)}/-{len(pa0-pap)} (|{len(pap)}|)  "
              f"greedy-token {'same' if gp_==g0 else '%d->%d CHANGED'%(g0,gp_)}", flush=True)
    # union over the whole admissible band = forks the attacker can reach via attention alone
    frU = fr0 | reach_sets(gp)[0] | reach_sets(gm)[0] | reach_sets(adv)[0] | reach_sets(rand_best)[0]
    paU = pa0 | reach_sets(gp)[1] | reach_sets(gm)[1] | reach_sets(adv)[1] | reach_sets(rand_best)[1]
    print(f"\n  UNION over sampled admissible band:  free-r {len(fr0)}->{len(frU)} "
          f"(+{len(frU-fr0)})   paper {len(pa0)}->{len(paU)} (+{len(paU-pa0)})", flush=True)

if __name__ == "__main__":
    main()
