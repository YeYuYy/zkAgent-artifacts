#!/usr/bin/env python3
"""attn-on: attention-tolerance reachable-set augmentation (STRAIGHTFORWARD v0).

Shared helper used by decision_bestofN.py and eval_at_scale.py.  The zkAgent
attention row output scales by g in [1-rho, 1+rho], rho = eps/theta = 0.676%,
chosen per row by the prover (see attn_slack_logits.py for the faithful per-(layer,
head,row) model).  For a first, architecture-agnostic placeholder we scale each
attention SUBLAYER's output by ONE per-layer scalar g_L (coarser than per-row; to
be refined later).  At a decode step we produce a few perturbed logit vectors:

  honest (g=1), coherent+ (all 1+rho), coherent- (all 1-rho),
  adversarial (per-layer sign = argmax of the top1->top2 margin swing, via autograd)

attn-on reachable set = UNION of the decode-reachable sets over {honest + these}.
This over-approximates what a single committed attention configuration gives (the
union mixes several admissible configs), so it is a LOOSE upper bound on per-step
attn-augmented reachability -- fine as a placeholder; tighten when the real
per-row enumeration lands.

Honesty: (1) per-layer scalar, not per-(head,row); (2) union over a handful of
probe configs, not the full admissible band; (3) o_proj bias (if any) makes the
scale slightly inexact (Qwen o_proj has no bias; GPT-2 c_proj does).
"""
import torch

class AttnSlack:
    """Registers output hooks on every *Attention submodule; scales each by a
    per-module factor set in self.scale (id(module) -> float|Tensor|None)."""
    def __init__(self, model, rho, device, dtype):
        self.rho = float(rho); self.device = device; self.dtype = dtype
        self.scale = {}                      # id(module) -> factor or None (honest)
        self.mods = [m for m in model.modules() if type(m).__name__.endswith("Attention")]
        self.handles = [m.register_forward_hook(self._mk(m)) for m in self.mods]

    def _mk(self, mod):
        def hook(_m, _in, out):
            s = self.scale.get(id(mod))
            if s is None:
                return out
            if isinstance(out, tuple):
                return (out[0] * s,) + tuple(out[1:])
            return out * s
        return hook

    def off(self):
        self.scale.clear()

    def _coherent(self, g):
        for m in self.mods:
            self.scale[id(m)] = g

    @torch.no_grad()
    def _logits(self, ids, pos):
        return model_forward_logits(self._model, ids, pos)

    def bind(self, model):
        self._model = model; return self

    def perturbed_logits(self, ids, pos=-1):
        """Return [honest, coherent+, coherent-, adversarial] logit vectors
        (torch, float32, on cpu) at position `pos` of prefix `ids`."""
        m = self._model
        with torch.no_grad():
            self.off();            L0 = m(ids).logits[0, pos, :].float()
            self._coherent(1 + self.rho); Lp = m(ids).logits[0, pos, :].float()
            self._coherent(1 - self.rho); Lm = m(ids).logits[0, pos, :].float()
        t1, t2 = torch.topk(L0, 2).indices.tolist()
        # adversarial per-layer sign via autograd on the top1->top2 margin.
        # enable_grad: callers run under torch.no_grad(), but this backward needs grad.
        with torch.enable_grad():
            for mod in self.mods:
                self.scale[id(mod)] = torch.ones((), device=self.device, dtype=self.dtype, requires_grad=True)
            lg = m(ids).logits[0, pos, :]
            (lg[t1].float() - lg[t2].float()).backward()
        for mod in self.mods:
            g = self.scale[id(mod)]
            self.scale[id(mod)] = (1 - self.rho * torch.sign(g.grad)).detach() if g.grad is not None \
                                  else torch.tensor(1.0, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            Ladv = m(ids).logits[0, pos, :].float()
        self.off()
        return [x.detach().cpu() for x in (L0, Lp, Lm, Ladv)]


def model_forward_logits(model, ids, pos):
    return model(ids).logits[0, pos, :].float()


def union_reach(logit_vecs, reach_fn):
    """logit_vecs: list of 1-D torch logit tensors; reach_fn(logits_np)->set(token ids).
    Returns the union of reachable token ids over all vectors."""
    import numpy as np
    u = set()
    for L in logit_vecs:
        u |= reach_fn(L.numpy().astype(np.float64))
    return u
