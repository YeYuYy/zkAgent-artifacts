"""Runtime attention-tolerance patch for vLLM (pinned 0.25.0, V1 engine).

zkAgent's attention normalizer accepts any committed row mass in [theta - eps,
theta + eps]; choosing mass Theta_row instead of the canonical theta scales the
row's attention output by g = theta / Theta_row in 1 +/- rho with
rho = eps/theta = 7088/2^20 ~= 0.676% (the E1b-attn "two-pole" band model — a
documented first-order subset of the admissible set).  Scaling the attention
output is exactly the row-mass choice: o = (sum_i m_i v_i) / Theta_row =
o_canonical * theta / Theta_row, and o_proj is linear so scaling before or
after it is equivalent.

This module monkeypatches Qwen3Attention.forward / Qwen2Attention.forward so
that every attention output row computed in the CURRENT forward batch is
scaled per (layer, head) by (1 + rho * sign).  Rows already in the prefix
cache are never recomputed, so each row's perturbation is frozen at the time
the row is computed — precisely a per-row, independently admissible zeta
choice; canonical rows are the sign=0 choice.  The sign matrix lives in ONE
pre-allocated CUDA buffer that is updated in place, so decode CUDA graphs
captured at engine init (which replay the broadcast multiply against the
buffer's fixed address) stay valid across config changes; the multiply always
executes (zeros = canonical) to keep the captured graph branch-free.

IMPORTANT: install() must run BEFORE LLM(...) construction — vLLM captures
CUDA graphs during engine init, and a patch installed afterwards would never
take effect on graphed decode steps.

Public API:
    install(rho=RHO)            patch the classes (idempotent)
    set_signs(matrix)           matrix: [L, H] {−1,0,+1} (numpy/list/tensor);
                                copied in place; None -> all zeros (canonical)
    signs()                     current matrix as an [L, H] int tensor
    clear()                     restore original forwards
"""
import re

import numpy as np

RHO = 7088 / (1 << 20)

_SIGNED = {}


def _layer_index(mod):
    """model.layers.<N>.self_attn.* -> N, cached on the module instance."""
    li = getattr(mod, '_slack_layer_idx', None)
    if li is not None:
        return li
    prefix = getattr(getattr(mod, 'o_proj', None), 'prefix', '') or \
        getattr(getattr(mod, 'attn', None), 'prefix', '') or ''
    m = re.search(r'layers\.(\d+)\.', prefix)
    li = int(m.group(1)) if m else -1
    mod._slack_layer_idx = li
    return li


def _mk_layer_init(orig):
    """Wrap Qwen2/3DecoderLayer.__init__: resolve the layer index from its
    prefix and pin the shared sign buffer on the attention submodule AT
    CONSTRUCTION — before torch.compile traces anything — so the traced
    forward only reads an int attribute and a module tensor attribute (lifted
    as a graph input; in-place set_signs updates stay visible)."""
    def init(self, *args, prefix='', **kw):
        orig(self, *args, prefix=prefix, **kw)
        try:
            li = int(prefix.split('.')[2])
        except (IndexError, ValueError):
            li = -1
        sa = getattr(self, 'self_attn', None)
        buf = _SIGNED.get('buffer')
        if sa is not None and buf is not None:
            object.__setattr__(sa, '_slack_layer_idx', li)
            object.__setattr__(sa, '_slack_signs', buf)
    return init


def _mk_forward(orig, num_heads_attr='num_heads'):
    def forward(self, positions, hidden_states):
        import torch
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if getattr(self, 'q_norm', None) is not None:
            # Qwen3 qk-norm path
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                               self.head_dim)
            q = self.q_norm(q_by_head).view(q.shape)
            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                               self.head_dim)
            k = self.k_norm(k_by_head).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        # Prefer the per-module attribute (attached by attach()): torch.compile
        # lifts module tensor attributes as graph inputs, so in-place sign
        # updates stay visible in compiled/captured graphs.  The global-dict
        # fallback is only for un-attached engines.
        signs = getattr(self, '_slack_signs', None)
        if signs is None:
            signs = _SIGNED.get('buffer')
        li = getattr(self, '_slack_layer_idx', None)
        if li is None:
            li = _layer_index(self)
        if _SIGNED.get('debug_on'):
            st = _SIGNED.setdefault('debug', dict(calls=0, scaled=0, lis={}, snone=0))
            st['calls'] += 1
            if signs is None:
                st['snone'] += 1
        if signs is not None:
            h = getattr(self, num_heads_attr)
            if 0 <= li < signs.shape[0] and h <= signs.shape[1]:
                if _SIGNED.get('debug_on'):
                    st['scaled'] += 1
                    st['lis'][li] = st['lis'].get(li, 0) + 1
                scale = 1.0 + _SIGNED['rho'] * signs[li, :h].to(
                    dtype=attn_output.dtype, device=attn_output.device)
                attn_output = attn_output.view(
                    attn_output.shape[0], h, self.head_dim)
                attn_output = attn_output * scale.view(1, h, 1)
                attn_output = attn_output.view(
                    attn_output.shape[0], h * self.head_dim)
        output, _ = self.o_proj(attn_output)
        return output
    return forward


def install(rho=RHO, num_layers=64, num_heads=64, device='cuda'):
    """Patch Qwen2/Qwen3 attention forwards and pre-allocate the sign buffer.

    Must be called before vllm.LLM construction.  vLLM V1 spawns EngineCore in
    a SUBPROCESS by default — the model would run where this patch does not
    exist — so force the in-process engine here (checked at LLM init, before
    any vllm import is required).  The buffer is allocated outside any
    CUDA-graph capture pool; set_signs() updates it in place so captured
    decode graphs observe new sign matrices without re-capture.
    """
    import os
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    import torch
    from vllm.model_executor.models import qwen2, qwen3
    if _SIGNED.get('installed'):
        return
    _SIGNED.update(rho=float(rho), installed=True,
                   buffer=torch.zeros(num_layers, num_heads, dtype=torch.float32,
                                      device=device))
    for mod, cls in ((qwen2, 'Qwen2Attention'), (qwen3, 'Qwen3Attention')):
        cls_obj = getattr(mod, cls, None)
        if cls_obj is None:
            continue
        _SIGNED.setdefault('orig', {})[cls] = cls_obj.forward
        cls_obj.forward = _mk_forward(cls_obj.forward)
    for mod, cls in ((qwen2, 'Qwen2DecoderLayer'), (qwen3, 'Qwen3DecoderLayer')):
        cls_obj = getattr(mod, cls, None)
        if cls_obj is None:
            continue
        _SIGNED.setdefault('orig_init', {})[cls] = cls_obj.__init__
        cls_obj.__init__ = _mk_layer_init(cls_obj.__init__)
    return _SIGNED['buffer']


def set_signs(matrix):
    """Copy a {−1,0,+1}^{L×H} matrix into the live buffer (None -> zeros)."""
    import torch
    buf = _SIGNED.get('buffer')
    if buf is None:
        raise RuntimeError('install() first')
    if matrix is None:
        buf.zero_()
        return buf
    t = torch.as_tensor(np.asarray(matrix, dtype=np.float32))
    if t.shape != buf.shape:
        buf.zero_()
        n = min(t.shape[0], buf.shape[0]), min(t.shape[1], buf.shape[1])
        buf[:n[0], :n[1]].copy_(t[:n[0], :n[1]])
    else:
        buf.copy_(t.to(buf.device))
    return buf


def _find_model(obj, depth=0, seen=None):
    """Bounded BFS over the engine object graph for the wrapped nn.Module."""
    import torch.nn as nn
    from vllm.model_executor.models import qwen2, qwen3
    if depth > 4 or obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return None
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return None
    seen.add(id(obj))
    if isinstance(obj, nn.Module):
        has_attn = any(isinstance(m, (qwen2.Qwen2Attention, qwen3.Qwen3Attention))
                       for m in obj.modules())
        if has_attn:
            return obj
    kids = []
    if isinstance(obj, dict):
        kids = list(obj.values())
    elif isinstance(obj, (list, tuple, set)):
        kids = list(obj)
    else:
        kids = [v for k, v in vars(obj).items()
                if not k.startswith('_') or k in
                ('_engine_core', 'engine_core', 'model_executor', 'driver_worker',
                 'model_runner', 'model')] if hasattr(obj, '__dict__') else []
        kids += [getattr(obj, k, None) for k in
                 ('engine_core', 'model_executor', 'driver_worker', 'model_runner',
                  'model') if hasattr(obj, k)]
    for k in kids[:64]:
        found = _find_model(k, depth + 1, seen)
        if found is not None:
            return found
    return None


def attach(llm):
    """Point every attention module's `_slack_signs` at the shared buffer.

    torch.compile lifts module tensor attributes as graph inputs (identity
    guard), so compiled/captured graphs observe in-place set_signs() updates;
    a global-dict lookup can be constant-specialised to the buffer's initial
    zeros.  Call after LLM construction; harmless before any generate.
    """
    from vllm.model_executor.models import qwen2, qwen3
    buf = _SIGNED.get('buffer')
    if buf is None:
        raise RuntimeError('install() first')
    model = _find_model(llm)
    if model is None:
        return 0
    n = 0
    for mod in model.modules():
        if isinstance(mod, (qwen2.Qwen2Attention, qwen3.Qwen3Attention)):
            object.__setattr__(mod, '_slack_signs', buf)
            n += 1
    return n


def signs():
    import torch
    return _SIGNED['buffer'].detach().cpu().to(torch.int32)


def clear():
    from vllm.model_executor.models import qwen2, qwen3
    for mod, cls in ((qwen2, 'Qwen2Attention'), (qwen3, 'Qwen3Attention')):
        cls_obj = getattr(mod, cls, None)
        orig = _SIGNED.get('orig', {}).get(cls)
        if cls_obj is not None and orig is not None:
            cls_obj.forward = orig
    for mod, cls in ((qwen2, 'Qwen2DecoderLayer'), (qwen3, 'Qwen3DecoderLayer')):
        cls_obj = getattr(mod, cls, None)
        orig = _SIGNED.get('orig_init', {}).get(cls)
        if cls_obj is not None and orig is not None:
            cls_obj.__init__ = orig
    _SIGNED.pop('buffer', None)
    _SIGNED.pop('installed', None)
