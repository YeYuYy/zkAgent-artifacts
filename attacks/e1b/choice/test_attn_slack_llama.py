"""CPU stub test for the LlamaAttention extension of attn_slack_vllm.

Builds a fake `vllm.model_executor.models` package whose Llama/Qwen2/Qwen3
classes mirror the exact next-style forward bodies found in the installed envs
(vLLM 0.11.0 in vllm-env and 0.33.0 in py312-vllm35 — both identical for
LlamaAttention), then verifies:

  1. install() patches LlamaAttention (and still Qwen2/Qwen3), idempotently;
  2. install() REFUSES a legacy-signature LlamaAttention before any state is
     allocated;
  3. decoder-layer __init__ wrapping pins layer indices + the shared buffer on
     the Llama attention submodule;
  4. forward rows are scaled by exactly (1 + rho * sign) per (layer, head);
  5. positions-limited set_signs only touches the given absolute positions;
  6. set_signs(None) is bitwise-identity canonical;
  7. attach() finds Llama attention modules through a fake engine graph;
  8. clear() restores the original forwards/inits and drops the buffer;
  9. the Qwen3 qk-norm path still scales (no regression from this change).

Run: cd /root/zk_agent && python3 -I tests/test_attn_slack_llama.py
"""
import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RHO = 7088 / (1 << 20)


# ---------------------------------------------------------------- fake vllm
class _Linear:
    """vLLM-style linear: returns (output, bias=None)."""

    def __init__(self, din, dout, seed):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(dout, din, generator=g)

    def __call__(self, x):
        return x @ self.w.t(), None


class _Rotary:
    def __call__(self, positions, q, k):
        return q, k


class _Attn:
    """Value passthrough per head: output[t] = v[t] per head, so per-head
    scaling is directly observable."""

    def __call__(self, q, k, v):
        return v


class _Norm(nn.Module):
    def forward(self, x):
        return x


def _mk_attention_cls(qk_norm=False):
    class _Attention(nn.Module):
        def __init__(self, num_heads=2, head_dim=4, seed=0):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            d = num_heads * head_dim
            self.q_size = d
            self.kv_size = d
            self.qkv_proj = _Linear(d, 3 * d, seed)
            # identity o_proj: per-head scaling is applied BEFORE o_proj, and a
            # head-mixing projection would make per-head factors non-trivially
            # observable; identity keeps the assertion exact
            self.o_proj = lambda x: (x, None)
            self.rotary_emb = _Rotary()
            self.attn = _Attn()
            if qk_norm:
                self.q_norm = _Norm()
                self.k_norm = _Norm()

        def forward(self, positions, hidden_states):
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            if getattr(self, 'q_norm', None) is not None:
                q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                                   self.head_dim)
                q = self.q_norm(q_by_head).view(q.shape)
                k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                                   self.head_dim)
                k = self.k_norm(k_by_head).view(k.shape)
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output

    return _Attention


def _mk_decoder_cls(attn_cls):
    class _DecoderLayer(nn.Module):
        def __init__(self, config=None, prefix=''):
            super().__init__()
            self.self_attn = attn_cls()

        def forward(self, positions, hidden_states):
            return self.self_attn(positions, hidden_states)

    return _DecoderLayer


def install_fake_vllm(llama_legacy_sig=False):
    """(Re)build the fake vllv package + a fresh attn_slack_vllm module."""
    for name in [n for n in sys.modules if n == 'vllm' or n.startswith('vllm.')]:
        del sys.modules[name]
    pkg = types.ModuleType('vllm')
    me = types.ModuleType('vllm.model_executor')
    mem = types.ModuleType('vllm.model_executor.models')
    qwen2 = types.ModuleType('vllm.model_executor.models.qwen2')
    qwen3 = types.ModuleType('vllm.model_executor.models.qwen3')
    llama = types.ModuleType('vllm.model_executor.models.llama')

    qwen2.Qwen2Attention = _mk_attention_cls(qk_norm=False)
    qwen3.Qwen3Attention = _mk_attention_cls(qk_norm=True)
    qwen2.Qwen2DecoderLayer = _mk_decoder_cls(qwen2.Qwen2Attention)
    qwen3.Qwen3DecoderLayer = _mk_decoder_cls(qwen3.Qwen3Attention)

    class _LegacyLlama(nn.Module):
        def forward(self, key, value, kv_cache, attn_metadata):  # old vLLM style
            raise AssertionError('legacy forward must never run')

    llama.LlamaAttention = (_LegacyLlama if llama_legacy_sig
                            else _mk_attention_cls(qk_norm=False))
    llama.LlamaDecoderLayer = _mk_decoder_cls(llama.LlamaAttention)

    for name, mod in [('vllm', pkg), ('vllm.model_executor', me),
                      ('vllm.model_executor.models', mem),
                      ('vllm.model_executor.models.qwen2', qwen2),
                      ('vllm.model_executor.models.qwen3', qwen3),
                      ('vllm.model_executor.models.llama', llama)]:
        sys.modules[name] = mod
    pkg.model_executor = me
    me.models = mem
    mem.qwen2, mem.qwen3, mem.llama = qwen2, qwen3, llama

    slack = importlib.import_module('experiments.attn_slack_vllm')
    slack = importlib.reload(slack)  # fresh _SIGNED
    return slack, sys.modules['vllm.model_executor.models.llama'], \
        sys.modules['vllm.model_executor.models.qwen3']


def expected(out0, sign_row, num_heads):
    """Apply ONE layer's per-head factors to a canonical [T, H*D] output."""
    e = out0.view(out0.shape[0], num_heads, -1).clone()
    for h in range(num_heads):
        e[:, h] *= (1.0 + RHO * float(sign_row[h]))
    return e.view(out0.shape)


def main():
    # --- 2. legacy-signature guard: install must refuse, state untouched ----
    slack, llama_mod, _ = install_fake_vllm(llama_legacy_sig=True)
    try:
        slack.install(device='cpu')
        raise AssertionError('legacy-signature LlamaAttention was accepted')
    except RuntimeError as e:
        assert 'LlamaAttention.forward has unexpected signature' in str(e)
    assert not slack._SIGNED.get('installed'), 'guard must fire before _alloc'
    print('PASS legacy-signature guard')

    # --- fresh proper stubs --------------------------------------------------
    slack, llama_mod, qwen3_mod = install_fake_vllm()
    LlamaAttention = llama_mod.LlamaAttention
    Qwen3Attention = qwen3_mod.Qwen3Attention
    orig_llama_fwd = LlamaAttention.forward
    orig_llama_init = llama_mod.LlamaDecoderLayer.__init__
    orig_q3_fwd = Qwen3Attention.forward

    buf = slack.install(device='cpu')
    assert LlamaAttention.forward is not orig_llama_fwd, 'Llama not patched'
    assert Qwen3Attention.forward is not orig_q3_fwd, 'Qwen3 not patched'
    slack.install(device='cpu')  # idempotent
    assert LlamaAttention.forward is not orig_llama_fwd
    print('PASS install patches Llama + Qwen (idempotent)')

    # --- 3. decoder __init__ pins layer idx + buffer ------------------------
    layers = [llama_mod.LlamaDecoderLayer(prefix=f'model.layers.{i}') for i in (0, 1)]
    for i, layer in enumerate(layers):
        assert layer.self_attn._slack_layer_idx == i, 'layer index not pinned'
        assert layer.self_attn._slack_signs is buf, 'sign buffer not pinned'
    print('PASS decoder init pins layer idx + buffer')

    torch.manual_seed(7)
    T, H, D = 3, 2, 4
    x = torch.randn(T, H * D)
    positions = torch.tensor([5, 6, 7])

    # --- 6. canonical (bitwise identity) ------------------------------------
    slack.set_signs(None)
    out0 = layers[0](positions, x)
    with torch.no_grad():
        ref0 = orig_llama_fwd(layers[0].self_attn, positions, x)
    assert torch.equal(out0, ref0), 'canonical must be bitwise identical'
    print('PASS canonical bitwise identity')

    # --- 4. per-(layer, head) scaling ---------------------------------------
    S = torch.tensor([[+1.0, -1.0], [0.0, +1.0]])
    slack.set_signs(S)
    out_layer0 = layers[0](positions, x)
    out_layer1 = layers[1](positions, x)
    assert torch.allclose(out_layer0, expected(ref0, S[0], H), atol=1e-6), \
        f'layer0 mismatch: {(out_layer0 - expected(ref0, S[0], H)).abs().max()}'
    assert torch.allclose(out_layer1, expected(ref0, S[1], H), atol=1e-6), \
        f'layer1 mismatch: {(out_layer1 - expected(ref0, S[1], H)).abs().max()}'
    print('PASS per-(layer,head) sign scaling on Llama')

    # --- 5. positions-limited signs ------------------------------------------
    slack.set_signs(S, positions=[6])
    out_pos = layers[0](positions, x)
    want = ref0.view(T, H, D).clone()
    want[1] *= (1 + RHO * torch.tensor([1.0, -1.0])).view(H, 1)
    assert torch.allclose(out_pos, want.view(T, H * D), atol=1e-6), \
        'position-limited scaling mismatch'
    print('PASS positions-limited signs')

    # --- 7. attach through a fake engine graph -------------------------------
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([l.self_attn for l in layers])

    class _Engine:
        def __init__(self):
            self.model = _Model()

    n = slack.attach(_Engine())
    assert n == 2, f'attach found {n} Llama attention modules, want 2'
    print('PASS attach finds Llama attention modules')

    # --- 9. Qwen3 qk-norm path still scales (no regression) ------------------
    q3_layer = qwen3_mod.Qwen3DecoderLayer(prefix='model.layers.0')
    slack.set_signs(S)
    out_q3 = q3_layer(positions, x)
    with torch.no_grad():
        ref_q3 = orig_q3_fwd(q3_layer.self_attn, positions, x)
    assert torch.allclose(out_q3, expected(ref_q3, S[0], H), atol=1e-6), \
        'Qwen3 qk-norm path regressed'
    print('PASS Qwen3 qk-norm path unchanged')

    # --- 8. clear() restores everything --------------------------------------
    slack.clear()
    assert LlamaAttention.forward is orig_llama_fwd
    assert llama_mod.LlamaDecoderLayer.__init__ is orig_llama_init
    assert Qwen3Attention.forward is orig_q3_fwd
    assert 'buffer' not in slack._SIGNED and 'installed' not in slack._SIGNED
    try:
        slack.set_signs(None)
        raise AssertionError('set_signs after clear must raise')
    except RuntimeError:
        pass
    with torch.no_grad():
        after = orig_llama_fwd(layers[0].self_attn, positions, x)
    assert torch.equal(after, ref0), 'restored forward changed numerics'
    print('PASS clear() restores originals')

    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
