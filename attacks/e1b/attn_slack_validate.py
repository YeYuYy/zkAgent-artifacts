"""Validate the vLLM attention-slack patch against HF eager semantics.

Findings from the first version (kept as measured facts):
  * bf16 prefix-cache reads are NOT bit-reproducible across batch shapes: a
    fully-cached probe (only the last row recomputed) drifts ~0.13 logprob on
    top tokens vs the fresh compute.  So (a) per-token delta correlation
    between backends is hopeless at this granularity, and (b) cache
    "restoration" after an S-probe is exact only to this recompute noise —
    the runner's protocol stays sound (it records the readout it acts on),
    but "exact" must be claimed as within-noise.

This version therefore measures:
  A. repeat determinism, same call pattern, cache ON        (should be ~0)
  B. recompute-vs-fresh noise, cache ON                     (documented ~0.1)
  C. clean perturbation effect, cache OFF: margin shift (top1-top2) and
     max |logprob delta| under {coherent+, coherent-, random x K}
  D. HF-eager cross-check: same prompts/patterns, float32, per-(layer,head)
     pre-o_proj hook (attn_slack_logits.py convention); correlation between
     vLLM and HF MARGIN SHIFTS across (prompt, pattern) pairs.

Pass: A ~ 0; C median |margin shift| under random > 5x A; D pearson > 0.5.
"""
import argparse
import json
from pathlib import Path

import numpy as np

PROMPTS = [
    'Web task: Find a low-priced refurbished Microsoft Xbox one console in white color.\n\nCandidate products:\n1. Xbox One Console White $199\n2. PlayStation 4 $249\n3. Xbox Controller $59\n4. Nintendo Switch $299\n\nWhich product',
    'Shopping instruction: I want a large blue ocean painting under $100 for my living room\n\nCandidate products:\n1. Abstract Ocean Canvas Art 24x36, $89.99\n2. Blue Whale Print 12x18, $25\n3. Modern Seascape Oil Painting 36x24, $299\n4. Coastal Wall Decor Set, $45\n\nWhich product',
    'Situation: Your friend asks you to cover for them at work while they attend a family emergency.\n\nOption A: Agree to cover the shift without asking details\nOption B: Refuse and suggest they find someone else\n\nWhich action would you take? Reply with',
    'Question: A store sells 3 kinds of soap. If each kind has 7 bars and 2 bars are sold from each kind, how many bars remain? Answer step by step. The final answer is',
]


def margins(lp):
    if lp is None or len(lp) < 2:
        return None
    top = sorted(lp.items(), key=lambda kv: -kv[1])[:2]
    return top[0][1] - top[1][1], top[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--gpu-mem', type=float, default=0.55)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--rand-patterns', type=int, default=4)
    ap.add_argument('--enforce-eager', action='store_true',
                    help='disable torch.compile + CUDA graphs (isolates '
                         'compile-time constant specialisation of the sign '
                         'buffer)')
    ap.add_argument('--phase', choices=['cached', 'clean', 'hf'], default='all',
                    help='vLLM engines do not release GPU memory in-process; '
                         'run each phase as its own process/job. "hf" merges '
                         'with the clean phase output.')
    ap.add_argument('--clean-report', type=Path,
                    default=Path('/mnt/shared-storage-user/ai4good1-share/'
                                 'guodadi/zk_agent_gpu/runs-choice/'
                                 'attn-validate-clean.json'))
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    import torch
    from experiments import attn_slack_vllm as slack
    slack.install()
    # dict-mutating counters are eager-only (dynamo would graph-break on them)
    slack._SIGNED['debug_on'] = bool(a.enforce_eager)
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models import qwen3 as _q3
    print('[validate] Qwen3Attention.forward =', _q3.Qwen3Attention.forward.__qualname__,
          flush=True)
    SP = SamplingParams

    def boot(cache):
        return LLM(model=a.model, dtype='bfloat16', gpu_memory_utilization=a.gpu_mem,
                   max_model_len=4096, enable_prefix_caching=cache, max_logprobs=64,
                   disable_log_stats=True, enforce_eager=a.enforce_eager)

    def read(llm, ids, signs):
        slack.set_signs(signs)
        dbg0 = dict(slack._SIGNED.get('debug') or {})
        sp = SP(temperature=0.0, max_tokens=1, logprobs=64, detokenize=False)
        out = llm.generate([{'prompt_token_ids': ids}], sp, use_tqdm=False)
        slack.set_signs(None)
        dbg1 = slack._SIGNED.get('debug') or {}
        if dbg1.get('calls', 0) > dbg0.get('calls', 0):
            print('[debug] patched forwards this call:',
                  dbg1.get('calls', 0) - dbg0.get('calls', 0),
                  'scaled:', dbg1.get('scaled', 0) - dbg0.get('scaled', 0),
                  'signs-none:', dbg1.get('snone', 0) - dbg0.get('snone', 0),
                  'layers:', len(dbg1.get('lis', {})), flush=True)
        lp = out[0].outputs[0].logprobs[0]
        return {int(t): float(l.logprob) for t, l in lp.items()}

    report = {}
    rng = np.random.default_rng(a.seed)

    # ---- cache-ON engine: determinism + recompute noise -------------------
    if a.phase in ('all', 'cached'):
        llm = boot(True)
        print('[attach] attention modules:', slack.attach(llm), flush=True)
        tok = llm.get_tokenizer()
        ids_list = [tok.encode(p, add_special_tokens=False) for p in PROMPTS]
        n_layers = llm.llm_engine.model_config.hf_config.num_hidden_layers
        n_heads = llm.llm_engine.model_config.hf_config.num_attention_heads
        S_rands = [rng.integers(-1, 2, size=(n_layers, n_heads))
                   for _ in range(a.rand_patterns)]
        S_coh = np.ones((n_layers, n_heads), dtype=int)
        ids = ids_list[0]
        b1 = read(llm, ids, None)
        b2 = read(llm, ids, None)
        A = max(abs(b1[t] - b2[t]) for t in b1)
        ids_fresh = ids[:-1] + [ids[-1]]  # second read recomputes the last row
        b3 = read(llm, ids_fresh, None)
        B = max(abs(b1[t] - b3[t]) for t in b1)
        p1 = read(llm, ids, S_rands[0])
        b4 = read(llm, ids, None)
        C_restore = max(abs(b1[t] - b4[t]) for t in b1)
        report.update(n_layers=n_layers, n_heads=n_heads, rho=slack.RHO,
                      repeat_determinism_max=float(A),
                      recompute_vs_fresh_max=float(B),
                      restore_after_probe_max=float(C_restore))
        print(json.dumps(report, indent=1), flush=True)
        if a.out:
            a.out.write_text(json.dumps(report, indent=2))
        return 0

    # ---- cache-OFF engine: clean perturbation deltas ----------------------
    llm = boot(False)
    print('[attach] attention modules:', slack.attach(llm), flush=True)
    tok = llm.get_tokenizer()
    ids_list = [tok.encode(p, add_special_tokens=False) for p in PROMPTS]
    n_layers = llm.llm_engine.model_config.hf_config.num_hidden_layers
    n_heads = llm.llm_engine.model_config.hf_config.num_attention_heads
    S_rands = [rng.integers(-1, 2, size=(n_layers, n_heads))
               for _ in range(a.rand_patterns)]
    S_coh = np.ones((n_layers, n_heads), dtype=int)
    v_shifts = []
    for i, ids in enumerate(ids_list):
        base = read(llm, ids, None)
        base_rep = read(llm, ids, None)
        det = max(abs(base[t] - base_rep[t]) for t in base)
        m0 = margins(base)
        for name, S in [('coherent+', S_coh), ('coherent-', -S_coh)] + \
                [(f'random{k}', S_rands[k]) for k in range(a.rand_patterns)]:
            lp = read(llm, ids, S)
            dmax = max(abs(lp.get(t, 0.) - base.get(t, 0.)) for t in set(lp) | set(base))
            m1 = margins(lp)
            v_shifts.append(dict(prompt=i, pattern=name,
                                 margin_shift=float(m1[0] - m0[0]) if m1 else None,
                                 top1_same=bool(m1 and m1[1] == m0[1]),
                                 max_delta=float(dmax), determinism=float(det)))
    det0 = np.median([v['determinism'] for v in v_shifts])
    rand_ms = [abs(v['margin_shift']) for v in v_shifts
               if v['pattern'].startswith('random') and v['margin_shift'] is not None]
    coh_ms = [abs(v['margin_shift']) for v in v_shifts
              if v['pattern'].startswith('coherent') and v['margin_shift'] is not None]
    report.update(n_layers=n_layers, n_heads=n_heads, rho=slack.RHO,
                  cacheoff_repeat_determinism_median=float(det0),
                  cacheoff_random_margin_shift_median=float(np.median(rand_ms)),
                  cacheoff_random_margin_shift_max=float(np.max(rand_ms)),
                  cacheoff_coherent_margin_shift_median=float(np.median(coh_ms)),
                  cacheoff_top1_flips=int(sum(not v['top1_same'] for v in v_shifts)))
    del llm
    import gc; gc.collect(); torch.cuda.empty_cache()
    if a.phase == 'clean':
        report['v_shifts'] = v_shifts
        out = a.out or a.clean_report
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != 'v_shifts'},
                         indent=1), flush=True)
        return 0

    # ---- HF eager reference (float32) on the same prompts/patterns --------
    if a.phase == 'hf':
        clean = json.loads(a.clean_report.read_text())
        v_shifts = clean['v_shifts']
        report.update({k: v for k, v in clean.items() if k != 'v_shifts'})
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        ids_list = [tok.encode(p, add_special_tokens=False) for p in PROMPTS]
        n_layers = int(clean['n_layers'])
        n_heads = int(clean['n_heads'])
    from transformers import AutoModelForCausalLM as HF
    hf = HF.from_pretrained(a.model, dtype=torch.float32,
                            attn_implementation='eager', local_files_only=True
                            ).cuda().eval()
    SCALE = {}
    import transformers.models.qwen3.modeling_qwen3 as g3
    import transformers.models.qwen2.modeling_qwen2 as g2
    for g in (g3, g2):
        if not hasattr(g, 'eager_attention_forward'):
            continue
        orig = g.eager_attention_forward

        def patched(module, query, key, value, attention_mask, scaling, dropout=0.0,
                    **kw):
            res = orig(module, query, key, value, attention_mask, scaling,
                       dropout=dropout, **kw)
            out, rest = (res[0], res[1:]) if isinstance(res, tuple) else (res, ())
            s = SCALE.get(module.layer_idx)
            if s is not None:
                s_t = torch.as_tensor(s, dtype=out.dtype, device=out.device)
                out = out * s_t.view(1, 1, -1, 1) if out.dim() == 4 \
                    else out * s_t.view(1, -1, 1)
            return (out,) + rest if rest else out
        g.eager_attention_forward = patched

    def hf_margin(ids, S):
        SCALE.clear()
        if S is not None:
            for li in range(n_layers):
                SCALE[li] = 1.0 + slack.RHO * S[li]
        x = torch.tensor([ids]).cuda()
        with torch.no_grad():
            lg = hf(x).logits[0, -1, :]
        t1, t2 = torch.topk(lg, 2)
        return float(t1[0] - t2[0])

    pairs_v, pairs_h, names = [], [], []
    for i, ids in enumerate(ids_list):
        m0 = hf_margin(ids, None)
        for name, S in [('coherent+', S_coh), ('coherent-', -S_coh)] + \
                [(f'random{k}', S_rands[k]) for k in range(a.rand_patterns)]:
            m1 = hf_margin(ids, S)
            v = next(x for x in v_shifts if x['prompt'] == i and x['pattern'] == name)
            if v['margin_shift'] is not None:
                pairs_v.append(v['margin_shift'])
                pairs_h.append(m1 - m0)
                names.append(name)
    pear = float(np.corrcoef(pairs_v, pairs_h)[0, 1]) if len(pairs_v) > 2 else None
    report.update(hf_margin_shift_pearson=pear, n_pairs=len(pairs_v),
                  hf_random_margin_shift_median=float(np.median(
                      [abs(h) for h, nm in zip(pairs_h, names)
                       if nm.startswith('random')])))

    ok = (report.get('cacheoff_repeat_determinism_median', 1.) < 1e-3
          and report.get('cacheoff_random_margin_shift_median', 0.)
          > 5 * max(report.get('cacheoff_repeat_determinism_median', 1e-6), 1e-6)
          and (pear or 0) > 0.5)
    report['pass'] = bool(ok)
    report['v_shifts'] = v_shifts
    print(json.dumps({k: v for k, v in report.items() if k != 'v_shifts'}, indent=2))
    if a.out:
        a.out.write_text(json.dumps(report, indent=2))
    del hf
    torch.cuda.empty_cache()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
