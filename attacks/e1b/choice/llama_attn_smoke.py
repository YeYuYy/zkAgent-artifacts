"""GPU smoke for the LlamaAttention attention-lever patch (P0 item 3).

Boots R1-Distill-Llama-8B fp32 in vLLM 0.33, installs the attn_slack patch,
and checks functionally, in order:
  1. the live model's attention modules are instances of the patched
     LlamaAttention class and carry the construction-time sign pinning
     (`_slack_signs` set by the wrapped LlamaDecoderLayer.__init__) with
     correct per-layer indices 0..L-1;
  2. greedy completions are bitwise identical with the sign buffer zeroed
     before/after a perturbation (restore check);
  3. an all-negative sign matrix measurably moves the top-k logprobs
     (perturbation check; fp32 + eager — plumbing, not circuit faithfulness).

attach() itself is *informational* here: its bounded BFS (depth 4) cannot
reach the model through vLLM 0.33's InprocClient graph, so it returns 0 on
this env for every model family (Qwen campaigns log `attached=0` too and
still work — construction-time pinning is the operative binding).

Run from a staged snapshot via the standard worker bootstrap, e.g.
  env ZK_INV=1 PYTHONPATH=<py312-vllm35> <vllm-env>/bin/python -c \
    "import sys; sys.path.insert(0, '<py312-vllm35>'); import site; \
     site.addsitedir('<py312-vllm35>'); import runpy; \
     runpy.run_module('experiments.llama_attn_smoke', run_name='__main__')"

PYTHONPATH must be set in the *environment*, not just sys.path: vLLM 0.33's
registry inspects fresh architectures in a subprocess
(`sys.executable -m vllm.model_executor.models.registry`), which inherits
env vars but not in-process sys.path edits. Without env PYTHONPATH that
subprocess imports the interpreter's own vLLM (0.11 in vllm-env) and the
resulting _ModelInfo pickles back with mismatched fields, crashing engine
init with "_ModelInfo object has no attribute 'supports_kv_cache_grouping'"
(this is exactly what killed llamasmoke2 on 2026-11-16; the choice jobs
never hit it because they always carry PYTHONPATH in argv).
"""
import json
import os
import sys

SITE = '/mnt/shared-storage-user/ai4good1-share/guodadi/zk_agent_gpu/py312-vllm35'
# belt and suspenders: make the vLLM 0.33 site-dir authoritative for this
# process AND for any subprocess vLLM spawns (registry inspection).
for p in (SITE, ''):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.environ['PYTHONPATH'] = SITE + os.pathsep + os.environ.get('PYTHONPATH', '')

import vllm
assert vllm.__version__.startswith('0.33'), f'unexpected vLLM {vllm.__version__}'

MODEL = ('/mnt/shared-storage-user/ai4good1-share/guodadi/zk_agent_gpu/models/'
         'R1-Distill-Llama-8B-6a6f4aa4197940add57724a7707d069478df56b1')

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from experiments import attn_slack_vllm as slack
slack.install()

from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, dtype='float32', gpu_memory_utilization=0.9,
          max_model_len=4096, enforce_eager=True)


def _find_model_unbounded(obj):
    """BFS without a depth cap over the engine graph (attach() caps at 4 and
    cannot reach model_runner.model on vLLM 0.33)."""
    import torch.nn as nn
    classes = slack._attention_classes()
    seen, queue = set(), [obj]
    while queue:
        o = queue.pop(0)
        if o is None or id(o) in seen or isinstance(o, (str, bytes, int, float, bool)):
            continue
        seen.add(id(o))
        if isinstance(o, nn.Module) and any(isinstance(m, classes) for m in o.modules()):
            return o
        kids = []
        if isinstance(o, dict):
            kids = list(o.values())
        elif isinstance(o, (list, tuple, set)):
            kids = list(o)
        elif hasattr(o, '__dict__'):
            kids = [v for k, v in vars(o).items()
                    if not k.startswith('_') or k in
                    ('_engine_core', 'engine_core', 'model_executor', 'driver_worker',
                     'model_runner', 'model')]
        kids += [getattr(o, k, None) for k in
                 ('engine_core', 'model_executor', 'driver_worker', 'model_runner',
                  'model') if hasattr(o, k)]
        queue.extend(kids[:64])
    return None


model = _find_model_unbounded(llm)
classes = slack._attention_classes()
mods = [m for m in model.modules() if isinstance(m, classes)] if model is not None else []
pinned = [m for m in mods if getattr(m, '_slack_signs', None) is not None]
idx = sorted(getattr(m, '_slack_layer_idx', None) for m in mods)
diag = {
    'vllm': vllm.__version__,
    'attach_reported': slack.attach(llm),  # informational; expected 0 on 0.33
    'model_found': model is not None,
    'attn_modules': len(mods),
    'pinned': len(pinned),
    'attn_classes': sorted({type(m).__module__ + '.' + type(m).__name__ for m in mods}),
    'layer_idx': idx,
}
print(json.dumps(diag, indent=1), flush=True)
assert mods, 'no patched-family attention modules found in the live model'
assert pinned and len(pinned) == len(mods), \
    'attention modules lack _slack_signs: install() must run before LLM() ' \
    'so LlamaDecoderLayer.__init__ pins the buffer at construction'
n_layers = len(mods)
assert idx == list(range(n_layers)), \
    f'bad layer indices {idx[:8]}... for {n_layers} layers; per-layer ' \
    'sign targeting would be misaligned'

tok = llm.get_tokenizer()
prompt = tok.apply_chat_template([dict(role='user', content='What is 7 times 8?')],
                                 add_generation_prompt=True, tokenize=False)
sp = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5)


def run():
    r = llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0]
    # (token_id, logprob) pairs per position — comparable across runs
    return (r.text, [[(tid, round(lp.logprob, 10)) for tid, lp in pos.items()]
                     for pos in (r.logprobs or [])])


slack.set_signs(None)
base_text, base_lp = run()

buf = slack.set_signs(None)
signs = np.full((buf.shape[0], buf.shape[1]), -1.0, dtype=np.float32)
slack.set_signs(signs)
neg_text, neg_lp = run()

slack.set_signs(None)
base2_text, base2_lp = run()

out = {
    **diag,
    'base_text': base_text,
    'neg_text': neg_text,
    'restore_identical': (base2_text, base2_lp) == (base_text, base_lp),
    'perturbation_changed': base_lp != neg_lp or base_text != neg_text,
    'max_abs_logprob_shift': max((abs(a[1] - b[1])
                                  for pa, pb in zip(base_lp, neg_lp)
                                  for a, b in zip(pa, pb)), default=0.0),
}
print(json.dumps(out, indent=1), flush=True)
slack.clear()
ok = out['restore_identical'] and out['perturbation_changed']
print('LLAMA PATCH SMOKE:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
