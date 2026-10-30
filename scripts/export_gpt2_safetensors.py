#!/usr/bin/env python3
"""Export GPT-2 safetensors to the layout written by scripts/download_gpt2_124m.py, without torch.

Usage: export_gpt2_safetensors.py MODEL_DIR
MODEL_DIR must contain model.safetensors and config.json from openai-community/gpt2.
"""
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

out = Path(sys.argv[1])
t = load_file(str(out / "model.safetensors"))
t = {k.removeprefix("transformer."): v for k, v in t.items()}
cfg = json.loads((out / "config.json").read_text())

(out / "config.txt").write_text("\n".join([
    f"vocab_size={cfg['vocab_size']}",
    f"n_embd={cfg['n_embd']}",
    f"n_layer={cfg['n_layer']}",
    f"n_head={cfg['n_head']}",
    f"n_positions={cfg['n_positions']}",
    f"head_dim={cfg['n_embd'] // cfg['n_head']}",
    f"ffn_dim={cfg['n_embd'] * 4}",
]) + "\n")


def save(name, arr, path):
    np.ascontiguousarray(arr, dtype=np.float32).tofile(path / name)


save("wte.bin", t["wte.weight"], out)
save("wpe.bin", t["wpe.weight"], out)
for i in range(cfg["n_layer"]):
    d = out / f"layer_{i}"
    d.mkdir(exist_ok=True)
    p = f"h.{i}."
    # HF Conv1D stores [in, out]; the reference exporter writes the transpose.
    save("c_attn_w.bin", t[p + "attn.c_attn.weight"].T, d)
    save("c_attn_b.bin", t[p + "attn.c_attn.bias"], d)
    save("c_proj_w.bin", t[p + "attn.c_proj.weight"].T, d)
    save("c_proj_b.bin", t[p + "attn.c_proj.bias"], d)
    save("c_fc_w.bin", t[p + "mlp.c_fc.weight"].T, d)
    save("c_fc_b.bin", t[p + "mlp.c_fc.bias"], d)
    save("mlp_proj_w.bin", t[p + "mlp.c_proj.weight"].T, d)
    save("mlp_proj_b.bin", t[p + "mlp.c_proj.bias"], d)
    save("ln_1_w.bin", t[p + "ln_1.weight"], d)
    save("ln_1_b.bin", t[p + "ln_1.bias"], d)
    save("ln_2_w.bin", t[p + "ln_2.weight"], d)
    save("ln_2_b.bin", t[p + "ln_2.bias"], d)
save("ln_f_w.bin", t["ln_f.weight"], out)
save("ln_f_b.bin", t["ln_f.bias"], out)
print("export complete ->", out)
