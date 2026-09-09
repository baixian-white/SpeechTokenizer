import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path("h:/H-CODE/speechtokenizer")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

CFG = "output/experiments/redoB_base_handcrafted_seanet_20260620_seed42/configs/scit_speech_base_config.json"
with open(CFG, encoding="utf-8-sig") as f:
    cfg = json.load(f)

results = {}
import math
prod = math.prod(cfg["strides"])
results["strides_product==320"] = (prod == 320, prod)
results["no_nas_encoder_config_key"] = ("nas_encoder_config" not in cfg, "nas_encoder_config" in cfg)
results["no_exp1_handoff_key"] = ("exp1_handoff" not in cfg, "exp1_handoff" in cfg)
results["distill_loss_lambda==30.0"] = (cfg["distill_loss_lambda"] == 30.0, cfg["distill_loss_lambda"])
results["seed==42"] = (cfg["seed"] == 42, cfg["seed"])
results["dimension==1024"] = (cfg["dimension"] == 1024, cfg["dimension"])
results["n_q==3"] = (cfg["n_q"] == 3, cfg["n_q"])
results["codebook_size==1024"] = (cfg["codebook_size"] == 1024, cfg["codebook_size"])

# file resolution
for key in ["train_files", "valid_files"]:
    p = cfg[key]
    exists = os.path.isfile(p)
    n = 0
    first_ok = None
    if exists:
        with open(p, encoding="utf-8") as f:
            ls = [l for l in f if l.strip()]
        n = len(ls)
        a, b = ls[0].strip().split("\t")
        first_ok = os.path.isfile(a) and os.path.isfile(b)
    results[f"{key}_exists"] = (exists, p)
    results[f"{key}_count"] = (True, n)
    results[f"{key}_firstline_resolves"] = (first_ok, first_ok)

# build the model to confirm hand-designed SEANet + param count
from speechtokenizer import SpeechTokenizer
gen = SpeechTokenizer(cfg)
cls_name = type(gen).__name__
total_params = sum(p.numel() for p in gen.parameters())
enc_params = sum(p.numel() for p in gen.encoder.parameters())
results["generator_class"] = (cls_name == "SpeechTokenizer", cls_name)
results["downsample_rate"] = (gen.downsample_rate == 320, int(gen.downsample_rate))

print("=== B0 ASSERTIONS ===")
ok_all = True
for k, (ok, val) in results.items():
    flag = "PASS" if ok else "FAIL"
    if not ok:
        ok_all = False
    print(f"[{flag}] {k}: {val}")
print(f"generator total params: {total_params:,}")
print(f"generator encoder params: {enc_params:,}")
print(f"encoder strides (ratios): {cfg['strides']}")
print("ALL_PASS" if ok_all else "SOME_FAIL")
PY_DONE = True
