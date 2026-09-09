#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp19: run Base-vs-LCA clean objective eval on the VCTK 300 sample list.
Mirrors exp7's proven invocation of evaluate_clean_large_nosave.py. GPU.
Writes metrics/full_clean_results.csv (300x3Lx2model=1800 rows) + summary.

NOT launched until VCTK is extracted and the sample list is built.
Run: conda run -n speechtokenizer python <this file>
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"h:\H-CODE\speechtokenizer")
RUN = ROOT / "output/experiments/exp19_cross_corpus_zeroshot_20260613_seed42"
PY = r"C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe"

BASE_CFG = ROOT / "output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json"
BASE_CKPT = ROOT / "output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt"
LCA_CFG = ROOT / "output/experiments/exp6_librispeech_test_subset_20260606/configs/full_lca_clean_eval_config.json"
LCA_CKPT = ROOT / "output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt"
SAMPLE_LIST = RUN / "artifacts/vctk_300.txt"
EVAL_RUN_DIR = RUN / "eval_vctk_300"
LOG = RUN / "logs/eval.log"


def main():
    for p in (BASE_CFG, BASE_CKPT, LCA_CFG, LCA_CKPT, SAMPLE_LIST):
        if not p.exists():
            print(f"MISSING input: {p}")
            sys.exit(1)
    EVAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, "scripts/evaluate_clean_large_nosave.py",
        "--base-config", str(BASE_CFG.relative_to(ROOT)),
        "--base-checkpoint", str(BASE_CKPT.relative_to(ROOT)),
        "--lca-config", str(LCA_CFG.relative_to(ROOT)),
        "--lca-checkpoint", str(LCA_CKPT.relative_to(ROOT)),
        "--sample-list", str(SAMPLE_LIST.relative_to(ROOT)),
        "--run-dir", str(EVAL_RUN_DIR.relative_to(ROOT)),
        "--device", "cuda",
    ]
    print("launching:", " ".join(cmd))
    with open(LOG, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"eval exit code {rc}; log -> {LOG}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
