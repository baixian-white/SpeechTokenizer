
#!/usr/bin/env bash
set -euo pipefail

CONFIG="config/spt_base_cfg.json"

# ===== 选一种启动方式 =====
# A) 单卡摸底（推荐先跑稳）
# export CUDA_VISIBLE_DEVICES=0
# python scripts/train_example.py --config "${CONFIG}"

# B) 两卡并行（你的机器是 0、1 两张 4090）
pip install -U accelerate tensorboard >/dev/null 2>&1 || true
# 首次可生成默认 accelerate 配置（无交互版）
[ -f ~/.cache/huggingface/accelerate/default_config.yaml ] || accelerate config --default

export CUDA_VISIBLE_DEVICES=0,1
accelerate launch scripts/train_example.py --config "${CONFIG}"
# 如需断点续训再加： --continue_train
