#!/usr/bin/env bash
# nas/run_nas.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAS_DIR="${SCRIPT_DIR}"

# =================配置区域=================
CONFIG="${PROJECT_ROOT}/config/spt_base_cfg.json"
NAS_CONFIG="${NAS_DIR}/best_seanet_config.json"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# =========================================

if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    echo "⚠️ 未检测到 Accelerate 配置，正在生成默认配置..."
    accelerate config --default
fi

if [ ! -f "${CONFIG}" ]; then
    echo "❌ 找不到训练配置文件: ${CONFIG}"
    exit 1
fi

if [ ! -f "${NAS_CONFIG}" ]; then
    echo "❌ 找不到 NAS 配置文件: ${NAS_CONFIG}"
    echo "请先运行: python -m nas.export_best_model"
    exit 1
fi

echo "========================================================"
echo "🚀 Launching NAS Training (Retrain Phase)..."
echo "   Project Root : ${PROJECT_ROOT}"
echo "   GPUs Used    : ${CUDA_VISIBLE_DEVICES}"
echo "   Train Config : ${CONFIG}"
echo "   NAS Structure: ${NAS_CONFIG}"
echo "========================================================"

# 默认从头训练；断点续训时手动追加 --continue_train
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
accelerate launch -m nas.train_nas \
    --config "${CONFIG}" \
    --nas_config "${NAS_CONFIG}"

# 断点续训示例:
# PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
# accelerate launch -m nas.train_nas \
#     --config "${CONFIG}" \
#     --nas_config "${NAS_CONFIG}" \
#     --continue_train
