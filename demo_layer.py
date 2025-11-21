#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加载 SpeechTokenizer 模型

对任意音频进行 encode → select RVQ layers → decode

可指定任意 RVQ 层组合（如 0,1,2 或 1-3 或 all）

可选择使用 GPU / CPU

可选自动从 HuggingFace 下载模型

提供完整耗时统计（加载、重采样、编码、解码、写盘、总耗时）
"""

import argparse
import time
import torch
import torchaudio
import numpy as np
from typing import List
from scipy.io.wavfile import write

try:
    # 可选用；仅在 --download 时调用
    from huggingface_hub import snapshot_download
    _HAVE_HF = True
except Exception:
    _HAVE_HF = False

from speechtokenizer import SpeechTokenizer


# ==============================
# 参数解析
# ==============================
parser = argparse.ArgumentParser(description="SpeechTokenizer 推理（可指定 RVQ 层，GPU 加速 & 计时）")
parser.add_argument("--config_path", type=str,
                    default="model_hub/speechtokenizer_hubert_avg/config.json",
                    help="模型配置文件路径")
parser.add_argument("--ckpt_path", type=str,
                    default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt",
                    help="模型权重文件路径")
parser.add_argument("--speech_file", type=str, required=True, help="输入语音文件路径（建议单声道 WAV）")
parser.add_argument("--output_file", type=str, default="example_output.wav", help="输出音频路径")
parser.add_argument("--rvq_layers", type=str, default="all",
                    help="指定 RVQ 层，可选 'all','0,1,2','1-3','1:3' 等")
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                    help="推理设备：auto（默认，优先cuda），或显式指定 cuda / cpu")
parser.add_argument("--download", action="store_true",
                    help="需要时从 HuggingFace 下载/更新到 model_hub（默认不下载，避免启动开销）")
args = parser.parse_args()


# ==============================
# 工具函数：解析 RVQ 层选择
# ==============================
def parse_rvq_layers_spec(spec: str, n_layers: int) -> List[int]:
    """
    把 'all' / '0,2,3' / '1-3' / '1:3' 解析成有效层索引列表
    """
    spec = spec.strip().lower()
    if spec in ["all", "*"]:
        return list(range(n_layers))
    indices = set()
    for p in spec.split(","):
        p = p.strip()
        if "-" in p or ":" in p:
            sep = "-" if "-" in p else ":"
            a, b = map(int, p.split(sep))
            lo, hi = (a, b) if a <= b else (b, a)
            for i in range(lo, hi + 1):
                indices.add(i)
        elif p:
            indices.add(int(p))
    valid = sorted([i for i in indices if 0 <= i < n_layers])
    if not valid:
        raise ValueError(f"没有有效 RVQ 层（输入: {spec}，模型层数: {n_layers}）")
    return valid


# ==============================
# 可选：下载模型
# ==============================
if args.download:
    if not _HAVE_HF:
        print("[WARN] 未安装 huggingface_hub，跳过下载。如需下载，请先安装：pip install huggingface_hub")
    else:
        t_dl0 = time.time()
        print("[INFO] 开始从 HuggingFace 检查/下载到 model_hub（仅本次应需触发）...")
        # 只在你主动传 --download 时检查/下载，避免每次启动做 I/O 与哈希
        try:
            snapshot_download(repo_id="fnlp/SpeechTokenizer", local_dir="model_hub")
        except Exception as e:
            print(f"[WARN] 下载失败：{e}")
        print(f"[TIME] 下载检查耗时: {(time.time() - t_dl0)*1000:.1f} ms")


# ==============================
# 设备选择
# ==============================
if args.device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = args.device
if device == "cuda" and not torch.cuda.is_available():
    print("[WARN] 你指定了 cuda，但当前环境不可用，自动回落到 cpu")
    device = "cpu"
print(f"[INFO] 使用设备: {device}")

# ==============================
# 加载模型
# ==============================
t0_all = time.time()
t0 = time.time()
try:
    model = SpeechTokenizer.load_from_checkpoint( ckpt_path=args.ckpt_path,
                                                 config_path=args.config_path)
except TypeError:
    # 部分版本 API 只接收 checkpoint
    model = SpeechTokenizer.load_from_checkpoint(args.ckpt_path)

# 推理状态 + 迁移设备
model.eval()
model.to(device)
t1 = time.time()
print(f"[TIME] 加载模型耗时: {(t1 - t0)*1000:.1f} ms")

# 获取模型采样率
model_sample_rate = getattr(model, "sample_rate", 16000)
print(f"[INFO] 模型采样率: {model_sample_rate}")

# ==============================
# 加载与重采样音频
# ==============================
t0 = time.time()
wav, sr = torchaudio.load(args.speech_file)  # [C, T], torch.float32 on CPU
# 转单声道
if wav.shape[0] > 1:
    wav = wav[:1, :]
# 如需重采样，先在 CPU 上完成（torchaudio transforms 多为 CPU 实现）
if sr != model_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sample_rate)
    wav = resampler(wav)
# 扩维到 [B=1, C=1, T]，并迁移到推理设备
wav = wav.unsqueeze(0).to(device)  # [1, 1, T]
t1 = time.time()
print(f"[TIME] 加载+重采样耗时: {(t1 - t0)*1000:.1f} ms")

# ==============================
# 编码（encode）
# ==============================
t0 = time.time()
with torch.inference_mode():
    codes = model.encode(wav)  # 可能是 (n_q, B, T) 或 (B, n_q, T)
t1 = time.time()
print(f"[TIME] 编码（encode）耗时: {(t1 - t0)*1000:.1f} ms")

# ==============================
# 解析层数与选层
# ==============================
t0 = time.time()
if not isinstance(codes, torch.Tensor):
    raise RuntimeError("model.encode 返回类型不是 torch.Tensor")

# 推断返回形状并统一为 (B, n_q, T)
if codes.ndim == 3 and codes.shape[0] <= 16:  # 经验判断 n_q 不会太大
    # 形状为 (n_q, B, T) -> 转为 (B, n_q, T)
    codes = codes.permute(1, 0, 2)
n_q = codes.shape[1]
print(f"[INFO] 模型 RVQ 层数: {n_q}")

selected_layers = parse_rvq_layers_spec(args.rvq_layers, n_q)
print(f"[INFO] 选中的 RVQ 层索引: {selected_layers}")

# 在同一设备上完成切片，避免不必要的数据迁移
codes_selected = codes[:, selected_layers, :]  # (B, L_sel, T)
t1 = time.time()
print(f"[TIME] 选层（permute/slice）耗时: {(t1 - t0)*1000:.1f} ms")

# ==============================
# 解码（decode）
# ==============================
t0 = time.time()
with torch.inference_mode():
    wav_out = model.decode(codes_selected)  # 约定支持 (B, L_sel, T)
t1 = time.time()
print(f"[TIME] 解码（decode）耗时: {(t1 - t0)*1000:.1f} ms")

# ==============================
# 写盘
# ==============================
t0 = time.time()
# 转 numpy：batch 取 0，裁剪到 [-1,1] 再以 float32 写盘
if isinstance(wav_out, torch.Tensor):
    wav_out = wav_out.detach().to("cpu")
wav_out_np = np.squeeze(np.array(wav_out)[0])
wav_out_np = np.clip(wav_out_np, -1.0, 1.0).astype(np.float32)

write(args.output_file, model_sample_rate, wav_out_np)
t1 = time.time()
print(f"[TIME] 写盘耗时: {(t1 - t0)*1000:.1f} ms")

# ==============================
# 总时间
# ==============================
t_all = time.time() - t0_all
print(f"[TIME] 总耗时: {t_all*1000:.1f} ms")
print(f"[DONE] 已保存输出音频到: {args.output_file}")
