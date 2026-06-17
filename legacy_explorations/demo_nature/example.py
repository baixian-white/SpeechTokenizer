#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpeechTokenizer 推理示例（带详细注释 & 可选多层保存）
---------------------------------------------------
该脚本演示如何加载 SpeechTokenizer 预训练模型，
对单个语音文件进行语义+音色编码、解码，并保存重建音频。

在保留原始示例功能/注释的基础上，新增：
- 逐层导出离散 tokens（.npy）
- 可选指定多层组合导出与解码 (--layers / --cumulative / --save_wavs)
- ✅ 额外保存 PyTorch 张量格式（.pt）输入、tokens、模型权重，便于复现

功能说明：
1. 自动从 HuggingFace 下载官方预训练模型权重到本地 (model_hub/)
2. 支持命令行参数自定义 config / ckpt / 输入语音 / 输出文件路径
3. 使用 SpeechTokenizer.encode() 将语音转换为离散 token（形状：[n_q, B, Tq]）
4. 使用 SpeechTokenizer.decode() 重建语音波形
5. 逐层/按组合导出 tokens，并可选对组合直接解码保存
"""

import numpy as np
from pathlib import Path
import argparse
import os

import torchaudio  # 音频读写（我们强制使用 soundfile 后端，避免 torchcodec 依赖）
import torch
# ⚠ 原示例 from speechtokenizer import SpeechTokenizer 会连带导入 trainer 依赖（tensorboard、beartype）
#   为了只做推理更干净，改为只引入模型本体：
from speechtokenizer.model import SpeechTokenizer
from scipy.io.wavfile import write  # 与原示例保持一致，用 scipy 写 wav
from huggingface_hub import snapshot_download  # 从 HF Hub 下载模型

# 为避免 torchaudio 2.4+ 触发 torchcodec 依赖，这里强制选用 soundfile 后端（需系统已装 libsndfile1）
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass


# ------------------------------
# 工具：解析 --layers 的写法（如 0-2,5 -> [0,1,2,5]）
# ------------------------------
def parse_layers(spec: str, n_q: int):
    """
    将 --layers 的字符串解析为有序不重复的层索引列表。
    支持：'0', '0,1,2', '0-3', '0-1,3,5-7' 等。
    越界索引会被剔除，最终排序返回。
    """
    if not spec:
        return []
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a, b = int(a), int(b)
            except ValueError:
                continue
            if a > b:
                a, b = b, a
            for x in range(a, b + 1):
                if 0 <= x < n_q:
                    out.add(x)
        else:
            try:
                x = int(part)
            except ValueError:
                continue
            if 0 <= x < n_q:
                out.add(x)
    return sorted(out)


def layers_tag(layers: list[int]) -> str:
    """把层列表压缩为 tag，用于文件名（例如 [0,1,2,5] -> '0-2_5'）。"""
    if not layers:
        return "none"
    runs = []
    start = prev = layers[0]
    for x in layers[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    parts = [f"{a}-{b}" if a != b else f"{a}" for (a, b) in runs]
    return "_".join(parts)


# =============================================
# 1. 从 HuggingFace 下载模型权重到本地 model_hub/
# =============================================
# 官方模型仓库为 "fnlp/SpeechTokenizer"
# snapshot_download 会自动从 HF Hub 拉取 config.json 与 SpeechTokenizer.pt（若本地不存在）
snapshot_download(repo_id="fnlp/SpeechTokenizer", local_dir="model_hub")

# =============================================
# 2. 命令行参数解析
# =============================================
parser = argparse.ArgumentParser( description="加载 SpeechTokenizer 并处理语音文件。")

parser.add_argument(
    "--config_path",
    type=str,
    help="模型配置文件路径 (config.json)",
    default="model_hub/speechtokenizer_hubert_avg/config.json",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    help="模型权重文件路径 (SpeechTokenizer.pt)",
    default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt",
)
parser.add_argument(
    "--speech_file",
    type=str,
    required=True,
    help="输入语音文件路径 (建议 wav；mp3/flac 也可，但需系统有 ffmpeg)",
)
parser.add_argument(
    "--output_file",
    type=str,
    help="输出重建音频保存路径（完整重建：语义+音色）",
    default="example_output.wav",
)
# 下面是新增参数（与原功能兼容，不影响原来的基本用法）
parser.add_argument(
    "--export_dir",
    type=str,
    default="tokens_export",
    help="导出 tokens 的目录（会保存 all_codes.npy 和每层 codes_layer{i}.npy）",
)
parser.add_argument(
    "--layers",
    type=str,
    default="",
    help="可选：指定想要额外导出的层集合（以及对该集合解码）例如：'0-2,5'",
)
parser.add_argument(
    "--cumulative",
    action="store_true",
    help="与 --layers 配合：将每个 k 视为 {0..k} 累积集合（便于逐层听效果）",
)
parser.add_argument(
    "--save_wavs",
    action="store_true",
    help="对 --layers / --cumulative 选出的层集合进行解码并额外保存 wav（与 output_file 并存）",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    choices=["cuda", "cpu"],
    help="运行设备（默认 cuda；若无 GPU 则自动回退到 cpu）",
)
parser.add_argument(
    "--max_len_sec",
    type=float,
    default=0.0,
    help=">0 时按此秒数对输入分块编码，降低显存峰值（长音频更稳）",
)

args = parser.parse_args()

# =============================================
# 3. 加载模型
# =============================================
# 从指定路径加载配置与权重（只引入模型，不引 trainer，避免额外依赖）
model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
model.eval()  # 进入推理模式（关闭 Dropout、BN 更新）

# 自动选择设备：若用户选 cuda 且可用，则用 GPU，否则用 CPU
use_cuda = (args.device == "cuda") and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
model.to(device)

# 获取模型的采样率（通常为 16 kHz）
model_sample_rate = model.sample_rate

# =============================================
# 4. 加载输入语音文件
# =============================================
# 加载波形 (shape: [channels, samples])
wav, sr = torchaudio.load(args.speech_file)

# 若采样率与模型不一致，则重采样
if sr != model_sample_rate:
    resample_transform = torchaudio.transforms.Resample(
        orig_freq=sr, new_freq=model_sample_rate
    )
    wav = resample_transform(wav)

# 转单声道（只保留第 1 个通道）
if wav.dim() == 1:
    wav = wav.unsqueeze(0)  # [C, T] 保证有通道维
if wav.shape[0] > 1:
    wav = wav[:1, :]

# 添加 Batch 维度 -> shape: [1, 1, T]，并移动到目标设备
wav = wav.unsqueeze(0).to(device)

# =============================================
#  保存模型输入为 .npz 与 .pt
# =============================================
export_dir = Path(args.export_dir)
export_dir.mkdir(parents=True, exist_ok=True)  # ✅ 自动创建目录

input_npz_path = export_dir / "input_waveform.npz"
np.savez_compressed(
    input_npz_path,
    waveform=wav.detach().cpu().numpy(),
    sample_rate=model_sample_rate,
)
print(f"[SAVE] 输入波形已保存：{input_npz_path}  shape={wav.shape}")

# 新增保存 .pt 格式
input_pt_path = export_dir / "input_waveform.pt"
torch.save({"waveform": wav.detach().cpu(), "sample_rate": model_sample_rate}, input_pt_path)
print(f"[SAVE] 输入张量已保存：{input_pt_path} (PyTorch Tensor)")

# =============================================
# 5. 编码 (encode)
# =============================================
def encode_chunk(x: torch.Tensor) -> torch.Tensor:
    """单段音频编码，支持 CUDA 混合精度以进一步省显存。"""
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
        return model.encode(x)  # (n_q, B, Tq)

# 可选的分块编码逻辑（长音频显存更稳）
if args.max_len_sec and args.max_len_sec > 0:
    hop = int(model_sample_rate * args.max_len_sec)
    T = wav.shape[-1]
    parts = []
    s = 0
    while s < T:
        e = min(s + hop, T)
        parts.append(encode_chunk(wav[..., s:e]))
        s = e
    n_q = parts[0].shape[0]
    cat_list = []
    for i in range(n_q):
        cat_list.append(torch.cat([p[i] for p in parts], dim=-1))
    codes = torch.stack(cat_list, dim=0)
else:
    codes = encode_chunk(wav)

n_q, B, Tq = codes.shape
print(f"[INFO] codes shape = (n_q={n_q}, B={B}, Tq={Tq})")

RVQ_1 = codes[:1, :, :]
RVQ_supplement = codes[1:, :, :]

# =============================================
# 6. 解码 (decode)
# =============================================
with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
    wav_out_full = model.decode(torch.cat([RVQ_1, RVQ_supplement], dim=0))

# =============================================
# 7. 保存输出音频（完整重建）
# =============================================
wav_np = wav_out_full.detach().cpu().numpy()
write(args.output_file, model_sample_rate, wav_np.astype(np.float32))
print(f"[DONE] 输出文件已保存：{args.output_file} (采样率={model_sample_rate} Hz)")

# =============================================
# 8. 逐层导出 tokens（以及 all_codes）
# =============================================
all_codes_np = codes.detach().cpu().numpy()
np.save(export_dir / "all_codes.npy", all_codes_np)
print(f"[SAVE] {export_dir/'all_codes.npy'}  shape={all_codes_np.shape}")

# ✅ 新增保存 all_codes 为 .pt 格式
torch.save(codes.detach().cpu(), export_dir / "all_codes.pt")
print(f"[SAVE] {export_dir/'all_codes.pt'}  (PyTorch Tensor)")

# 逐层保存 .npy + .pt
for i in range(n_q):
    np.save(export_dir / f"codes_layer{i}.npy", all_codes_np[i])
    print(f"[SAVE] {export_dir/f'codes_layer{i}.npy'}  shape={all_codes_np[i].shape}")
    torch.save(codes[i].detach().cpu(), export_dir / f"codes_layer{i}.pt")
    print(f"[SAVE] {export_dir/f'codes_layer{i}.pt'}  (PyTorch Tensor)")

# ✅ 新增：保存模型权重（便于复现）
model_pt_path = export_dir / "model_weights.pt"
torch.save(model.state_dict(), model_pt_path)
print(f"[SAVE] 模型权重已保存：{model_pt_path}")

# =============================================
# 9. 可选：根据 --layers / --cumulative 额外导出与解码
# =============================================
sel_layers = parse_layers(args.layers, n_q)

layer_sets: list[list[int]] = []
if sel_layers:
    if args.cumulative:
        for k in sel_layers:
            layer_sets.append(list(range(0, k + 1)))
    else:
        layer_sets = [sel_layers]

for layers in layer_sets:
    if not layers:
        continue
    idx = torch.tensor(layers, dtype=torch.long, device=codes.device)
    sel = codes.index_select(dim=0, index=idx)
    tag = layers_tag(layers)

    np.save(export_dir / f"codes_layers_{tag}.npy", sel.detach().cpu().numpy())
    print(f"[SAVE] {export_dir/f'codes_layers_{tag}.npy'}  layers={layers} shape={sel.shape}")

    # ✅ 新增保存 .pt 格式组合层
    torch.save(sel.detach().cpu(), export_dir / f"codes_layers_{tag}.pt")
    print(f"[SAVE] {export_dir/f'codes_layers_{tag}.pt'}  layers={layers} (PyTorch Tensor)")

    if args.save_wavs:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
            wav_rec = model.decode(sel, st=min(layers))
        torchaudio.save(
            str(export_dir / f"recon_layers_{tag}.wav"),
            wav_rec.detach().cpu().squeeze(0),
            sample_rate=model_sample_rate,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        print(f"[SAVE] {export_dir/f'recon_layers_{tag}.wav'}  layers={layers}")

np.save(export_dir / "codes_semantic_layer0.npy", RVQ_1.detach().cpu().numpy())
if args.save_wavs:
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
        wav_sem = model.decode(RVQ_1, st=0)
        wav_full = model.decode(codes)
    torchaudio.save(
        str(export_dir / "semantic_only.wav"),
        wav_sem.detach().cpu().squeeze(0),
        sample_rate=model_sample_rate,
        encoding="PCM_S",
        bits_per_sample=16,
    )
    torchaudio.save(
        str(export_dir / "full_recon.wav"),
        wav_full.detach().cpu().squeeze(0),
        sample_rate=model_sample_rate,
        encoding="PCM_S",
        bits_per_sample=16,
    )
    print(f"[SAVE] {export_dir/'semantic_only.wav'}")
    print(f"[SAVE] {export_dir/'full_recon.wav'}")
