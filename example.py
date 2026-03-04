# inference.py
"""
脚本名称: inference.py
功能描述:
    该脚本用于加载预训练或微调后的 SpeechTokenizer 模型，对指定的音频文件进行完整的
    "编码 -> 量化 -> 解码" (Reconstruction) 流程测试。
    
    主要功能包括:
    1. 加载模型: 通过 config 和 checkpoint 路径加载 SpeechTokenizer。
    2. 音频预处理: 自动处理重采样(Resample)和单声道转换(Mono)，转为 Tensor。
    3. 离散编码 (Encode): 提取音频的离散 Code (RVQ Indices)。
       - RVQ_1: 第一层量化器，主要包含语义信息 (Content)。
       - RVQ_supplement: 后续量化器，主要包含音色/细节信息 (Paralinguistic)。
    4. 音频重构 (Decode): 将离散 Code 还原为波形文件，用于评估模型的重构质量。

使用示例:
    python inference.py \
        --config_path config.json \
        --ckpt_path checkpoint.pt \
        --speech_file input.wav \
        --output_file recon_output.wav

依赖库:
    - torchaudio, torch, scipy, numpy
    - speechtokenizer (核心库)
"""

import argparse
import torchaudio
import torch
from speechtokenizer import SpeechTokenizer
from scipy.io.wavfile import write
import numpy as np

from huggingface_hub import snapshot_download

# snapshot_download(repo_id="fnlp/SpeechTokenizer", local_dir="model_hub")


# 设置参数解析器
parser = argparse.ArgumentParser(
    description="加载 SpeechTokenizer 模型并处理音频文件。"
)
parser.add_argument(
    "--config_path",
    type=str,
    help="模型配置文件的路径。",
    default="Log/spt_base/config.json",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    help="模型检查点 (checkpoint) 文件的路径。",
    default=" Log/spt_base/SpeechTokenizer_best_dev.pt",
)
parser.add_argument(
    "--speech_file",
    type=str,
    required=True,
    help="待处理的语音文件路径。",
)
parser.add_argument(
    "--output_file",
    type=str,
    help="输出音频文件的保存路径。",
    default="example_output.wav",
)

args = parser.parse_args()

# 从指定的检查点加载模型
model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
model.eval()

# 获取模型预期的采样率
model_sample_rate = model.sample_rate

# 使用模型的采样率加载并预处理语音波形
# 这一步会将以文件形式存储的音频（如 flac/wav）转化为 Tensor 张量
wav, sr = torchaudio.load(args.speech_file)

if sr != model_sample_rate:
    resample_transform = torchaudio.transforms.Resample(
        orig_freq=sr, new_freq=model_sample_rate
    )
    wav = resample_transform(wav)

# 确保波形是单声道的
if wav.shape[0] > 1:
    wav = wav[:1, :]

# 增加 Batch 维度 (C, T) -> (B, C, T)
wav = wav.unsqueeze(0)


# 从 SpeechTokenizer 中提取离散编码 (Discrete codes)
with torch.no_grad():
    codes = model.encode(wav)  # codes 维度: (n_q, B, T)

RVQ_1 = codes[:1, :, :]  # 包含内容/语义信息，可视作语义 Token (Semantic tokens)
RVQ_supplement = codes[
    1:, :, :
]  # 包含音色等细节信息，补全了第一层量化器丢失的信息

# 拼接语义 Token (RVQ_1) 和补充的音色 Token，然后进行解码
wav_out = model.decode(torch.cat([RVQ_1, RVQ_supplement], axis=0))

# 解码来自第 i 到第 j 个量化器的 token
# 示例：原本的代码逻辑是将它们拼回去进行完整的重构
wav_out = wav_out.detach().numpy()

# 注意：为了防止写入报错，通常需要压缩维度，将 (1, 1, T) 或 (1, T) 变为 (T,)
wav_out = wav_out.squeeze()

# 保存音频文件
write(args.output_file, model_sample_rate, wav_out.astype(np.float32))