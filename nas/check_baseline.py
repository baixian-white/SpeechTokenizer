"""
脚本用途：评估 SpeechTokenizer (SEANet 架构) 的计算复杂度 (FLOPs)。
1. 实例化标准参数的 Encoder 和 Decoder。
2. 模拟 1 秒 (16,000 samples) 的音频输入。
3. 统计推理过程中的总浮点运算量 (GFLOPs)。
4. 输出 Baseline 基准值，并为后续轻量化改进设定 50%~60% 的压缩目标红线。
"""
from pathlib import Path
import sys
import torch
from thop import profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speechtokenizer.modules import SEANetEncoder, SEANetDecoder

# 1. 实例化原版配置 (Standard Config)
# 这是 SpeechTokenizer / EnCodec 论文中的标准参数
encoder = SEANetEncoder(
    channels=1, dimension=1024, n_filters=32, n_residual_layers=1,
    ratios=[8, 5, 4, 2], kernel_size=7, lstm=2
).cuda()

decoder = SEANetDecoder(
    channels=1, dimension=1024, n_filters=32, n_residual_layers=1,
    ratios=[8, 5, 4, 2], kernel_size=7, lstm=2
).cuda()

# 2. 伪造 1秒 音频输入
dummy_input = torch.randn(1, 1, 16000).cuda()

# 3. 计算 FLOPs
print("正在计算原版 FLOPs...")
enc_flops, _ = profile(encoder, inputs=(dummy_input,), verbose=False)

# Decoder 输入需要先跑一次 Encoder
with torch.no_grad():
    z = encoder(dummy_input)
dec_flops, _ = profile(decoder, inputs=(z,), verbose=False)

total_flops = enc_flops + dec_flops

print("="*40)
print(f"原版 Encoder: {enc_flops/1e9:.4f} G")
print(f"原版 Decoder: {dec_flops/1e9:.4f} G")
print(f"原版总计 (Baseline): {total_flops/1e9:.4f} G")
print("="*40)

# 4. 推荐红线
print(f"建议红线 (50% 压缩): {(total_flops * 0.5)/1e9:.4f} G")
print(f"建议红线 (60% 压缩): {(total_flops * 0.4)/1e9:.4f} G")
