# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# 源自 Meta 的 EnCodec 项目（SEANet 结构）
# 这是 SpeechTokenizer 编解码器的主干，用于从波形提取特征与重建。

"""Encodec SEANet-based encoder and decoder implementation."""

import typing as tp
import numpy as np
import torch.nn as nn
import torch

# 导入自定义模块（来自 ./modules/ 目录）
from . import (
    SConv1d,             # 一维卷积（带归一化）
    SConvTranspose1d,    # 一维转置卷积（上采样）
    SLSTM                # LSTM 层封装（支持双向）
)

# ==========================
# 1️⃣ Snake 激活函数定义
# ==========================

@torch.jit.script
def snake(x, alpha):
    """
    Snake 激活函数：
    结合正弦波特性，能更好地拟合周期性信号（如语音波形）。
    公式: f(x) = x + (1/α) * sin^2(αx)
    """
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """一维 Snake 激活模块，带可学习参数 α"""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))  # 每通道一个可学习参数

    def forward(self, x):
        return snake(x, self.alpha)

# ==========================
# 2️⃣ SEANet 残差块定义
# ==========================

class SEANetResnetBlock(nn.Module):
    """
    SEANet 模型中的残差块。
    每个块包含多层 1D 卷积 + 激活，可选 true skip 跳连。
    """
    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 1], dilations: tp.List[int] = [1, 1],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect', compress: int = 2, true_skip: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), '卷积核数必须与膨胀率数一致'
        act = getattr(nn, activation) if activation != 'Snake' else Snake1d
        hidden = dim // compress   # 压缩通道，用于残差瓶颈，dim 64 composs 2 hidden  = 32
        block = []
        # 构建卷积层序列
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):#(kernel_size, dilation)第一轮等于[3,1].第二轮等于[1,1],i从0取到1
            in_chs = dim if i == 0 else hidden# 第一轮 i = 0, in_chs = dim = 64
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden # 第一轮 i = 0 != len(kernel_sizes) - 1 = 1,out_chs = hidden = 32
            block += [
                act(**activation_params) if activation != 'Snake' else act(in_chs),
                SConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)

        # 决定是否用 identity 跳连
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                    causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        """残差连接：y = shortcut(x) + F(x)"""
        return self.shortcut(x) + self.block(x)

# ==========================
# 3️⃣ 编码器（Encoder）
# ==========================

class SEANetEncoder(nn.Module):
    """
    SEANet 编码器：将波形编码为隐空间特征。
    下采样路径：多层卷积 + 残差块 + LSTM。
    输出形状: (B, dimension, T')。
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2, lstm: int = 2, bidirectional:bool = False):
        super().__init__()
        # ========== 参数初始化 ==========
        self.channels = channels
        self.dimension = dimension #1024
        self.n_filters = n_filters #64
        self.ratios = list(reversed(ratios))        # 编码器使用反向的下采样顺序[2,4,5,8]
        del ratios #在 Python 中显式删除不再需要的局部变量，减少潜在引用和内存占用
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)      # 总下采样率，2*4*5*8

        act = getattr(nn, activation) if activation != 'Snake' else Snake1d #取激活函数,如果activation是snake,那么就取自定义的这个snaKe激活函数，如果activation不是sanke，就去原本参数里的激活函数
        mult = 1 #通道倍增系数

        # 第一层卷积（输入通道 → n_filters）承担着输入预处理与特征基底提取的作用
        model: tp.List[nn.Module] = [
            SConv1d(channels, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        # 主干结构：多层残差块 + 下采样卷积,它的目标是把时间维度很长的语音波形信号压缩成短时间序列、高语义维度的特征表示。这个过程同时完成了“特征提取”和“时序压缩”两项任务
        for i, ratio in enumerate(self.ratios):#i取0123，ratio取2458
            # 残差块堆叠
            for j in range(n_residual_layers):# j只有一个取值0
                model += [
                    SEANetResnetBlock(mult * n_filters,
                                      kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],#dilation_base = 2
                                      norm=norm, norm_params=norm_params,
                                      activation=activation, activation_params=activation_params,#activation = ELU
                                      causal=causal, pad_mode=pad_mode,#causal = FALSE,pad_mode: 'reflect'
                                      compress=compress, true_skip=true_skip)#compress = 2，true_skip = false
                ]
            # 下采样卷积（stride = ratio）
            model += [
                act(**activation_params) if activation != 'Snake' else act(mult * n_filters),
                SConv1d(mult * n_filters, mult * n_filters * 2, #(1*64,1*64*2,kernel_size = 2*2,stride  = 2, )
                        kernel_size=ratio * 2, stride=ratio,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode),
            ]
            mult *= 2   # 通道倍增（每次下采样后加倍）

        # LSTM 层（捕捉长时依赖）
        if lstm:# 此时 nult = 16，n_fileters  = 64,  mult * n_filters = 1024,num_layers = 2 bidirectionaln = true
            model += [SLSTM(mult * n_filters, num_layers=lstm, bidirectional=bidirectional)]

        # 若使用双向 LSTM，则输出通道翻倍
        mult = mult * 2 if bidirectional else mult

        # 最终卷积层（映射到目标维度 dimension）
        model += [
            act(**activation_params) if activation != 'Snake' else act(mult * n_filters),
            SConv1d(mult * n_filters, dimension, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """前向传播：返回编码特征 e ∈ ℝ^(B, D, T')"""
        return self.model(x)

# ==========================
# 4️⃣ 解码器（Decoder）
# ==========================

class SEANetDecoder(nn.Module):
    """
    SEANet 解码器：从隐空间特征重建波形。
    上采样路径：转置卷积 + 残差块 + LSTM。
    输入形状: (B, D, T') → 输出: (B, 1, T)
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2, lstm: int = 2,
                 trim_right_ratio: float = 1.0, bidirectional:bool = False):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)     # 对应下采样率的逆过程

        act = getattr(nn, activation) if activation != 'Snake' else Snake1d
        mult = int(2 ** len(self.ratios))          # 初始通道倍数（对称于编码器最后层）

        # 第一层卷积（从隐特征维度映射到高通道特征）
        # dimension = 1024 ,mult = 2**4 = 16 n_filters = 64  mult * n_filters = 16*1024 = 1024
        model: tp.List[nn.Module] = [
            SConv1d(dimension, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        # LSTM 层 解码器中 lstm用的单向的,所以形状不变
        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm, bidirectional=bidirectional)]

        # 主干：上采样 + 残差层
        for i, ratio in enumerate(self.ratios):
            # 上采样层（使用转置卷积）
            model += [
                act(**activation_params) if activation != 'Snake' else act(mult * n_filters),
                SConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
            ]
            # 残差块堆叠
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      activation=activation, activation_params=activation_params,
                                      norm=norm, norm_params=norm_params, causal=causal,
                                      pad_mode=pad_mode, compress=compress, true_skip=true_skip)
                ]
            mult //= 2  # 通道减半（对称于编码器的倍增）

        # 最后输出卷积 + 激活
        model += [
            act(**activation_params) if activation != 'Snake' else act(n_filters),
            # n_filter = 64,channels  = 1,k = 7,stride = 1
            SConv1d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]
        # 可选最终激活函数（如 Tanh 限幅）
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]

        self.model = nn.Sequential(*model)

    def forward(self, z):
        """前向传播：重建波形 y ∈ ℝ^(B, 1, T)"""
        y = self.model(z)
        return y

# ==========================
# 5️⃣ 简单测试函数
# ==========================
def test():
    import torch
    encoder = SEANetEncoder()
    decoder = SEANetDecoder()
    x = torch.randn(1, 1, 24000)   # 输入 1 秒 16kHz 波形
    z = encoder(x)
    print('z ', z.shape)           # 期望输出: [1, 128, 75]（压缩约 320 倍）
    y = decoder(z)
    print('y ', y.shape)
    assert y.shape == x.shape, (x.shape, y.shape)  # 解码器应还原至原始形状

if __name__ == '__main__':
    test()
