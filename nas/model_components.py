# modules.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
模块名称: modules.py
功能描述: 
    该文件定义了构建 SpeechTokenizer 及 NAS 模型所需的所有底层组件。
    它包含三个主要部分:
    1. 基础卷积层封装 (SConv1d, SConvTranspose1d 等): 处理因果填充(Causal Padding)和归一化。
    2. NAS 专用组件: 包括 Snake 激活函数、SE 模块、深度可分离卷积 (DSConv)。
    3. NAS 搜索逻辑: 定义了搜索空间 (OPS 字典) 和可配置的残差块 (SearchableResBlock)。

依赖关系:
    - 被 custom_model.py 引用以构建 NASSpeechTokenizer。
    - 也可以被原本的 SEANet 结构引用。
"""

import math
import typing as tp
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

# ==========================================
# 1️⃣ 基础卷积层定义 (SConv1d, Norm 等)
# ------------------------------------------
# 这部分代码是对 PyTorch 原生卷积层的封装，
# 主要目的是统一处理 "Weight Norm" / "Spectral Norm" 以及 "Causal Padding" (因果填充)。
# ==========================================

CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                 'time_layer_norm', 'layer_norm', 'time_group_norm'])

def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    """对卷积层应用参数化归一化 (如 WeightNorm, SpectralNorm)"""
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
        return module

def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    """获取通过层后的归一化模块 (如 LayerNorm, GroupNorm)"""
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        try:
            from .norm import ConvLayerNorm
            return ConvLayerNorm(module.out_channels, **norm_kwargs)
        except ImportError:
            # Fallback 到原生 LayerNorm
            return nn.LayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()

def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """计算为了让输出形状匹配所需的额外填充量"""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length

def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))

def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """支持 'reflect' 模式的 1D 填充封装"""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)

def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """去除填充，用于 ConvTranspose1d 后还原长度"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]

class NormConv1d(nn.Module):
    """带归一化的 Conv1d 封装"""
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class NormConvTranspose1d(nn.Module):
    """带归一化的 ConvTranspose1d (反卷积) 封装"""
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x

class SConv1d(nn.Module):
    """
    流式卷积层 (Streamable Conv1d)
    自动处理 Padding 以确保输入输出长度一致 (或符合下采样预期)，支持因果推断 (Causal)。
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect'):
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn('SConv1d has been initialized with stride > 1 and dilation > 1')
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                               dilation=dilation, groups=groups, bias=bias, causal=causal,
                               norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        padding_total = (kernel_size - 1) * dilation - (stride - 1)
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        
        # 因果卷积：Padding 全部加在左边 (过去)
        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        # 非因果卷积：Padding 左右均分
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)

class SConvTranspose1d(nn.Module):
    """
    流式反卷积层 (Streamable ConvTranspose1d)
    用于 Decoder 上采样，同样处理了 Padding 问题。
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, causal: bool = False,
                 norm: str = 'none', trim_right_ratio: float = 1.,
                 norm_kwargs: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                          causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride
        y = self.convtr(x)
        
        # 反卷积输出通常比需要长，需要裁剪 (Unpad)
        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y

class SLSTM(nn.Module):
    """简单的 LSTM 封装，调整了 Permute 顺序以适配 Conv1d 的 (B, C, T) 格式"""
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True, bidirectional: bool = False):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers, bidirectional=bidirectional)

    def forward(self, x):
        x = x.permute(2, 0, 1) # (T, B, C)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0) # (B, C, T)
        return y

# ==========================================
# 2️⃣ NAS 专用组件 (Snake, SE, DSConv)
# ------------------------------------------
# 这些是轻量化模型常用的组件：
# Snake: 周期性激活函数，适合音频波形建模。
# SEBlock: 通道注意力机制 (Squeeze-and-Excitation)。
# DSConv: 深度可分离卷积，大幅减少参数量。
# ==========================================

@torch.jit.script
def snake(x, alpha):
    """Snake 激活函数公式: x + sin^2(alpha * x) / alpha"""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class Snake1d(nn.Module):
    """Snake 1D 激活层，alpha 是可学习参数"""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 模块: 学习通道间的权重"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class DSConv1d(nn.Module):
    """深度可分离卷积 (Depthwise Separable Conv)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 bias=True, causal=False, norm='none', norm_kwargs={}, pad_mode='reflect'):
        super().__init__()
        # 1. Depthwise: groups=in_channels，每个通道独立卷积
        self.depthwise = SConv1d(
            in_channels, in_channels, kernel_size, 
            stride=stride, dilation=dilation, groups=in_channels,
            bias=bias, causal=causal, norm=norm, 
            norm_kwargs=norm_kwargs, pad_mode=pad_mode
        )
        # 2. Pointwise: kernel_size=1，混合通道信息
        self.pointwise = SConv1d(
            in_channels, out_channels, kernel_size=1, 
            stride=1, dilation=1, groups=1,
            bias=bias, causal=causal, norm=norm, 
            norm_kwargs=norm_kwargs, pad_mode=pad_mode
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# ==========================================
# 3️⃣ NAS 逻辑 (OPS 字典与 ResBlock)
# ------------------------------------------
# 这里定义了 NAS 的 "搜索空间" 和 "构建块"。
# 搜索空间包含了不同 kernel size, 不同类型的卷积算子。
# ==========================================

def get_nas_ops(norm: str, pad_mode: str, causal: bool):
    """
    生成 NAS 候选算子字典。
    闭包函数，捕获全局配置 (norm, pad_mode, causal)，返回一个构建算子的工厂函数字典。
    
    C: 通道数 (Channels)
    D: 膨胀率 (Dilation)
    """
    return {
        # 标准卷积 K=3
        'std_k3': lambda C, D: SConv1d(C, C, kernel_size=3, dilation=D, stride=1, 
                                       norm=norm, pad_mode=pad_mode, causal=causal),
        # 标准卷积 K=5
        'std_k5': lambda C, D: SConv1d(C, C, kernel_size=5, dilation=D, stride=1, 
                                       norm=norm, pad_mode=pad_mode, causal=causal),
        # 深度可分离卷积 K=7 (轻量化)
        'sep_k7': lambda C, D: DSConv1d(C, C, kernel_size=7, dilation=D, 
                                        norm=norm, pad_mode=pad_mode, causal=causal),
        # 深度可分离卷积 K=9 (轻量化)
        'sep_k9': lambda C, D: DSConv1d(C, C, kernel_size=9, dilation=D, 
                                        norm=norm, pad_mode=pad_mode, causal=causal),
        # 大感受野卷积 K=9, Dilation 翻倍
        'dil_k9': lambda C, D: SConv1d(C, C, kernel_size=9, dilation=D*2, stride=1, 
                                       norm=norm, pad_mode=pad_mode, causal=causal),
        # 跳过连接 (Identity)
        'skip':   lambda C, D: nn.Identity()
    }

class SearchableResBlock(nn.Module):
    """
    NAS 中使用的可搜索残差块。
    根据传入的 `op_name` 字符串，从 ops_dict 中实例化具体的算子。
    结构: Input -> Op -> Norm -> Act -> SE(可选) -> Residual Add
    """
    def __init__(self, channels, dilation, op_name, use_se, 
                 ops_dict, activation_layer):
        super().__init__()
        
        self.is_skip = (op_name == 'skip')
        if self.is_skip:
            self.op = nn.Identity()
            self.se = nn.Identity()
            return

        # 1. 实例化核心算子 (如 std_k3, sep_k7)
        self.op = ops_dict[op_name](channels, dilation)
        
        # 2. 激活函数 (如 Snake1d 或 ELU)
        self.act = activation_layer
        
        # 3. 归一化 (此处固定使用 GroupNorm)
        self.norm = nn.GroupNorm(1, channels) 
        
        # 4. SE 模块 (可选)
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x):
        if self.is_skip:
            return x
            
        residual = x
        out = self.op(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.se(out)
            
        return residual + out