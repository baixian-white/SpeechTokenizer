# custom_model.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
模块名称: custom_model.py
功能描述: 
    该文件定义了支持 NAS (神经网络架构搜索) 的 SEANet 模型结构。
    它包含三个主要类:
    1. SEANetEncoder: 负责将音频波形下采样并编码为特征向量。
    2. SEANetDecoder: 负责将特征向量上采样并重构回音频波形。
    3. SEANet: 包装类，自动处理 Encoder 和 Decoder 的对称/镜像关系。

核心逻辑:
    - 动态构建: 不再使用固定的卷积层，而是根据传入的 `layer_ops_list` 和 `layer_se_list`
      动态选择每一层的算子 (如 Kernel Size, Dilation, 是否 Skip 等)。
    - 镜像对称: 在 SEANet 初始化时，会自动将 Encoder 的配置列表反转 (Reverse)，
      传给 Decoder，从而保证 Autoencoder 结构的对称性。
"""

import typing as tp
import numpy as np
import torch.nn as nn
import torch

# 🌟 统一从 model_components 导入所有组件
# 这样就保证了来源一致性，不会有 ImportError
# 请确保 modules.py (或 model_components.py) 在同一目录下
try:
    from .model_components import (
        SConv1d,
        SConvTranspose1d,
        SLSTM,
        Snake1d,
        SearchableResBlock,
        get_nas_ops,
    )
except ImportError:
    from model_components import (
        SConv1d,
        SConvTranspose1d,
        SLSTM,
        Snake1d,
        SearchableResBlock,
        get_nas_ops,
    )

# ==========================
# 1️⃣ 编码器（Encoder）
# --------------------------
# 结构: [ConvIn] -> [Block_1 + Downsample_1] -> ... -> [Block_N + Downsample_N] -> [LSTM] -> [ConvOut]
# ==========================
class SEANetEncoder(nn.Module):
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2, lstm: int = 2, bidirectional:bool = False,
                 # === [NAS 接口] ===
                 # 这两个列表决定了每一个残差块的具体结构
                 layer_ops_list: tp.Optional[tp.List[str]] = None, 
                 layer_se_list: tp.Optional[tp.List[bool]] = None):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        # ratios 是下采样率列表 (如 [8, 5, 4, 2])
        # Encoder 通常是逐层缩减，这里 reversed 是为了代码习惯 (从外到内)
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        # 准备激活函数工厂 (支持 Snake 或 ELU)
        act_cls = getattr(nn, activation) if activation != 'Snake' else Snake1d
        def get_act(ch):
            return act_cls(**activation_params) if activation != 'Snake' else act_cls(ch)

        # === NAS 配置初始化 ===
        # 如果没有传入 NAS 配置，默认回退到标准结构 (std_k3, 无SE)
        total_blocks = len(self.ratios) * n_residual_layers
        if layer_ops_list is None:
            layer_ops_list = ['std_k3'] * total_blocks
        if layer_se_list is None:
            layer_se_list = [False] * total_blocks
            
        # 获取上下文感知的 OPS 字典 (来自 model_components)
        self.ops_dict = get_nas_ops(norm, pad_mode, causal)

        mult = 1
        # 1. 初始卷积层 (Input Convolution)
        model: tp.List[nn.Module] = [
            SConv1d(channels, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        global_block_idx = 0

        # 2. 循环构建层级结构 (Block + Downsample)
        for i, ratio in enumerate(self.ratios):
            # === 2.1 NAS 搜索层 (Residual Blocks) ===
            # 这些层在 NAS 搜索中会被替换为 best_config.json 指定的算子
            for j in range(n_residual_layers):
                op_name = layer_ops_list[global_block_idx]
                use_se = layer_se_list[global_block_idx]
                global_block_idx += 1
                
                # 计算空洞率 (Dilation)，逐层指数增长以扩大感受野
                current_dilation = dilation_base ** j
                
                model += [
                    SearchableResBlock(
                        channels=mult * n_filters,
                        dilation=current_dilation,
                        op_name=op_name,     # NAS 决定的算子 (如 'sep_k7')
                        use_se=use_se,       # NAS 决定的 SE 开关
                        ops_dict=self.ops_dict,
                        activation_layer=get_act(mult * n_filters)
                    )
                ]

            # === 2.2 下采样层 (Downsampling Layer) ===
            # 这一层是固定的，负责减小时间维度，增加通道数
            model += [
                get_act(mult * n_filters),
                SConv1d(mult * n_filters, mult * n_filters * 2,
                        kernel_size=ratio * 2, stride=ratio,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode),
            ]
            mult *= 2

        # 3. 瓶颈层 LSTM (用于增加时序上下文信息)
        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm, bidirectional=bidirectional)]

        mult = mult * 2 if bidirectional else mult

        # 4. 最终投影层 (Project to dimension)
        model += [
            get_act(mult * n_filters),
            SConv1d(mult * n_filters, dimension, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ==========================
# 2️⃣ 解码器（Decoder）
# --------------------------
# 结构: [ConvIn] -> [LSTM] -> [Upsample_N + Block_N] -> ... -> [Upsample_1 + Block_1] -> [ConvOut]
# ==========================
class SEANetDecoder(nn.Module):
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2, lstm: int = 2,
                 trim_right_ratio: float = 1.0, bidirectional:bool = False,
                 # === [NAS 接口] ===
                 layer_ops_list: tp.Optional[tp.List[str]] = None, 
                 layer_se_list: tp.Optional[tp.List[bool]] = None):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        
        act_cls = getattr(nn, activation) if activation != 'Snake' else Snake1d
        def get_act(ch):
            return act_cls(**activation_params) if activation != 'Snake' else act_cls(ch)

        # === NAS 配置初始化 ===
        total_blocks = len(self.ratios) * n_residual_layers
        if layer_ops_list is None:
            layer_ops_list = ['std_k3'] * total_blocks
        if layer_se_list is None:
            layer_se_list = [False] * total_blocks
            
        self.ops_dict = get_nas_ops(norm, pad_mode, causal)

        mult = int(2 ** len(self.ratios))
        
        # 1. 初始投影层 (Input Projection)
        model: tp.List[nn.Module] = [
            SConv1d(dimension, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        # 2. 瓶颈层 LSTM
        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm, bidirectional=bidirectional)]

        global_block_idx = 0

        # 3. 循环构建层级结构 (Upsample + Block)
        for i, ratio in enumerate(self.ratios):
            # === 3.1 上采样层 (Upsampling / Transpose Conv) ===
            model += [
                get_act(mult * n_filters),
                SConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
            ]
            
            # === 3.2 NAS 搜索层 (Residual Blocks) ===
            # 注意：这里的 block 是接在上采样之后的
            for j in range(n_residual_layers):
                op_name = layer_ops_list[global_block_idx]
                use_se = layer_se_list[global_block_idx]
                global_block_idx += 1
                
                current_dilation = dilation_base ** j

                model += [
                    SearchableResBlock(
                        channels=mult * n_filters // 2,
                        dilation=current_dilation,
                        op_name=op_name,
                        use_se=use_se,
                        ops_dict=self.ops_dict,
                        activation_layer=get_act(mult * n_filters // 2)
                    )
                ]
            
            mult //= 2

        # 4. 最终输出卷积 (Output Conv)
        model += [
            get_act(n_filters),
            SConv1d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        # 5. 最终激活 (如 Tanh, 用于归一化到 [-1, 1])
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]

        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y

# ==========================
# 3️⃣ 包装类 SEANet (处理镜像逻辑)
# --------------------------
# 这个类是外部调用的入口。
# 关键逻辑：它接收 NAS 搜索出的 `layer_ops_list`，
# 然后把这个列表倒序 (Reverse) 传给 Decoder，从而保证 Encoder/Decoder 结构对称。
# ==========================
class SEANet(nn.Module):
    def __init__(self, 
                 # 这里列出所有常用参数，或者用 **kwargs 透传
                 channels=1, dimension=128, n_filters=32, ratios=[8, 5, 4, 2],
                 activation='ELU', norm='weight_norm', causal=False, 
                 # NAS 列表
                 layer_ops_list=None, layer_se_list=None,
                 **kwargs):
        super().__init__()
        
        # 1. Encoder (正序传入 NAS 配置)
        self.encoder = SEANetEncoder(
            channels=channels, dimension=dimension, n_filters=n_filters, ratios=ratios,
            activation=activation, norm=norm, causal=causal,
            layer_ops_list=layer_ops_list,
            layer_se_list=layer_se_list,
            **kwargs
        )
        
        # 2. Decoder (准备镜像配置)
        dec_ops = None
        dec_se = None
        if layer_ops_list is not None:
            # ⭐ 核心：镜像翻转
            # 如果 Encoder 用了 [A, B, C]，Decoder 应该用 [C, B, A] 才能对称
            dec_ops = layer_ops_list[::-1]
            dec_se = layer_se_list[::-1]
            
        # 3. Decoder (倒序传入)
        self.decoder = SEANetDecoder(
            channels=channels, dimension=dimension, n_filters=n_filters, ratios=ratios,
            activation=activation, norm=norm, causal=causal,
            layer_ops_list=dec_ops,
            layer_se_list=dec_se,
            **kwargs
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z # 返回重建音频和特征
