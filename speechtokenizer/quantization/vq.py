# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Residual vector quantizer implementation.

残差向量量化（Residual Vector Quantization, RVQ）模块封装。
作用：把连续特征 x ∈ R^(B, D, T) 量化为若干层码本的离散索引，
并返回量化后的连续近似向量（可供解码器使用）与承诺损失。
"""

from dataclasses import dataclass, field
import math
import typing as tp

import torch
from torch import nn

from .core_vq import ResidualVectorQuantization
# ↑ 实际的 RVQ 算法核心实现（含码本、EMA 更新、encode/decode 等）在 core_vq 中，
# 本文件主要是更高一层的封装（对外暴露统一接口、做形状与参数管理）。


@dataclass
class QuantizedResult:
    """（可选的数据结构）记录量化结果的容器。

    注意：当前类在本实现里并未直接被 forward 返回（forward 返回的是 tuple），
    但给出该 dataclass 以说明典型量化输出中可能会包含的字段。
    """
    quantized: torch.Tensor   # 量化的连续近似向量（供解码器使用），形状通常为 (B, D, T)
    codes: torch.Tensor       # 每层码本的离散索引，形状 (n_q, B, T)
    bandwidth: torch.Tensor   # 每个样本对应的码率 (kb/s)。不同带宽模式下会用到。
    penalty: tp.Optional[torch.Tensor] = None  # 可选的额外惩罚项（如带宽/稀疏性正则）
    metrics: dict = field(default_factory=dict) # 记录额外统计（如激活码本数、使用分布等）


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer（残差向量量化器）的外层封装。

    参数：
        dimension (int): 码本向量的维度 D（必须与 encoder 输出/decoder 输入的通道维一致）。
        n_q (int): 量化器（残差层）数量。越多层 → 越精细的近似 → 更高码率。
        bins (int): 每层码本大小（codebook size，离散向量的个数）。
        decay (float): EMA（指数滑动平均）用于码本向量更新时的衰减系数。
        kmeans_init (bool): 是否使用 K-Means 对码本做初始化（更稳定地开始训练）。
        kmeans_iters (int): K-Means 初始化的迭代次数。
        threshold_ema_dead_code (int): “死码”阈值。如果某些码本向量长期未被选中，
                                       其 EMA 计数小于该阈值，则用当前 batch 的样本随机替换，避免塌陷。
    典型输入/输出形状（以 x 为例）：
        x: (B, D, T)  —— D=dimension，T 为时间步（或帧序列长度）
        codes: (n_q, B, T)
        quantized: (B, D, T)
        commit_loss: 标量（或维度为 (B,) 的张量），训练时纳入总损失
    """
    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # 实例化核心 RVQ 对象（真正的码本、EMA 更新、编码/解码逻辑在其中）
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )

    def forward(self, x: torch.Tensor, n_q: tp.Optional[int] = None, layers: tp.Optional[list] = None):
        """对输入特征执行残差向量量化（用于训练/推理的正向路径）。

        参数：
            x (torch.Tensor): 输入连续特征，形状 (B, D, T)。
                              注意 D 必须与 dimension 一致。
            n_q (int, 可选): 实际使用的量化层数。默认使用 self.n_q 全部层。
                             可以在推理时降低层数以控制带宽/码率（有损更大）。
            layers (list[int], 可选): 指定想要返回的某些量化层的“量化后特征”。
                                     例如 [0] 只返回第 1 层的量化特征（常用于语义对齐）。
                                     若为 None，则内部仍会完成所有层量化，只是不会单独挑选列表返回。

        返回：
            quantized (torch.Tensor): 叠加所有已用量化层后的连续近似特征 (B, D, T)，
                                      供解码器使用。
            codes (torch.Tensor): 每层的离散索引（码本条目下标），形状 (n_q, B, T)。
            mean_commit_loss (torch.Tensor): 对应 RVQ 的承诺损失（取 batch/层平均），标量。
            quantized_list (List[Tensor]): 若指定 layers，则按顺序返回这些层各自的量化特征，
                                           每个元素形状均为 (B, D, T)。

        说明：
            - RVQ 的核心思想：
                e ≈ q1 + q2 + ... + q_{n_q}
              其中 qk 是第 k 个量化器从其码本中选出的量化向量（按残差迭代）。
            - commit_loss（承诺损失）鼓励编码器输出靠近选中的码本向量，避免震荡。
            - 若 layers 的最大层索引超出 n_q-1，会抛异常（见下方检查）。
        """
        n_q = n_q if n_q else self.n_q

        # layers 合法性检查：最大层索引必须 < 使用的层数 n_q
        if layers and max(layers) >= n_q:
            raise ValueError(
                f'Last layer index in layers: A {max(layers)}. '
                f'Number of quantizers in RVQ: B {self.n_q}. A must less than B.'
            )

        # 调用核心 RVQ：返回
        #   quantized: (B, D, T)
        #   codes:     (n_q, B, T)
        #   commit_loss: (可能为 (B,) 或 (n_q, B) 等) —— 这里用 mean 聚合为标量
        #   quantized_list: 针对 layers 的每层量化特征列表（元素均 (B, D, T)）
        quantized, codes, commit_loss, quantized_list = self.vq(
            x, n_q=n_q, layers=layers
        )

        # 训练时一般把 commit_loss 记到总损失里，这里先做 mean（按 batch/层聚合）
        return quantized, codes, torch.mean(commit_loss), quantized_list


    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None, st: tp.Optional[int] = None) -> torch.Tensor:
        """仅做“编码”：把连续特征 x 映射为离散 codes（不返回连续量化特征）。

        常用于推理（提取语音 token）或离线标注阶段。

        参数：
            x (torch.Tensor): 输入特征，形状 (B, D, T)。
            n_q (int, 可选): 使用的量化层数（默认全部 self.n_q）。
            st (int, 可选): 从第几层量化器开始编码（默认 0）。
                            例如 st=2, n_q=3 → 使用第 2、3、4 层来编码。

        返回：
            codes (torch.Tensor): 离散索引，形状 (n_q, B, T)。
        """
        n_q = n_q if n_q else self.n_q
        st = st or 0
        codes = self.vq.encode(x, n_q=n_q, st=st)
        return codes

    def decode(self, codes: torch.Tensor, st: int = 0) -> torch.Tensor:
        """仅做“解码”：把离散 codes 查码本还原为连续量化特征（供解码器重建波形）。

        参数：
            codes (torch.Tensor): 输入离散索引，形状 (n_q, B, T)。
            st (int): 从第几层开始解码（默认 0），需与 encode 时的起始层匹配。

        返回：
            quantized (torch.Tensor): 叠加各层量化向量后的连续特征 (B, D, T)，
                                      可直接输入解码器。
        """
        quantized = self.vq.decode(codes, st=st)
        return quantized
