# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Core vector quantization implementation.

import typing as tp

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F

from .distrib import broadcast_tensors, rank  # 分布式训练相关（本文件里处于注释/保留状态）


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    """若 val 非 None 则返回 val，否则返回默认值 d。"""
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    """对 moving_avg 做就地 EMA 更新：moving_avg = decay * moving_avg + (1-decay) * new"""
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    """拉普拉斯平滑，防止计数为 0 导致除零等数值问题。"""
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    """用 kaiming_uniform_ 初始化张量（用于码本向量的默认随机初始化）。"""
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    """从 samples（N × D）中随机采样 num 个向量，作为初始化用。"""
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    """简单的 k-means 聚类，用于码本向量初始化（kmeans_init=True 时）。"""
    dim, dtype = samples.shape[-1], samples.dtype
    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        # 计算每个样本到每个中心的负平方距离（越大越近）
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs ** 2).sum(dim=-1)

        # 分配簇
        buckets = dists.max(dim=-1).indices          # 最近中心的索引
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0                        # 处理空簇
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        # 计算新中心（均值）
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        # 空簇保持原中心不变
        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    """使用欧氏距离的码本（codebook）。
    负责：
      - 码本向量存储/初始化（支持 k-means）
      - 根据欧氏距离做最近邻量化（encode/quantize）
      - 根据索引反查向量（decode/dequantize）
      - 训练时的 EMA 更新与“死码”替换策略
    形状约定：
      - 码本：embed ∈ R[K, D]（K=codebook_size，D=dim）
      - 输入 x：任意形状，最后一维为 D；内部会展平为 (..., D) → (N, D)
    """
    def __init__(
        self,
        dim: int, #码本向量的维度
        codebook_size: int, #码本向量的数量
        kmeans_init: int = False,     # 可为 bool；此处保留原签名，是否用kmeans来初始化码本
        kmeans_iters: int = 10,# K-Means 初始化迭代次数
        decay: float = 0.99,#EMA 衰减系数；越接近 1，更新越平滑（训练中用于码本向量的滑动更新）
        epsilon: float = 1e-5, #数值稳定项（例如拉普拉斯平滑时防止除零）
        threshold_ema_dead_code: int = 2, #“死码”阈值。某个码本向量长期未被选择（EMA 统计小于该阈值），会被当前 batch 的样本替换掉。
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros 
        #这个实际就是个条件函数，如果kmeans_init == False，采用 uniform_init（随机初始化码本向量），如果用kmeans_init == True
        #那么暂时用 torch.zeros 创建一个占位张量，稍后会用 K-Means 结果替换
        embed = init_fn(codebook_size, dim)              # 初始码本 K×D

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # 维护若干缓冲区（buffer）：是否已初始化、簇大小（用于 EMA 统计）、码本、EMA 版码本
        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)             # 实际使用的码本向量
        self.register_buffer("embed_avg", embed.clone()) # EMA 平均的码本向量

    @torch.jit.ignore
    def init_embed_(self, data):
        """在第一批数据到来时做 k-means 初始化（仅 when 未初始化且 kmeans_init=True）。"""
        if self.inited:
            return
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # 可选：多卡同步
        # broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        """用 batch 中的样本替换“死码”（mask=True 的位置）。"""
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        """剔除/替换“死码”，阈值由 threshold_ema_dead_code 控制。"""
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        # broadcast_tensors(self.buffers())

    def preprocess(self, x):
        """把输入 x 的最后一维视为 D，把其余维度展平为 N： (..., D) → (N, D)。"""
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        """基于欧氏距离的最近邻查找（x ∈ R[N, D]，返回最近码本索引 ∈ R[N]）。"""
        embed = self.embed.t()  # D × K
        # dist = -||x - e||^2；用展开公式避免显式构造 N×K×D 的大张量
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )  # 形状 N × K
        embed_ind = dist.max(dim=-1).indices  # N
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        """把平铺的索引还原回原始 shape 的前几维（去掉最后一维 D）"""
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        """索引 → 码本向量（embedding 查表），返回 (..., D)。"""
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        """只做编码：返回最近邻索引。"""
        shape = x.shape
        x = self.preprocess(x)                # (N, D)
        embed_ind = self.quantize(x)          # (N,)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        """只做解码：索引 → 码本向量。"""
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        """量化前向：返回量化向量和索引；训练时做 EMA 更新与死码处理。"""
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)                # (N, D)

        self.init_embed_(x)                   # 若需要，做 k-means 初始化

        embed_ind = self.quantize(x)          # 最近邻索引 (N,)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)  # (N, K)
        embed_ind = self.postprocess_emb(embed_ind, shape)                   # 还原回输入 shape 的前几维
        quantize = self.dequantize(embed_ind) # 量化向量 (..., D)

        if self.training:
            # 训练时：先做死码替换，再做 EMA 更新
            self.expire_codes_(x)                                     # 处理死码
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)

            embed_sum = x.t() @ embed_onehot                          # D×K
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)    # K×D
 
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)  # K×D
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """单层向量量化器（VQ）。
    - 支持在进入码本前/后做线性投影（project_in/out），允许 codebook_dim 与特征维 dim 不同
    - 内部使用 EuclideanCodebook 做最近邻查找与 EMA 维护
    - 训练时用 straight-through（quantize = x + (quantize - x).detach()）保持梯度通过
    I/O 形状约定：
      输入 x: (B, D, N) → 先重排为 (B, N, D) 做量化 → 输出再重排回 (B, D, N)
      embed_ind: (B, N)
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        # 若 codebook_dim != dim，则在进入/离开码本前后各做一次线性映射
        requires_projection = _codebook_dim != dim
        self.project_in  = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim, codebook_size=codebook_size,
            kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
            decay=decay, epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        """返回码本矩阵（K × D）。"""
        return self._codebook.embed

    def encode(self, x):
        """仅编码：返回索引 (B, N)。"""
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)   # (b, n)
        return embed_in

    def decode(self, embed_ind):
        """仅解码：索引 → 量化向量 (B, D, N)。"""
        quantize = self._codebook.decode(embed_ind)     # (b, n, d_codebook)
        quantize = self.project_out(quantize)           # (b, n, dim)
        quantize = rearrange(quantize, "b n d -> b d n") 
        return quantize

    def forward(self, x):
        """量化前向：返回 (quantize, embed_ind, loss)。"""
        device = x.device
        x = rearrange(x, "b d n -> b n d")   # (B,D,N)->(B,N,D)
        x = self.project_in(x) #  (B, N, D) → (B, N, D')相当于把d在线性映射为D',这个D'是码本的维度

        quantize, embed_ind = self._codebook(x)  # quantize: (B,N,D'), embed_ind: (B,N)

        if self.training:
            # Straight-Through Estimator: 前向用 quantize，反向对 x 传梯度
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training and self.commitment_weight > 0:
            # 承诺损失：鼓励 x 接近选中的码本向量（避免编码器输出频繁跳码）
            commit_loss = F.mse_loss(quantize.detach(), x)
            loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """残差向量量化（RVQ）。
    参见 "Residual Quantization" 思想：第 1 层拟合主成分，后续层拟合剩余残差。
    伪代码（forward）：
        residual = x
        out = 0
        for each layer:
            q, idx, loss = VQ(residual)
            residual = residual - q
            out = out + q
    最终输出：
        - quantized_out: 叠加所有层的量化向量（连续） (B, D, N)
        - out_indices:   每层索引 (n_q, B, N)
        - out_losses:    每层 commitment loss（可堆叠求均值） (n_q, ...)
        - out_quantized: 若用户要求返回某些层的量化向量，则集中放这里（列表）
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x, n_q: tp.Optional[int] = None, layers: tp.Optional[list] = None):
        """对输入 x 执行前 n_q 层 RVQ 量化。
        参数：
          x: (B, D, N)
          n_q: 使用的层数（默认用完）
          layers: 想要额外返回量化向量的层号列表（如 [0]）
        返回：
          quantized_out: (B, D, N)        # 所有层量化向量之和
          out_indices:   (n_q, B, N)      # 每层的索引
          out_losses:    (n_q, ...)       # 每层的 loss（一般是标量；这里堆叠后再外部取均值）
          out_quantized: list[Tensor]     # 只包含 layers 指定层的量化结果，各为 (B, D, N)
        """
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []
        out_quantized = []

        n_q = n_q or len(self.layers)

        for i, layer in enumerate(self.layers[:n_q]):
            quantized, indices, loss = layer(residual)  # 单层 VQ
            residual = residual - quantized             # 更新残差
            quantized_out = quantized_out + quantized   # 累加量化向量

            all_indices.append(indices)
            all_losses.append(loss)
            if layers and i in layers:
                out_quantized.append(quantized)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses, out_quantized

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None, st: tp.Optional[int]= None) -> torch.Tensor:
        """仅编码：返回从第 st 层开始、共 n_q 层的索引堆叠 (n_q, B, N)。"""
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        st = st or 0
        for layer in self.layers[st:n_q]:
            indices = layer.encode(residual)    # (B, N)
            quantized = layer.decode(indices)   # (B, D, N) —— 用于递推残差
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)  # (n_q, B, N)
        return out_indices

    def decode(self, q_indices: torch.Tensor, st: int=0) -> torch.Tensor:
        """仅解码：把 (n_q, B, N) 索引序列解码为量化向量之和 (B, D, N)。"""
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[st + i]
            quantized = layer.decode(indices)   # (B, D, N)
            quantized_out = quantized_out + quantized
        return quantized_out
