# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:47:55 2023
@author: zhangxin
说明：
------
SpeechTokenizer 模型主文件
整体结构：
    输入波形 → 编码器（SEANetEncoder）
              → 残差向量量化器（ResidualVectorQuantizer, RVQ）
              → 解码器（SEANetDecoder）
可实现语音压缩与语义 Token 化。
"""

from .modules.seanet import SEANetEncoder, SEANetDecoder     # 导入 SEANet 编码器与解码器（Encodec 同源结构）
from .quantization  import ResidualVectorQuantizer            # 导入残差向量量化器（核心 Tokenizer）
import torch.nn as nn
from einops import rearrange                                 # 用于维度变换（简化 tensor 转置）
import torch
import numpy as np

class SpeechTokenizer(nn.Module):
    def __init__(self, config):
        '''
        初始化模型组件
        Parameters
        ----------
        config : dict/json
            模型参数配置，常来自 JSON 文件（见 config.json）
        '''
        super().__init__()

        # 1️⃣ 定义语音编码器（SEANetEncoder）
        self.encoder = SEANetEncoder(
            n_filters=config.get('n_filters'),               # 第一层卷积的通道数
            dimension=config.get('dimension'),               # 隐空间维度 D
            ratios=config.get('strides'),                    # 下采样步长（各层比例的乘积决定总压缩率）
            lstm=config.get('lstm_layers'),                  # LSTM 层数
            bidirectional=config.get('bidirectional'),       # 是否双向 LSTM
            dilation_base=config.get('dilation_base'),       # 空洞卷积倍数
            residual_kernel_size=config.get('residual_kernel_size'),  # 残差卷积核大小
            n_residual_layers=config.get('n_residual_layers'),        # 残差层数
            activation=config.get('activation')              # 激活函数（如 ELU）
        )

        # 2️⃣ 基本配置缓存
        self.sample_rate = config.get('sample_rate')         # 音频采样率（如 16000）
        self.n_q = config.get('n_q')                         # RVQ 层数（码本层数）
        self.downsample_rate = np.prod(config.get('strides'))# 总体时间下采样率

        # 3️⃣ 若编码维度 ≠ 语义维度，则添加线性变换层用于特征对齐
        if config.get('dimension') != config.get('semantic_dimension'):
            self.transform = nn.Linear(
                config.get('dimension'), 
                config.get('semantic_dimension') #768维
            )
        else:
            self.transform = nn.Identity()                   # 若相同，则不变换

        # 4️⃣ 残差向量量化器（核心 Tokenizer）
        self.quantizer = ResidualVectorQuantizer(
            dimension=config.get('dimension'),               # 输入特征维度1024
            n_q=config.get('n_q'),                           # RVQ 层数
            bins=config.get('codebook_size')                 # 每层码本大小1024
        )

        # 5️⃣ 定义语音解码器（SEANetDecoder）
        self.decoder = SEANetDecoder(
            n_filters=config.get('n_filters'),
            dimension=config.get('dimension'),
            ratios=config.get('strides'),                    # 与编码器对称的上采样比
            lstm=config.get('lstm_layers'),
            bidirectional=False,                             # 解码器一般使用单向 LSTM
            dilation_base=config.get('dilation_base'),
            residual_kernel_size=config.get('residual_kernel_size'),
            n_residual_layers=config.get('n_residual_layers'),
            activation=config.get('activation')
        )

    # 🔧 从配置与权重加载模型
    @classmethod
    def load_from_checkpoint(cls, config_path: str, ckpt_path: str):
        '''
        从配置文件与权重文件加载模型（常用于推理阶段）
        Parameters
        ----------
        config_path : str
            模型配置文件路径 (.json)
        ckpt_path : str
            模型权重文件路径 (.pt / .pth)
        Returns
        -------
        model : SpeechTokenizer
            已加载权重的完整模型
        '''
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        model = cls(cfg)
        params = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(params)
        return model

    # 🚀 前向传播（编码 → 量化 → 解码）
    def forward(self, x: torch.tensor, n_q: int=None, layers: list=[0]):
        '''
        Parameters
        ----------
        x : torch.tensor
            输入波形，形状 (batch, channels, timesteps)
        n_q : int, optional
            使用的量化层数（默认全部）
        layers : list[int], optional
            想要返回的 RVQ 层（默认第一层）
        Returns
        -------
        o : torch.tensor
            输出重建波形 (B, C, T)
        commit_loss : torch.tensor
            RVQ 的承诺损失（训练时约束码本使用）
        feature : torch.tensor
            RVQ 第1层输出特征 (B, T', D)
        '''
        n_q = n_q if n_q else self.n_q                      # 若未指定则使用全层
        e = self.encoder(x)                                 # ① 编码器输出 e: (B, D, T')
        quantized, codes, commit_loss, quantized_list = self.quantizer(
            e, n_q=n_q, layers=layers
        )                                                   # ② RVQ 量化
        feature = rearrange(quantized_list[0], 'b d t -> b t d') # ③ 取首层特征并转为 (B, T', D) 所以在前向传播的时候，
        feature = self.transform(feature)                   # ④ 若有 Linear，映射到语义维度
        o = self.decoder(quantized)                         # ⑤ 解码器重建波形
        return o, commit_loss, feature

    # 🔍 提取指定层的量化特征（不解码）
    def forward_feature(self, x: torch.tensor, layers: list=None):
        '''
        用于可视化 / 特征分析。
        Parameters
        ----------
        x : torch.tensor
            输入波形 (B, C, T)
        layers : list[int], optional
            返回的 RVQ 层编号（默认全部）
        Returns
        -------
        quantized_list : list[torch.tensor]
            每层量化特征列表 (B, D, T')
        '''
        e = self.encoder(x)
        layers = layers if layers else list(range(self.n_q))
        quantized, codes, commit_loss, quantized_list = self.quantizer(e, layers=layers)
        return quantized_list

    # 🔡 编码：将波形 → 离散语义 token
    def encode(self, x: torch.tensor, n_q: int=None, st: int=None):
        '''
        Parameters
        ----------
        x : torch.tensor
            输入波形 (B, C, T)
        n_q : int, optional
            使用多少层 RVQ 编码（默认全层）
        st : int, optional
            从第几层开始（默认 0）
        Returns
        -------
        codes : torch.tensor
            各层的离散码索引 (n_q, B, T')
        '''
        e = self.encoder(x)
        if st is None:
            st = 0
        n_q = n_q if n_q else self.n_q
        codes = self.quantizer.encode(e, n_q=n_q, st=st)    # 编码为离散索引
        return codes

    # 🔊 解码：将离散 token → 重建波形
    def decode(self, codes: torch.tensor, st: int=0):
        '''
        Parameters
        ----------
        codes : torch.tensor
            每层离散索引 (n_q, B, T')
        st : int, optional
            起始量化层（默认 0）
        Returns
        -------
        o : torch.tensor
            重建的波形 (B, C, T)
        '''
        quantized = self.quantizer.decode(codes, st=st)     # 离散码 → 连续特征
        o = self.decoder(quantized)                         # 解码回波形
        return o
