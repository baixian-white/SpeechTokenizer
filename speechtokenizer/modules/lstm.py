# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

from torch import nn


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True, bidirectional: bool=False):
        #dimension = 1024,num_layers = 2,bidirectional  = true skip  = true
        super().__init__()
        self.bidirectional = bidirectional
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers, bidirectional=bidirectional)
# 2 层 LSTM 是通用的高质量编码器配置。第 1 层：捕捉局部时间结构   第 2 层：捕捉更长时间依赖（整句的节奏、韵律）
    def forward(self, x):
        x = x.permute(2, 0, 1) #(B, C, T) -> (T, B, C)
        y, _ = self.lstm(x) #将x传入做LSTM处理，得到y 如果bidirectional = true,那么   y = (T, B, C*2)，如果bidirectional = false,那么   y = (T, B, C)
        if self.bidirectional:#如果lstm网络是双向的
            x = x.repeat(1, 1, 2) #把x复制一份，然后contact，（B, C, T) -> (B, C, T*2)）,方便后面残差相加
        if self.skip: #如果skip为true
            y = y + x #相加
        y = y.permute(1, 2, 0) #(T, B, C) -> (B, C, T) 
        return y
