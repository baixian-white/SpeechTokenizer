import torch
from torch import nn
from torch.nn import functional as F


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, value, mask):
        update = self.depthwise(value)
        update = self.pointwise(F.gelu(update))
        update = self.norm(update)
        return F.gelu(value + update) * mask


class RVQTokenEncoder(nn.Module):
    def __init__(self, rvq_layers=3, codebook_size=1024, embedding_dim=128, model_dim=256, output_dim=256, block_count=4, pad_index=1024):
        super().__init__()
        self.rvq_layers = rvq_layers
        self.pad_index = pad_index
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size + 1, embedding_dim, padding_idx=pad_index)
            for _ in range(rvq_layers)
        ])
        self.layer_embeddings = nn.Parameter(torch.zeros(rvq_layers, embedding_dim))
        nn.init.normal_(self.layer_embeddings, std=0.02)
        self.input_projection = nn.Linear(embedding_dim, model_dim)
        self.blocks = nn.ModuleList([TemporalResidualBlock(model_dim, 2 ** index) for index in range(block_count)])
        self.attention = nn.Sequential(nn.Conv1d(model_dim, model_dim // 2, 1), nn.Tanh(), nn.Conv1d(model_dim // 2, 1, 1))
        self.output_projection = nn.Linear(model_dim * 2, output_dim)

    def forward(self, codes, frame_mask, layer_mask):
        if codes.ndim != 3 or codes.shape[1] != self.rvq_layers:
            raise ValueError('codes must have shape [B,L,T]')
        if frame_mask.shape != (codes.shape[0], codes.shape[2]):
            raise ValueError('frame_mask must have shape [B,T]')
        if layer_mask.shape != (codes.shape[0], self.rvq_layers):
            raise ValueError('layer_mask must have shape [B,L]')
        frame_weight = frame_mask.to(codes.device, torch.float32).unsqueeze(1)
        layer_weight = layer_mask.to(codes.device, torch.float32)
        combined = 0.0
        for layer_index, embedding_table in enumerate(self.token_embeddings):
            value = embedding_table(codes[:, layer_index].clamp(0, self.pad_index))
            value = value + self.layer_embeddings[layer_index]
            combined = combined + value * layer_weight[:, layer_index].view(-1, 1, 1)
        denominator = layer_weight.sum(dim=1).clamp_min(1.0).view(-1, 1, 1)
        value = self.input_projection(combined / denominator).transpose(1, 2) * frame_weight
        for block in self.blocks:
            value = block(value, frame_weight)
        attention_logits = self.attention(value).squeeze(1)
        attention_logits = attention_logits.masked_fill(~frame_mask.to(torch.bool), -1e4)
        weights = torch.softmax(attention_logits, dim=-1).unsqueeze(1)
        mean = (value * weights).sum(dim=-1)
        variance = ((value - mean.unsqueeze(-1)) ** 2 * weights).sum(dim=-1).clamp_min(1e-6)
        pooled = torch.cat([mean, variance.sqrt()], dim=-1)
        return F.normalize(self.output_projection(pooled), dim=-1)
