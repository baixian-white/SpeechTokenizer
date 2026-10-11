import torch
from torch import nn
from torch.nn import functional as F


class GatedEmbeddingFusion(nn.Module):
    def __init__(self, input_dim=256, output_dim=256):
        super().__init__()
        self.token_projection = nn.Linear(input_dim, output_dim)
        self.audio_projection = nn.Linear(input_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(input_dim * 2, output_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, token_embedding, audio_embedding):
        gate = self.gate(torch.cat([token_embedding, audio_embedding], dim=-1))
        fused = gate * self.token_projection(token_embedding)
        fused = fused + (1.0 - gate) * self.audio_projection(audio_embedding)
        return F.normalize(self.norm(fused), dim=-1), gate
