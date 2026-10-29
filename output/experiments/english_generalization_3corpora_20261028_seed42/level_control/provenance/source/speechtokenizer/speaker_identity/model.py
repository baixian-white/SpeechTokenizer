from dataclasses import dataclass

import torch
from torch import nn

from .fusion import GatedEmbeddingFusion
from .losses import AAMSoftmaxHead


@dataclass
class SpeakerModelOutput:
    token_embedding: torch.Tensor
    audio_embedding: torch.Tensor
    fusion_embedding: torch.Tensor
    token_logits: torch.Tensor
    audio_logits: torch.Tensor
    fusion_logits: torch.Tensor
    gate: torch.Tensor


class SpeakerIdentityModel(nn.Module):
    def __init__(self, token_encoder, audio_encoder, speaker_count=110, embedding_dim=256, scale=30.0, margin=0.2):
        super().__init__()
        self.token_encoder = token_encoder
        self.audio_encoder = audio_encoder
        self.fusion = GatedEmbeddingFusion(embedding_dim, embedding_dim)
        self.token_head = AAMSoftmaxHead(embedding_dim, speaker_count, scale, margin)
        self.audio_head = AAMSoftmaxHead(embedding_dim, speaker_count, scale, margin)
        self.fusion_head = AAMSoftmaxHead(embedding_dim, speaker_count, scale, margin)

    def forward(self, batch):
        labels = batch.get('labels') if self.training else None
        token_embedding = self.token_encoder(batch['codes'], batch['frame_mask'], batch['layer_mask'])
        audio_input = batch.get('audio_features', batch.get('waveform'))
        audio_embedding = self.audio_encoder(audio_input)
        fusion_embedding, gate = self.fusion(token_embedding, audio_embedding)
        return SpeakerModelOutput(
            token_embedding=token_embedding,
            audio_embedding=audio_embedding,
            fusion_embedding=fusion_embedding,
            token_logits=self.token_head(token_embedding, labels),
            audio_logits=self.audio_head(audio_embedding, labels),
            fusion_logits=self.fusion_head(fusion_embedding, labels),
            gate=gate,
        )
