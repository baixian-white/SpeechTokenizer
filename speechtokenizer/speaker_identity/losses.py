import math

import torch
from torch import nn
from torch.nn import functional as F


class AAMSoftmaxHead(nn.Module):
    def __init__(self, embedding_dim, class_count, scale=30.0, margin=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(class_count, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(self, embedding, labels=None):
        cosine = F.linear(F.normalize(embedding, dim=-1), F.normalize(self.weight, dim=-1)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
        target = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).to(cosine.dtype)
        return (one_hot * target + (1.0 - one_hot) * cosine) * self.scale


def joint_classification_loss(fusion_logits, token_logits, audio_logits, labels, auxiliary_weight=0.3, consistency_weight=0.0, token_embedding=None, audio_embedding=None):
    total = F.cross_entropy(fusion_logits, labels)
    total = total + float(auxiliary_weight) * F.cross_entropy(token_logits, labels)
    total = total + float(auxiliary_weight) * F.cross_entropy(audio_logits, labels)
    if consistency_weight and token_embedding is not None and audio_embedding is not None:
        total = total + float(consistency_weight) * (1.0 - F.cosine_similarity(token_embedding, audio_embedding, dim=-1)).mean()
    return total
