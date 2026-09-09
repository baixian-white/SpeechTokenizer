"""exp15 C2: isolated reinit-fix patch + ultra-short real-model continued training to prove
the fix reduces dead-code rate in the REAL distill30+GAN pipeline (not synthetic data).

DOES NOT modify the main speechtokenizer/quantization/core_vq.py. It monkey-patches each
EuclideanCodebook instance's forward at runtime, ONLY for this exp15 run.

Goal: a few hundred steps, dumping true cumulative codebook usage periodically, to show the
dead-code TREND under the fix. NOT a full 5-10 epoch retrain, NOT a quality/robustness claim.

This script is authored during C1; it is NOT launched until C1 is reviewed and user approves C2.
"""
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speechtokenizer.quantization.core_vq import (
    EuclideanCodebook, ema_inplace, laplace_smoothing, sample_vectors,
)


def fixed_forward(self, x):
    """Reinit-fix forward: expire_codes_ AFTER embed recompute + reset revived code EMA.
    Mirrors the original EuclideanCodebook.forward exactly except for the reinit placement."""
    shape, dtype = x.shape, x.dtype
    x = self.preprocess(x)
    self.init_embed_(x)
    embed_ind = self.quantize(x)
    embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
    embed_ind = self.postprocess_emb(embed_ind, shape)
    quantize = self.dequantize(embed_ind)

    if self.training:
        ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
        embed_sum = x.t() @ embed_onehot
        ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
        cluster_size = (
            laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
            * self.cluster_size.sum()
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embed.data.copy_(embed_normalized)
        # FIX: revive dead codes AFTER recompute; reset their EMA so they survive next step
        if self.threshold_ema_dead_code > 0:
            expired = self.cluster_size < self.threshold_ema_dead_code
            if bool(expired.any()):
                samples = sample_vectors(x, self.codebook_size)
                self.embed.data.copy_(torch.where(expired[..., None], samples, self.embed))
                self.embed_avg.data[expired] = self.embed.data[expired]
                self.cluster_size.data[expired] = float(self.threshold_ema_dead_code)
    return quantize, embed_ind


def patch_codebooks(model):
    """Replace forward on every EuclideanCodebook in the model's RVQ stack."""
    patched = 0
    for module in model.modules():
        if isinstance(module, EuclideanCodebook):
            module.forward = types.MethodType(fixed_forward, module)
            patched += 1
    return patched


if __name__ == "__main__":
    # Smoke: build nothing heavy here; just confirm patch wiring on a toy codebook.
    cb = EuclideanCodebook(dim=8, codebook_size=16, kmeans_init=False, threshold_ema_dead_code=2)
    before = cb.forward.__func__.__name__ if hasattr(cb.forward, "__func__") else cb.forward.__name__

    class _M(torch.nn.Module):
        def __init__(self, c): super().__init__(); self.c = c
    n = patch_codebooks(_M(cb))
    after = cb.forward.__func__.__name__
    print(f"patched {n} codebook(s); forward before/after = {before} -> {after}")
    assert n == 1 and after == "fixed_forward", "patch wiring failed"
    print("patch wiring OK")
