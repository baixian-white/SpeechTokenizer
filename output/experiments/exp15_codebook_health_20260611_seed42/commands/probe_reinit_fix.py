"""exp15 feasibility probe (CPU, minutes): does fixing the reinit-clobber bug actually
reduce dead-code rate? Compares original EuclideanCodebook vs a corrected version where
expire_codes_ runs AFTER the EMA embed recompute (and resets the revived code's EMA state
so it is not immediately re-killed).

This is a CHEAP go/no-go test before investing 1-2 days of GPU on full exp15.
NOT a final result — uses synthetic clustered data, not the real distill30 pipeline.

Run: conda run -n speechtokenizer python <this file>
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speechtokenizer.quantization.core_vq import (
    EuclideanCodebook, ema_inplace, laplace_smoothing,
)


class FixedEuclideanCodebook(EuclideanCodebook):
    """Same as EuclideanCodebook but expire_codes_ runs AFTER embed is recomputed
    from EMA, and the revived dead code's embed_avg/cluster_size are reset so the
    fresh vector survives (instead of being overwritten same-step)."""

    def forward(self, x):
        import torch.nn.functional as F
        from einops import rearrange  # noqa
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
            # ---- FIX: revive dead codes AFTER embed recompute, and reset their EMA ----
            if self.threshold_ema_dead_code > 0:
                expired = self.cluster_size < self.threshold_ema_dead_code
                n_exp = int(expired.sum().item())
                if n_exp > 0:
                    from speechtokenizer.quantization.core_vq import sample_vectors
                    samples = sample_vectors(x, self.codebook_size)
                    new_embed = torch.where(expired[..., None], samples, self.embed)
                    self.embed.data.copy_(new_embed)
                    # reset revived codes' EMA so they get a fair chance, not re-killed next step
                    self.embed_avg.data[expired] = self.embed.data[expired]
                    self.cluster_size.data[expired] = self.threshold_ema_dead_code  # seed above threshold
        return quantize, embed_ind


def make_codebook(cls, K=256, dim=16, seed=0):
    torch.manual_seed(seed)
    cb = cls(dim=dim, codebook_size=K, kmeans_init=False, threshold_ema_dead_code=2)
    cb.train()
    return cb


def dead_rate(cb):
    return float((cb.cluster_size < cb.threshold_ema_dead_code).float().mean().item())


def used_codes(cb, data, K):
    """count distinct codes actually selected on a held-out batch (eval mode)."""
    cb.eval()
    with torch.no_grad():
        idx = cb.encode(data)
    cb.train()
    return len(torch.unique(idx))


def run(cls, label, steps=400, K=256, dim=16, n_modes=40, seed=0):
    """Train on synthetic data drawn from n_modes Gaussian clusters. A healthy codebook
    should occupy many codes; a clobbered-reinit one collapses to few."""
    torch.manual_seed(seed)
    cb = make_codebook(cls, K=K, dim=dim, seed=seed)
    centers = torch.randn(n_modes, dim) * 5.0
    for _ in range(steps):
        # batch: pick random modes, add noise -> (B, T, dim)
        B, T = 4, 32
        mode_idx = torch.randint(0, n_modes, (B * T,))
        batch = centers[mode_idx] + torch.randn(B * T, dim) * 0.3
        batch = batch.view(B, T, dim)
        cb(batch)
    # eval on fresh data covering all modes
    eval_idx = torch.arange(n_modes).repeat_interleave(8)
    eval_data = (centers[eval_idx] + torch.randn(len(eval_idx), dim) * 0.3).view(1, -1, dim)
    used = used_codes(cb, eval_data, K)
    dr = dead_rate(cb)
    n_zero_norm = int((cb.embed.norm(dim=1) < 1e-6).sum().item())
    print(f"{label:14s}: dead_rate={dr:.1%}  used_codes_on_eval={used}/{K}  "
          f"zero_norm_codes={n_zero_norm}/{K}")
    return dr, used


def main():
    print("Synthetic data: 40 Gaussian clusters, K=256 codebook, 400 train steps.")
    print("A healthy codebook should USE many codes (>=40, ideally more); a no-op-reinit one collapses.\n")
    for seed in (0, 1, 2):
        print(f"--- seed {seed} ---")
        run(EuclideanCodebook, "ORIGINAL", seed=seed)
        run(FixedEuclideanCodebook, "FIXED(reinit)", seed=seed)
        print()


if __name__ == "__main__":
    main()
