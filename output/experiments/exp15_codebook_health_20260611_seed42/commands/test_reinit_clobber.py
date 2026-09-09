"""exp15 mechanism test (CPU-only): does EuclideanCodebook.expire_codes_ actually
persist a reinitialized dead code, or is it clobbered by the same-step EMA embed
recompute (line ~221: embed <- embed_avg / cluster_size)?

Hypothesis: dead-code reinit is a no-op because forward() overwrites the entire
embed buffer from embed_avg after expire_codes_ runs, and the replaced code's
embed_avg was not updated this step.

This does NOT touch GPU and trains nothing. It isolates one codebook layer.

Run: conda run -n speechtokenizer python <this file>
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speechtokenizer.quantization.core_vq import EuclideanCodebook


def main():
    torch.manual_seed(0)
    dim, K = 8, 32
    cb = EuclideanCodebook(dim=dim, codebook_size=K, kmeans_init=False,
                           threshold_ema_dead_code=2)
    cb.train()
    # Make codebook "warm": run a few steps so embed/embed_avg/cluster_size are populated
    for _ in range(20):
        x = torch.randn(4, 10, dim)  # (B, T, D)
        cb(x)

    # Identify a dead code: pick the code with the smallest cluster_size
    dead_idx = int(cb.cluster_size.argmin().item())
    cs_dead = cb.cluster_size[dead_idx].item()
    print(f"chosen dead_idx={dead_idx} cluster_size={cs_dead:.4f} "
          f"(threshold={cb.threshold_ema_dead_code})")
    print(f"dead? {cs_dead < cb.threshold_ema_dead_code}")

    embed_before = cb.embed[dead_idx].clone()

    # Manually mark it definitely-dead and capture what expire_codes_ writes.
    cb.cluster_size[dead_idx] = 0.0
    # Snapshot embed right after expire_codes_ would run, by replicating one forward
    # but intercepting. Easiest: monkeypatch replace_ to record the post-replace embed.
    recorded = {}
    orig_replace = cb.replace_

    def spy_replace(samples, mask):
        orig_replace(samples, mask)
        recorded["after_expire"] = cb.embed[dead_idx].clone()
    cb.replace_ = spy_replace

    x = torch.randn(4, 10, dim)
    cb(x)  # full forward in training mode
    embed_after_forward = cb.embed[dead_idx].clone()

    if "after_expire" not in recorded:
        print("expire_codes_ did NOT fire for this code (mask empty). "
              "Re-check threshold logic.")
        return

    after_expire = recorded["after_expire"]
    delta_replace = (after_expire - embed_before).norm().item()
    delta_clobber = (embed_after_forward - after_expire).norm().item()
    print(f"\n||embed_after_expire - embed_before|| = {delta_replace:.4f}  "
          f"(expire_codes_ wrote a new vector: {'YES' if delta_replace>1e-6 else 'no'})")
    print(f"||embed_after_full_forward - embed_after_expire|| = {delta_clobber:.4f}  "
          f"(same-step EMA overwrote the replacement: {'YES' if delta_clobber>1e-6 else 'no'})")

    # Verdict
    if delta_replace > 1e-6 and delta_clobber > 1e-6:
        print("\nVERDICT: dead-code reinit is CLOBBERED within the same forward step. "
              "expire_codes_ writes embed, then embed <- embed_avg/cluster_size erases it. "
              "=> dead-code reinit is effectively a NO-OP in this implementation.")
    elif delta_replace > 1e-6:
        print("\nVERDICT: reinit PERSISTS (not clobbered). Hypothesis refuted.")
    else:
        print("\nVERDICT: expire_codes_ did not change embed; inconclusive.")


if __name__ == "__main__":
    main()
