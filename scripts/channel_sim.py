import argparse
import json
from pathlib import Path

import numpy as np


def apply_channel_sim(codes, L, codebook_size, p_drop=0.0, p_sub=0.0, seed=0):
    arr = np.asarray(codes).copy()
    if arr.ndim not in (2, 3):
        raise ValueError(f"expected codes with shape (L,T) or (L,B,T), got {arr.shape}")
    if L < 1 or L > arr.shape[0]:
        raise ValueError(f"L={L} is invalid for codes shape {arr.shape}")
    out = arr[:L].copy()
    if np.any(out < 0) or np.any(out >= codebook_size):
        raise ValueError("input contains illegal codebook indices")

    rng = np.random.default_rng(seed)
    total = int(out.size)
    drop_mask = rng.random(out.shape) < p_drop
    sub_mask = rng.random(out.shape) < p_sub

    replaced_by_previous = 0
    if p_drop > 0:
        if out.ndim == 2:
            for m in range(out.shape[0]):
                for t in range(out.shape[1]):
                    if drop_mask[m, t]:
                        if t > 0:
                            out[m, t] = out[m, t - 1]
                            replaced_by_previous += 1
        else:
            for m in range(out.shape[0]):
                for b in range(out.shape[1]):
                    for t in range(out.shape[2]):
                        if drop_mask[m, b, t]:
                            if t > 0:
                                out[m, b, t] = out[m, b, t - 1]
                                replaced_by_previous += 1

    substituted = 0
    if p_sub > 0:
        replacement = rng.integers(0, codebook_size, size=out.shape, endpoint=False)
        out[sub_mask] = replacement[sub_mask]
        substituted = int(sub_mask.sum())

    if out.shape != arr[:L].shape:
        raise ValueError("ChannelSim changed codes shape")
    if np.any(out < 0) or np.any(out >= codebook_size):
        raise ValueError("ChannelSim produced illegal codebook indices")

    stats = {
        "shape": list(out.shape),
        "L": L,
        "codebook_size": codebook_size,
        "p_drop": p_drop,
        "p_sub": p_sub,
        "seed": seed,
        "total_indices": total,
        "replaced_by_previous": replaced_by_previous,
        "substituted": substituted,
        "actual_p_drop": replaced_by_previous / total if total else 0.0,
        "actual_p_sub": substituted / total if total else 0.0,
    }
    return out, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--p-drop", type=float, default=0.0)
    parser.add_argument("--p-sub", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    codes = np.load(args.codes)
    out, stats = apply_channel_sim(codes, args.L, args.codebook_size, args.p_drop, args.p_sub, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, out)
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

