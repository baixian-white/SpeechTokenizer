"""Entropy lower-bound stub for I1 trichotomy (left term).

STATUS: UNTESTED / DEFERRED (BLOCKED-ON-BASE, Experiment B).
Requires numpy + real RVQ index arrays from a trained SCIT-Speech-Base, neither of
which exists in run redoA_interface_contract_20260620. Encodes the methodology in
reports/entropy_lower_bound_DEFERRED.md (steps 3-6) so Experiment B can run it as-is.

Input: one or more .npy files, each shape (n_q, T) of integer RVQ indices in [0, K).
       Pass all utterances of a split together; histograms accumulate across files.

Usage (Exp B):
    python entropy_lower_bound_stub.py --K 1024 --f_q 50 split_clean/*.npy
"""
import argparse
import math


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-layer RVQ index entropy -> bitrate lower bound (DEFERRED stub).")
    ap.add_argument("npy", nargs="+", help="index .npy files, each shape (n_q, T)")
    ap.add_argument("--K", type=int, default=1024, help="codebook size")
    ap.add_argument("--f_q", type=float, default=50.0, help="latent frame rate Hz")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    import numpy as np  # deferred import: not available/needed in run A

    n_q = None
    counts = None  # list of length-K count arrays, one per layer
    for path in args.npy:
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected 2D (n_q, T), got shape {arr.shape}")
        if n_q is None:
            n_q = arr.shape[0]
            counts = [np.zeros(args.K, dtype=np.int64) for _ in range(n_q)]
        elif arr.shape[0] != n_q:
            raise ValueError(f"{path}: n_q={arr.shape[0]} != {n_q}")
        for layer in range(n_q):
            idx = arr[layer].astype(np.int64)
            if idx.min() < 0 or idx.max() >= args.K:
                raise ValueError(f"{path} layer {layer}: index outside [0,{args.K})")
            binc = np.bincount(idx, minlength=args.K)
            counts[layer] += binc

    # Per-layer Shannon entropy H_l (bits/index), step 4.
    H = []
    for layer in range(n_q):
        total = counts[layer].sum()
        if total == 0:
            H.append(0.0)
            continue
        p = counts[layer][counts[layer] > 0] / total
        H_l = float(-(p * np.log2(p)).sum())
        H.append(H_l)

    # Cumulative lossless lower bound at level L (steps 5-6).
    results = []
    cum = 0.0
    for layer in range(n_q):
        cum += H[layer]
        L = layer + 1
        lb = args.f_q * cum
        uniform = 500.0 * L  # = f_q * 10 * L
        results.append(
            {
                "L": L,
                "H_layer_bits": round(H[layer], 4),
                "cum_bits_per_frame": round(cum, 4),
                "entropy_lower_bound_bps": round(lb, 2),
                "uniform_500L_bps": uniform,
                "below_uniform_pct": round((uniform - lb) / uniform * 100, 2),
                "trichotomy_lower_holds": lb <= uniform,
            }
        )

    import json
    out = {"K": args.K, "f_q": args.f_q, "n_q": n_q, "per_layer_entropy_bits": [round(h, 4) for h in H], "levels": results}
    text = json.dumps(out, indent=2)
    print(text)
    if args.json_out:
        from pathlib import Path
        Path(args.json_out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
