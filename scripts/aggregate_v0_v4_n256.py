"""Aggregate V0-V4 unified evaluation (n=256) and compute mel robustness improvement.

For each variant V_x and each (L, perturbation) cell:
    rob_imp(sample) = mel_l1(base, sample, L, channel) - mel_l1(lca, sample, L, channel)
                     - (mel_l1(base, sample, L, clean) - mel_l1(lca, sample, L, clean))

Equivalently: rob_imp = base_degradation - lca_degradation, where degradation =
mel_l1(perturbed) - mel_l1(clean) per sample. Positive rob_imp means LCA degrades
less than Base under perturbation = LCA is more robust.

We pair on sample_id, compute per-cell mean ± paired bootstrap 95% CI, plus
paired t-test V4 vs V3 to see whether consistency loss adds significant rob_imp
on top of random-L + ChannelSim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"H:\H-CODE\speechtokenizer")
EVAL = ROOT / "output" / "experiments" / "exp5_lca_component_factorial_20260603_seed42" / "eval_unified_n64_20260609"

VARIANTS = {
    "V0": "V0_clean_control_step30000",
    "V1": "V1_random_l_only_step25000",
    "V2": "V2_channelsim_only_step17500",
    "V3": "V3_random_l_channelsim_step32500",
    "V4": "V4_full_lca_step30000",
}
PERTURB_CHANNELS = ["dropout-mid", "dropout-high", "substitution-mid", "substitution-high"]
L_VALUES = [1, 2, 3]


def load_variant(name: str) -> pd.DataFrame:
    p = EVAL / VARIANTS[name] / "metrics" / "base_vs_lca_results.csv"
    df = pd.read_csv(p, low_memory=False)
    df["L"] = df["L"].astype(int)
    df["mel_l1"] = df["mel_l1"].astype(float)
    df["stoi"] = pd.to_numeric(df["stoi"], errors="coerce")
    return df


def per_cell_rob_imp(df: pd.DataFrame, metric: str = "mel_l1") -> pd.DataFrame:
    """Return long-form table: variant_unused × (L, channel, sample_id) -> rob_imp."""
    pivot = df.pivot_table(
        index=["sample_id", "L", "channel"],
        columns="model",
        values=metric,
        aggfunc="first",
    ).reset_index()
    if "base" not in pivot.columns or "lca" not in pivot.columns:
        raise ValueError("expected both base and lca rows")
    clean_pivot = pivot[pivot["channel"] == "clean"][["sample_id", "L", "base", "lca"]].rename(
        columns={"base": "base_clean", "lca": "lca_clean"}
    )
    pert_pivot = pivot[pivot["channel"].isin(PERTURB_CHANNELS)]
    merged = pert_pivot.merge(clean_pivot, on=["sample_id", "L"])
    merged["base_degradation"] = merged["base"] - merged["base_clean"]
    merged["lca_degradation"] = merged["lca"] - merged["lca_clean"]
    merged["rob_imp"] = merged["base_degradation"] - merged["lca_degradation"]
    if metric == "stoi":
        # For STOI (higher better) we flip sign: degradation = clean - perturbed.
        merged["base_degradation"] = merged["base_clean"] - merged["base"]
        merged["lca_degradation"] = merged["lca_clean"] - merged["lca"]
        merged["rob_imp"] = merged["base_degradation"] - merged["lca_degradation"]
    return merged[["sample_id", "L", "channel", "base", "lca", "base_degradation", "lca_degradation", "rob_imp"]]


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def aggregate_variant(df: pd.DataFrame, metric: str = "mel_l1") -> pd.DataFrame:
    rob = per_cell_rob_imp(df, metric=metric)
    rng = np.random.default_rng(42)
    out = []
    for L in L_VALUES:
        for ch in PERTURB_CHANNELS:
            cell = rob[(rob["L"] == L) & (rob["channel"] == ch)]
            vals = cell["rob_imp"].dropna().to_numpy()
            mean = float(np.mean(vals)) if len(vals) else float("nan")
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            lo, hi = bootstrap_ci(vals, rng=rng)
            out.append({
                "L": L,
                "channel": ch,
                "n": len(vals),
                "rob_imp_mean": mean,
                "rob_imp_std": std,
                "rob_imp_ci_lo": lo,
                "rob_imp_ci_hi": hi,
            })
    return pd.DataFrame(out)


def paired_t_test(rob_a: pd.DataFrame, rob_b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    """Per (L, channel) cell: paired t-test of (rob_imp_a - rob_imp_b) on shared (sample_id)."""
    out = []
    for L in L_VALUES:
        for ch in PERTURB_CHANNELS:
            a = rob_a[(rob_a["L"] == L) & (rob_a["channel"] == ch)][["sample_id", "rob_imp"]].rename(columns={"rob_imp": "a"})
            b = rob_b[(rob_b["L"] == L) & (rob_b["channel"] == ch)][["sample_id", "rob_imp"]].rename(columns={"rob_imp": "b"})
            m = a.merge(b, on="sample_id")
            diff = (m["a"] - m["b"]).dropna().to_numpy()
            if len(diff) < 5:
                out.append({"L": L, "channel": ch, "n_paired": len(diff),
                            "diff_mean": float("nan"), "t_stat": float("nan"), "p_value": float("nan"),
                            "diff_ci_lo": float("nan"), "diff_ci_hi": float("nan")})
                continue
            t, p = stats.ttest_rel(m["a"], m["b"])
            d_mean = float(np.mean(diff))
            d_se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
            ci_lo = d_mean - 1.96 * d_se
            ci_hi = d_mean + 1.96 * d_se
            out.append({"L": L, "channel": ch, "n_paired": len(diff),
                        "diff_mean": d_mean, "t_stat": float(t), "p_value": float(p),
                        "diff_ci_lo": ci_lo, "diff_ci_hi": ci_hi})
    return pd.DataFrame(out)


def main():
    print("=" * 80)
    print(f"V0-V4 unified evaluation (n=256) aggregation")
    print("=" * 80)

    dfs = {name: load_variant(name) for name in VARIANTS}
    rob_per_variant = {name: per_cell_rob_imp(df, metric="mel_l1") for name, df in dfs.items()}

    # Per-variant: 12-cell mean rob_imp + grand mean
    print("\n## Per-variant 12-cell mean mel rob_imp (n=256 samples per cell)")
    print()
    summary_rows = []
    for name in VARIANTS:
        agg = aggregate_variant(dfs[name], metric="mel_l1")
        grand_mean = agg["rob_imp_mean"].mean()
        n_pos = int((agg["rob_imp_mean"] > 0).sum())
        # Bootstrap on the per-sample × per-cell pool for grand-mean CI
        all_vals = rob_per_variant[name]["rob_imp"].dropna().to_numpy()
        gm_lo, gm_hi = bootstrap_ci(all_vals, n_boot=5000)
        print(f"{name}: 12-cell mean rob_imp = {grand_mean:+.5f} "
              f"(95% CI [{gm_lo:+.5f}, {gm_hi:+.5f}], pos cells {n_pos}/12)")
        summary_rows.append({"variant": name, "mean_rob_imp": grand_mean,
                             "ci_lo": gm_lo, "ci_hi": gm_hi, "pos_cells": n_pos})

    summary_df = pd.DataFrame(summary_rows)
    print()
    print("## Comparison vs V0 (V0 is full-depth-only clean control)")
    v0_mean = summary_df.loc[summary_df["variant"] == "V0", "mean_rob_imp"].iloc[0]
    for _, r in summary_df.iterrows():
        delta = r["mean_rob_imp"] - v0_mean
        print(f"  {r['variant']} - V0: {delta:+.5f}")

    # Paired comparisons that matter most for the paper's claim
    print("\n## Paired comparisons (per-cell paired t-test on rob_imp)")
    print()
    for label_a, label_b in [("V4", "V3"), ("V4", "V0"), ("V3", "V2"), ("V2", "V1")]:
        print(f"### {label_a} vs {label_b}")
        tt = paired_t_test(rob_per_variant[label_a], rob_per_variant[label_b], label_a, label_b)
        for _, r in tt.iterrows():
            sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else ""))
            print(f"  L={r['L']} {r['channel']:18s}: Δ = {r['diff_mean']:+.5f} "
                  f"(95% CI [{r['diff_ci_lo']:+.5f}, {r['diff_ci_hi']:+.5f}]) "
                  f"t={r['t_stat']:+.2f}  p={r['p_value']:.2e} {sig}")
        # Pooled (all 12 cells stacked) paired t-test
        a = rob_per_variant[label_a][["sample_id", "L", "channel", "rob_imp"]].rename(columns={"rob_imp": "a"})
        b = rob_per_variant[label_b][["sample_id", "L", "channel", "rob_imp"]].rename(columns={"rob_imp": "b"})
        m = a.merge(b, on=["sample_id", "L", "channel"]).dropna()
        t, p = stats.ttest_rel(m["a"], m["b"])
        d = (m["a"] - m["b"]).to_numpy()
        d_mean = float(np.mean(d))
        d_se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
        print(f"  POOLED (all 12 cells, N={len(m)} pairs): Δ = {d_mean:+.5f} "
              f"(95% CI [{d_mean - 1.96 * d_se:+.5f}, {d_mean + 1.96 * d_se:+.5f}]) "
              f"t={t:+.2f}  p={p:.2e}")
        print()

    # Save aggregated tables
    out_dir = EVAL / "_aggregated_n256"
    out_dir.mkdir(exist_ok=True)
    summary_df.to_csv(out_dir / "summary_grand_mean_per_variant.csv", index=False)
    for name in VARIANTS:
        aggregate_variant(dfs[name], metric="mel_l1").to_csv(out_dir / f"{name}_per_cell_mel.csv", index=False)
    print(f"\nSaved aggregated tables to {out_dir}")


if __name__ == "__main__":
    main()
