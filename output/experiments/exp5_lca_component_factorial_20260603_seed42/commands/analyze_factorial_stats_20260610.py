from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(r"h:/H-CODE/speechtokenizer/output/experiments/exp5_lca_component_factorial_20260603_seed42/eval_unified_n64_20260609")
OUT = Path(r"h:/H-CODE/speechtokenizer/output/experiments/exp5_lca_component_factorial_20260603_seed42/reports/statistical_tests_20260610")
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "V0": "V0_clean_control_step30000",
    "V1": "V1_random_l_only_step25000",
    "V2": "V2_channelsim_only_step17500",
    "V3": "V3_random_l_channelsim_step32500",
    "V4": "V4_full_lca_step30000",
}
PERT_CHANNELS = ["dropout-mid", "dropout-high", "substitution-mid", "substitution-high"]


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    if n <= 1:
        return mean, float("nan"), float("nan"), float("nan")
    se = float(np.std(x, ddof=1) / math.sqrt(n))
    return mean, mean - 1.96 * se, mean + 1.96 * se, se


def ttest_1samp_zero(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    if not sd or math.isnan(sd):
        return float("nan"), float("nan")
    t = mean / (sd / math.sqrt(n))
    try:
        from scipy import stats
        p = float(stats.ttest_1samp(x, 0.0).pvalue)
    except Exception:
        # scipy is not always installed in this environment. With n=256 or 3072,
        # a normal approximation is sufficient for reporting order-of-magnitude p-values.
        from statistics import NormalDist
        p = 2.0 * NormalDist().cdf(-abs(t))
    return float(t), float(p)


def load_variant(vkey, subdir):
    p = ROOT / subdir / "metrics" / "base_vs_lca_results.csv"
    df = pd.read_csv(p)
    df = df[df["channel"].isin(["clean"] + PERT_CHANNELS)].copy()
    base = df[df["model"] == "base"][["sample_id", "L", "channel", "mel_l1"]].rename(columns={"mel_l1": "base_mel"})
    lca = df[df["model"] == "lca"][["sample_id", "L", "channel", "mel_l1"]].rename(columns={"mel_l1": "lca_mel"})
    wide = base.merge(lca, on=["sample_id", "L", "channel"], how="inner")
    clean = wide[wide["channel"] == "clean"][["sample_id", "L", "base_mel", "lca_mel"]].rename(
        columns={"base_mel": "base_clean_mel", "lca_mel": "lca_clean_mel"}
    )
    pert = wide[wide["channel"].isin(PERT_CHANNELS)].merge(clean, on=["sample_id", "L"], how="inner")
    pert["base_degradation"] = pert["base_mel"] - pert["base_clean_mel"]
    pert["lca_degradation"] = pert["lca_mel"] - pert["lca_clean_mel"]
    pert["rob_imp"] = pert["base_degradation"] - pert["lca_degradation"]
    pert["variant"] = vkey
    return pert[["variant", "sample_id", "L", "channel", "base_degradation", "lca_degradation", "rob_imp"]]


all_df = pd.concat([load_variant(k, v) for k, v in VARIANTS.items()], ignore_index=True)
all_df.to_csv(OUT / "factorial_per_sample_robust_imp.csv", index=False)

summary_rows = []
for variant, g in all_df.groupby("variant"):
    mean, lo, hi, se = ci95(g["rob_imp"])
    cell_means = g.groupby(["L", "channel"])["rob_imp"].mean()
    summary_rows.append({
        "variant": variant,
        "n_pairs": len(g),
        "n_samples": g["sample_id"].nunique(),
        "n_cells": cell_means.shape[0],
        "mean_rob_imp": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "se": se,
        "pos_cells": int((cell_means > 0).sum()),
    })
summary = pd.DataFrame(summary_rows).sort_values("variant")
summary.to_csv(OUT / "factorial_variant_summary.csv", index=False)

wide = all_df.pivot_table(index=["sample_id", "L", "channel"], columns="variant", values="rob_imp", aggfunc="first").reset_index()
pairs = [
    ("V3_minus_V1", "V3", "V1", "ChannelSim added on top of random-L"),
    ("V4_minus_V3", "V4", "V3", "Consistency added on top of random-L + ChannelSim"),
    ("V3_minus_V2", "V3", "V2", "random-L added on top of ChannelSim"),
    ("V2_minus_V1", "V2", "V1", "ChannelSim-only vs random-L-only configuration contrast"),
    ("V4_minus_V0", "V4", "V0", "End-to-end full LCA vs full-depth clean control"),
]
rows = []
cell_rows = []
for name, a, b, desc in pairs:
    diff = wide[a] - wide[b]
    mean, lo, hi, se = ci95(diff)
    t, p = ttest_1samp_zero(diff)
    sig_cells = 0
    pos_cells = 0
    for (L, channel), cg in wide.groupby(["L", "channel"]):
        cdiff = cg[a] - cg[b]
        cmean, clo, chi, cse = ci95(cdiff)
        ct, cp = ttest_1samp_zero(cdiff)
        pos = cmean > 0
        sig = (not math.isnan(cp)) and cp < 0.05
        pos_cells += int(pos)
        sig_cells += int(sig and pos)
        cell_rows.append({
            "comparison": name,
            "L": L,
            "channel": channel,
            "n": len(cdiff),
            "mean_diff": cmean,
            "ci_lo": clo,
            "ci_hi": chi,
            "t_stat": ct,
            "p_value": cp,
            "positive": pos,
            "significant_positive_p_lt_0_05": bool(sig and pos),
        })
    rows.append({
        "comparison": name,
        "description": desc,
        "n_pairs": len(diff),
        "mean_diff": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "se": se,
        "t_stat": t,
        "p_value": p,
        "positive_cells": pos_cells,
        "significant_positive_cells_p_lt_0_05": sig_cells,
    })

pairwise = pd.DataFrame(rows)
pairwise.to_csv(OUT / "factorial_pairwise_tests.csv", index=False)
pd.DataFrame(cell_rows).to_csv(OUT / "factorial_pairwise_cell_tests.csv", index=False)


def fmt_p(p):
    if pd.isna(p):
        return "nan"
    if p == 0:
        return "<1e-300"
    return f"{p:.3e}"

with open(OUT / "factorial_stats_summary.md", "w", encoding="utf-8") as f:
    f.write("# E3 V0–V4 Factorial Statistical Tests (n=256)\n\n")
    f.write(f"Input root: `{ROOT}`\n\n")
    f.write("## Variant means\n\n")
    f.write("| Variant | n pairs | n samples | mean rob_imp | 95% CI | positive cells / 12 |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for _, r in summary.iterrows():
        f.write(f"| {r.variant} | {int(r.n_pairs)} | {int(r.n_samples)} | {r.mean_rob_imp:.6f} | [{r.ci_lo:.6f}, {r.ci_hi:.6f}] | {int(r.pos_cells)} |\n")
    f.write("\n## Pairwise pooled tests\n\n")
    f.write("| Comparison | Description | n pairs | mean diff | 95% CI | t | p | positive cells | significant positive cells |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in pairwise.iterrows():
        f.write(f"| {r.comparison} | {r.description} | {int(r.n_pairs)} | {r.mean_diff:.6f} | [{r.ci_lo:.6f}, {r.ci_hi:.6f}] | {r.t_stat:.2f} | {fmt_p(r.p_value)} | {int(r.positive_cells)} | {int(r.significant_positive_cells_p_lt_0_05)} |\n")
    f.write("\n## Notes\n\n")
    f.write("- Robustness improvement is `Base degradation - Variant degradation`; positive is better.\n")
    f.write("- Degradation is perturbed mel-L1 minus clean mel-L1 at matched `(sample_id, L)`.\n")
    f.write("- Pooled tests use 12 perturbation cells × 256 samples = 3072 paired values.\n")
    f.write("- Cell-level tests are saved to `factorial_pairwise_cell_tests.csv`.\n")

print("Wrote", OUT)
print(pairwise.to_string(index=False))
