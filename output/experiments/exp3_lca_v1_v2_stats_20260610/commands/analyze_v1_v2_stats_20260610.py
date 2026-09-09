from pathlib import Path
import math
import numpy as np
import pandas as pd

V1 = Path(r"h:/H-CODE/speechtokenizer/output/experiments/exp3_low_load_channel_aware_adaptation_20260530_seed42/metrics/base_vs_lca_results.csv")
V2 = Path(r"h:/H-CODE/speechtokenizer/output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/metrics/base_vs_lca_results.csv")
OUT = Path(r"h:/H-CODE/speechtokenizer/output/experiments/exp3_lca_v1_v2_stats_20260610")
(OUT / "metrics").mkdir(parents=True, exist_ok=True)
(OUT / "reports").mkdir(parents=True, exist_ok=True)

METRICS = {
    "mel_l1": "lower",
    "pesq_wb": "higher",
    "stoi": "higher",
    "si_snr_db": "higher",
    "corr": "higher",
    "wave_l1": "lower",
}


def ci95(x):
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    if len(x) <= 1:
        return mean, float("nan"), float("nan"), float("nan")
    se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    return mean, mean - 1.96 * se, mean + 1.96 * se, se


def p_norm_from_t(t):
    from statistics import NormalDist
    return float(2.0 * NormalDist().cdf(-abs(t)))


def one_sample_test(x):
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
    if not sd or math.isnan(sd):
        return float("nan"), float("nan")
    t = mean / (sd / math.sqrt(len(x)))
    return t, p_norm_from_t(t)


def load(path, version):
    df = pd.read_csv(path)
    out_rows = []
    for metric, direction in METRICS.items():
        base = df[df.model == "base"][["sample_id", "L", "channel", metric]].rename(columns={metric: "base"})
        lca = df[df.model == "lca"][["sample_id", "L", "channel", metric]].rename(columns={metric: "lca"})
        wide = base.merge(lca, on=["sample_id", "L", "channel"], how="inner")
        clean = wide[wide.channel == "clean"][["sample_id", "L", "base", "lca"]].rename(columns={"base": "base_clean", "lca": "lca_clean"})
        pert = wide[wide.channel != "clean"].merge(clean, on=["sample_id", "L"], how="inner")
        # degradation should be positive when perturbed condition worsens.
        if direction == "lower":
            pert["base_degradation"] = pert["base"] - pert["base_clean"]
            pert["lca_degradation"] = pert["lca"] - pert["lca_clean"]
        else:
            pert["base_degradation"] = pert["base_clean"] - pert["base"]
            pert["lca_degradation"] = pert["lca_clean"] - pert["lca"]
        pert["rob_imp"] = pert["base_degradation"] - pert["lca_degradation"]
        pert["metric"] = metric
        pert["direction"] = direction
        pert["version"] = version
        out_rows.append(pert[["version", "metric", "direction", "sample_id", "L", "channel", "base_degradation", "lca_degradation", "rob_imp"]])
    return pd.concat(out_rows, ignore_index=True)

v1 = load(V1, "v1")
v2 = load(V2, "v2")
all_df = pd.concat([v1, v2], ignore_index=True)
all_df.to_csv(OUT / "metrics" / "v1_v2_per_sample_robust_imp.csv", index=False)

# Per-version descriptive summaries over each version's available perturbation cells.
summary_rows = []
for (version, metric), g in all_df.groupby(["version", "metric"]):
    mean, lo, hi, se = ci95(g.rob_imp)
    cell_means = g.groupby(["L", "channel"]).rob_imp.mean()
    summary_rows.append({
        "version": version,
        "metric": metric,
        "n_pairs": len(g),
        "n_samples": g.sample_id.nunique(),
        "n_cells": len(cell_means),
        "mean_rob_imp": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "positive_cells": int((cell_means > 0).sum()),
    })
summary = pd.DataFrame(summary_rows).sort_values(["metric", "version"])
summary.to_csv(OUT / "metrics" / "v1_v2_descriptive_summary.csv", index=False)

# Paired v2-v1 only on common channels: dropout-mid and substitution-mid.
common_channels = sorted(set(v1.channel.unique()).intersection(set(v2.channel.unique())))
common_channels = [c for c in common_channels if c != "clean"]
common = all_df[all_df.channel.isin(common_channels)].copy()
wide = common.pivot_table(index=["metric", "sample_id", "L", "channel"], columns="version", values="rob_imp", aggfunc="first").reset_index()
wide["v2_minus_v1"] = wide["v2"] - wide["v1"]
wide.to_csv(OUT / "metrics" / "v2_minus_v1_common_mid_per_sample.csv", index=False)

pair_rows = []
cell_rows = []
for metric, g in wide.groupby("metric"):
    diff = g.v2_minus_v1
    mean, lo, hi, se = ci95(diff)
    t, p = one_sample_test(diff)
    pos_cells = 0
    sig_cells = 0
    for (L, channel), cg in g.groupby(["L", "channel"]):
        cdiff = cg.v2_minus_v1
        cmean, clo, chi, cse = ci95(cdiff)
        ct, cp = one_sample_test(cdiff)
        pos = cmean > 0
        sig = (not math.isnan(cp)) and cp < 0.05
        pos_cells += int(pos)
        sig_cells += int(sig and pos)
        cell_rows.append({
            "metric": metric,
            "L": L,
            "channel": channel,
            "n": len(cdiff),
            "mean_v2_minus_v1": cmean,
            "ci_lo": clo,
            "ci_hi": chi,
            "t_stat": ct,
            "p_value_norm_approx": cp,
            "positive": pos,
            "significant_positive_p_lt_0_05": bool(sig and pos),
        })
    pair_rows.append({
        "metric": metric,
        "common_channels": ";".join(common_channels),
        "n_pairs": len(diff),
        "n_samples": g.sample_id.nunique(),
        "n_cells": g.groupby(["L", "channel"]).ngroups,
        "mean_v2_minus_v1": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "t_stat": t,
        "p_value_norm_approx": p,
        "positive_cells": pos_cells,
        "significant_positive_cells_p_lt_0_05": sig_cells,
    })
pair = pd.DataFrame(pair_rows).sort_values("metric")
pair.to_csv(OUT / "metrics" / "v2_minus_v1_common_mid_tests.csv", index=False)
pd.DataFrame(cell_rows).to_csv(OUT / "metrics" / "v2_minus_v1_common_mid_cell_tests.csv", index=False)


def fmt_p(p):
    if pd.isna(p):
        return "nan"
    if p == 0:
        return "<1e-300"
    return f"{p:.3e}"

with open(OUT / "reports" / "v1_v2_stats_summary.md", "w", encoding="utf-8") as f:
    f.write("# E4 LCA v1 vs v2 Statistical Post-processing\n\n")
    f.write(f"v1 input: `{V1}`\n\n")
    f.write(f"v2 input: `{V2}`\n\n")
    f.write("## Important design note\n\n")
    f.write("v1 and v2 do not share the same perturbation grid: v1 has dropout-low/dropout-mid/substitution-low/substitution-mid; v2 has dropout-mid/dropout-high/substitution-mid/substitution-high. Therefore this report separates (1) descriptive summaries over each version's own 12 perturbation cells and (2) paired tests only on common mid cells.\n\n")
    f.write("## Descriptive summaries over each version's own perturbation grid\n\n")
    f.write("| metric | version | n pairs | mean rob_imp | 95% CI | positive cells / n_cells |\n")
    f.write("|---|---|---:|---:|---:|---:|\n")
    for _, r in summary.iterrows():
        f.write(f"| {r.metric} | {r.version} | {int(r.n_pairs)} | {r.mean_rob_imp:.6f} | [{r.ci_lo:.6f}, {r.ci_hi:.6f}] | {int(r.positive_cells)}/{int(r.n_cells)} |\n")
    f.write("\n## Paired v2-v1 tests on common mid perturbation cells only\n\n")
    f.write("Common channels: `" + ", ".join(common_channels) + "` (3 L × 2 channels × 8 samples = 48 pairs per metric).\n\n")
    f.write("| metric | n pairs | mean v2-v1 | 95% CI | t | p (normal approx) | positive cells | significant positive cells |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in pair.iterrows():
        f.write(f"| {r.metric} | {int(r.n_pairs)} | {r.mean_v2_minus_v1:.6f} | [{r.ci_lo:.6f}, {r.ci_hi:.6f}] | {r.t_stat:.2f} | {fmt_p(r.p_value_norm_approx)} | {int(r.positive_cells)} | {int(r.significant_positive_cells_p_lt_0_05)} |\n")
    f.write("\n## Interpretation\n\n")
    f.write("- The descriptive table is the correct source for reproducing the existing v1/v2 narrative because each version was evaluated at its own perturbation grid.\n")
    f.write("- The paired table is stricter but only covers mid perturbation cells. Do not use it to claim behavior under v2 high perturbations.\n")
    f.write("- Positive `v2-v1` means v2 has larger robustness improvement than v1.\n")

print("Wrote", OUT)
print(pair.to_string(index=False))
