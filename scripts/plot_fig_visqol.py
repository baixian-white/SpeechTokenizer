"""Figure: ViSQOL MOS-LQO vs bitrate, test-clean_300 and test-other_300."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT  = Path(__file__).resolve().parents[1]
EXP21 = ROOT / "output/experiments/exp21_visqol_perceptual_test300_20260618_seed42"
EXP12 = ROOT / "output/experiments/exp12_baseline_comparison_test300_20260610_seed42"
EXP20 = ROOT / "output/experiments/exp20_amrwb_codec2_test300_20260614_seed42"
OUT   = ROOT / "output/doc/paper_drafts/assets"
OUT.mkdir(parents=True, exist_ok=True)

visqol = pd.read_csv(EXP21 / "metrics/visqol_per_method_summary.csv")


def load_payload(exp_root, split):
    p = pd.read_csv(exp_root / f"runs/{split}/metrics/payload_summary.csv")
    return (p.groupby(["method", "codec_setting"])["packed_payload_bps"]
              .mean().reset_index())


def get_bitrates(split):
    p12 = load_payload(EXP12, split)
    p20 = load_payload(EXP20, split)
    return pd.concat([p12, p20], ignore_index=True)


STYLE = {
    "scit_lca":  dict(color="#d62728", marker="*",  ms=14, lw=2.0, label="SCIT-Speech-LCA (ours)"),
    "scit_base": dict(color="#ff7f0e", marker="o",  ms=8,  lw=1.6, label="SCIT-Speech-Base (ours)"),
    "opus":      dict(color="#1f77b4", marker="s",  ms=8,  lw=1.6, label="Opus"),
    "dac":       dict(color="#2ca02c", marker="D",  ms=7,  lw=1.6, label="DAC"),
    "encodec":   dict(color="#9467bd", marker="^",  ms=8,  lw=1.6, label="EnCodec"),
    "codec2":    dict(color="#8c564b", marker="v",  ms=8,  lw=1.6, label="Codec2"),
    "amrwb":     dict(color="#7f7f7f", marker="x",  ms=8,  lw=1.2, label="AMR-WB"),
}
ORDER = ["codec2", "dac", "encodec", "scit_base", "scit_lca", "opus", "amrwb"]
SPLITS = [("test-clean_300", "test-clean"), ("test-other_300", "test-other")]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (split_key, split_label) in zip(axes, SPLITS):
    bitrates = get_bitrates(split_key)
    sub_v = visqol[visqol["split"] == split_key].copy()
    df = sub_v.merge(bitrates, on=["method", "codec_setting"], how="inner")
    df = df[df["packed_payload_bps"] <= 26000]

    for method in ORDER:
        s = STYLE.get(method)
        if s is None:
            continue
        m = df[df["method"] == method].sort_values("packed_payload_bps")
        if m.empty:
            continue
        ax.plot(m["packed_payload_bps"], m["visqol_mean"],
                color=s["color"], marker=s["marker"], markersize=s["ms"],
                linewidth=s["lw"], label=s["label"],
                markeredgecolor="black" if method == "scit_lca" else None,
                markeredgewidth=0.8 if method == "scit_lca" else 0)
        ax.fill_between(m["packed_payload_bps"],
                         m["visqol_ci_low"], m["visqol_ci_high"],
                         alpha=0.12, color=s["color"])

    ax.set_xscale("log")
    ax.set_xlabel("Bitrate (bps, log scale)")
    ax.set_ylabel("ViSQOL MOS-LQO (higher is better)")
    ax.set_title(split_label)
    ax.set_ylim(1.0, 4.7)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.set_xticks([500, 1000, 1500, 3000, 6000, 12000, 24000])
    ax.set_xticklabels(["500", "1k", "1.5k", "3k", "6k", "12k", "24k"])
    ax.axvspan(500, 1500, color="#fff3b0", alpha=0.45, zorder=0)
    ax.axhline(4.44 if "clean" in split_key else 4.34,
               color="gray", ls=":", lw=1.0, label="PCM upper bound")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=9)
fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.22, wspace=0.28)

for fmt in ("png", "pdf"):
    fig.savefig(OUT / f"fig_visqol.{fmt}", dpi=300, bbox_inches="tight")
print("Saved fig_visqol.png / .pdf")
