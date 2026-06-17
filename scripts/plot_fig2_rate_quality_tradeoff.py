"""Figure 2: Bitrate vs quality trade-off across SCIT-Speech, Opus, DAC, EnCodec.

Reads exp4 audio_quality_results.csv + asr_results.csv + payload_summary.csv,
aggregates per (method, codec_setting) over the 8 fixed samples, and renders
a 1x3 panel (STOI, PESQ-WB, WER) with bitrate on log-x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "output" / "experiments" / "exp4_baseline_comparison_20260531_seed42"
OUT = ROOT / "output" / "doc" / "paper_drafts" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

aq = pd.read_csv(EXP / "metrics" / "audio_quality_results.csv")
asr = pd.read_csv(EXP / "metrics" / "asr_results.csv")
pay = pd.read_csv(EXP / "metrics" / "payload_summary.csv")

key = ["method", "codec_setting"]
aq_g = aq.groupby(key)[["stoi", "pesq_wb"]].mean().reset_index()
asr_g = asr.groupby(key)["wer_vs_gt"].mean().reset_index()
pay_g = pay.groupby(key)["packed_payload_bps"].mean().reset_index()

df = aq_g.merge(asr_g, on=key).merge(pay_g, on=key)
df = df[df["method"] != "pcm"].copy()
df = df[df["packed_payload_bps"] <= 26000].copy()

STYLE = {
    "scit_lca":  dict(color="#d62728", marker="*",  ms=14, lw=2.0, label="SCIT-Speech-LCA (ours)"),
    "scit_base": dict(color="#ff7f0e", marker="o",  ms=8,  lw=1.6, label="SCIT-Speech-Base (ours)"),
    "opus":      dict(color="#1f77b4", marker="s",  ms=8,  lw=1.6, label="Opus"),
    "dac":       dict(color="#2ca02c", marker="D",  ms=7,  lw=1.6, label="DAC"),
    "encodec":   dict(color="#9467bd", marker="^",  ms=8,  lw=1.6, label="EnCodec"),
}
ORDER = ["opus", "dac", "encodec", "scit_base", "scit_lca"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
metrics = [
    ("stoi",    "STOI (higher is better)",     None),
    ("pesq_wb", "PESQ-WB (higher is better)",  None),
    ("wer_vs_gt", "WER vs ground truth (lower is better)", None),
]

for ax, (col, title, _) in zip(axes, metrics):
    for method in ORDER:
        sub = df[df["method"] == method].sort_values("packed_payload_bps")
        if sub.empty:
            continue
        s = STYLE[method]
        ax.plot(
            sub["packed_payload_bps"], sub[col],
            color=s["color"], marker=s["marker"], markersize=s["ms"],
            linewidth=s["lw"], label=s["label"],
            markeredgecolor="black" if method == "scit_lca" else None,
            markeredgewidth=0.8 if method == "scit_lca" else 0,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Bitrate (bps, log scale)")
    ax.set_ylabel(title)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.set_xticks([500, 1000, 1500, 3000, 6000, 12000, 24000])
    ax.set_xticklabels(["500", "1k", "1.5k", "3k", "6k", "12k", "24k"])
    ax.axvspan(500, 1500, color="#fff3b0", alpha=0.45, zorder=0)

for ax in axes:
    ax.text(
        np.sqrt(500 * 1500), 0.99,
        "500–1500 bps",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=8, color="#8a6d00", style="italic",
    )

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center", ncol=5, frameon=False,
    bbox_to_anchor=(0.5, -0.01),
    fontsize=10,
)

fig.suptitle(
    "Figure 2. Bitrate vs quality trade-off on the fixed 8-sample baseline (exp4).",
    fontsize=12, y=0.995,
)

fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.20, wspace=0.28)

png = OUT / "fig2_rate_quality_tradeoff.png"
pdf = OUT / "fig2_rate_quality_tradeoff.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"Saved: {png}\nSaved: {pdf}")
