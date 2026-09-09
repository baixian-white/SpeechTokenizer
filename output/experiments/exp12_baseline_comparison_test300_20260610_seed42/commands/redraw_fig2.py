"""Redraw Figure 2 (rate-quality tradeoff, exp12 n=300) and build Appendix D
(full per-method table + paired Wilcoxon) for the SCIT-Speech method paper.

Replaces the old 8-sample fig2 (STOI/PESQ/WER) with a 2-row x 3-col layout:
rows = {test-clean_300, test-other_300}, cols = {mel-L1, STOI, PESQ-WB}.
x-axis = packed payload bitrate (bps, log scale).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path("H:/H-CODE/speechtokenizer/output/experiments/exp12_baseline_comparison_test300_20260610_seed42/runs")
OUT_FIG = Path("H:/H-CODE/speechtokenizer/output/doc/scit_speech_method_cn_draft_20260609论文素材")
OUT_FIG.mkdir(parents=True, exist_ok=True)

for cand in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False

SPLITS = ["test-clean_300", "test-other_300"]
METRICS = [("mel_l1_mean", "mel-L1 (↓)"), ("stoi_mean", "STOI (↑)"), ("pesq_wb_mean", "PESQ-WB (↑)")]

# style per method family
STYLE = {
    "scit_lca":  dict(label="SCIT-LCA (本文)", color="#d62728", marker="*", ms=13, zorder=6, lw=1.8),
    "scit_base": dict(label="SCIT-Base",       color="#ff7f0e", marker="o", ms=7,  zorder=5, lw=1.5),
    "dac":       dict(label="DAC",             color="#2ca02c", marker="D", ms=6,  zorder=4, lw=1.2),
    "encodec":   dict(label="EnCodec",         color="#9467bd", marker="^", ms=7,  zorder=4, lw=1.2),
    "opus":      dict(label="Opus",            color="#1f77b4", marker="s", ms=6,  zorder=3, lw=1.2),
    "pcm":       dict(label="PCM (无损)",      color="#7f7f7f", marker="x", ms=8,  zorder=2, lw=0),
}
ORDER = ["scit_lca", "scit_base", "dac", "encodec", "opus", "pcm"]


def load(split):
    m = ROOT / split / "metrics"
    pm = pd.read_csv(m / "per_method_summary.csv")
    pay = pd.read_csv(m / "payload_summary.csv")
    rate = pay.groupby(["method", "codec_setting"])["packed_payload_bps"].mean().reset_index()
    return pm.merge(rate, on=["method", "codec_setting"], how="left")


data = {s: load(s) for s in SPLITS}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for r, split in enumerate(SPLITS):
    df = data[split].sort_values("packed_payload_bps")
    for c, (col, ylabel) in enumerate(METRICS):
        ax = axes[r, c]
        ax.axvspan(490, 1530, color="#fff3bf", alpha=0.6, zorder=0)  # 500-1500 working region
        for fam in ORDER:
            sub = df[df["method"] == fam].sort_values("packed_payload_bps")
            if sub.empty:
                continue
            st = STYLE[fam]
            ax.plot(sub["packed_payload_bps"], sub[col],
                    color=st["color"], marker=st["marker"], ms=st["ms"],
                    lw=st["lw"], zorder=st["zorder"],
                    label=st["label"] if (r == 0 and c == 0) else None,
                    markeredgecolor="black", markeredgewidth=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("打包后实际码率 (bps, log)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(f"{split} — {ylabel.split(' ')[0]}", fontsize=10)
fig.legend(loc="upper center", ncol=6, fontsize=10, bbox_to_anchor=(0.5, 1.0), frameon=True)
fig.suptitle("图 2：码率—质量权衡（exp12，test-clean/other 各 n=300；黄色区为 500–1500 bps 工作区间）",
             fontsize=12, y=0.965)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT_FIG / "fig2_rate_quality_tradeoff.png", dpi=200)
fig.savefig(OUT_FIG / "fig2_rate_quality_tradeoff.pdf")
plt.close(fig)
print("wrote fig2_rate_quality_tradeoff.png/.pdf")
