from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path("H:/H-CODE/speechtokenizer")
CSV = (
    ROOT
    / "output/experiments/exp5_lca_component_factorial_20260603_seed42/reports/statistical_tests_20260610/factorial_variant_summary.csv"
)
OUT = (
    ROOT
    / "output/doc/paper_drafts/scit_speech_method_cn_draft_20260609论文素材/fig5_v0v4_cumulative_robustness.png"
)


for cand in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False

vs = pd.read_csv(CSV)
order = ["V0", "V1", "V2", "V3", "V4"]
vs = vs.set_index("variant").loc[order].reset_index()

labels = [
    "V0\n全深度\n对照",
    "V1\n+random-L",
    "V2\n仅ChannelSim",
    "V3\nrandom-L\n+ChannelSim",
    "V4\n完整LCA\n(+一致性)",
]
means = vs["mean_rob_imp"].to_numpy()
lo = vs["ci_lo"].to_numpy()
hi = vs["ci_hi"].to_numpy()
yerr = np.vstack([means - lo, hi - means])

fig, ax = plt.subplots(figsize=(8, 4.25))
colors = ["#c0c4cc", "#c0c4cc", "#7fb3d5", "#5499c7", "#e8743b"]
ax.bar(np.arange(len(order)), means, yerr=yerr, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("mel 鲁棒性改进 (Base 退化 − LCA 退化, ↑)")
ax.set_ylim(min(lo) - 0.0005, max(hi) + 0.00075)
ax.grid(axis="y", alpha=0.3)
for i, m in enumerate(means):
    ax.annotate(f"{m:+.4f}", (i, hi[i] + 0.0003), ha="center", fontsize=8)

fig.tight_layout()
fig.savefig(OUT, dpi=200)
plt.close(fig)
print(OUT)
