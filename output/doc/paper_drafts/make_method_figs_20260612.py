"""Generate figure 4 (perturbed ASR WER, Base vs LCA) and figure 5 (V0-V4 cumulative
robustness with 95% CI) for the SCIT-Speech method paper.

Data sources:
- fig4: output/experiments/exp11_perturbed_asr_wer_20260609/.../perturbed_asr_wer_results.csv
        (test-clean_100 + test-other_100), aggregated to per (split, L, condition) WER
- fig5: output/experiments/exp5_lca_component_factorial_20260603_seed42/reports/
        statistical_tests_20260610/factorial_variant_summary.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path("output/doc/paper_drafts/scit_speech_method_cn_draft_20260609论文素材")
OUT.mkdir(parents=True, exist_ok=True)

# Use a CJK-capable font if available, else fall back silently (labels are mostly ASCII).
for cand in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------------- Figure 4: perturbed ASR WER, Base vs LCA ----------------
TC = pd.read_csv("output/experiments/exp11_perturbed_asr_wer_20260609/eval_perturbed_asr_wer/test-clean_100/metrics/perturbed_asr_wer_results.csv")
TO = pd.read_csv("output/experiments/exp11_perturbed_asr_wer_20260609/eval_perturbed_asr_wer/test-other_100/metrics/perturbed_asr_wer_results.csv")
TC["split"] = "test-clean_100"
TO["split"] = "test-other_100"
df = pd.concat([TC, TO], ignore_index=True)

# condition name normalization
cond_map = {"dropout-high": "高度前帧覆盖", "substitution-high": "高度随机替换"}
df = df[df["channel"].isin(cond_map)].copy()
agg = df.groupby(["split", "L", "channel", "model"])["wer_vs_gt"].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
conds = ["dropout-high", "substitution-high"]
splits = ["test-clean_100", "test-other_100"]
Ls = [1, 2, 3]
group_labels = [f"L{L}\n{cond_map[c]}" for c in conds for L in Ls]
x = np.arange(len(group_labels))
width = 0.38

for ax, split in zip(axes, splits):
    base_vals, lca_vals = [], []
    for c in conds:
        for L in Ls:
            b = agg[(agg.split == split) & (agg.L == L) & (agg.channel == c) & (agg.model == "base")]["wer_vs_gt"]
            l = agg[(agg.split == split) & (agg.L == L) & (agg.channel == c) & (agg.model == "lca")]["wer_vs_gt"]
            base_vals.append(float(b.iloc[0]) if len(b) else np.nan)
            lca_vals.append(float(l.iloc[0]) if len(l) else np.nan)
    ax.bar(x - width / 2, base_vals, width, label="Base", color="#c0c4cc", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, lca_vals, width, label="LCA", color="#e8743b", edgecolor="black", linewidth=0.5)
    ax.set_title(split)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    # annotate WER drop (pp) where LCA improves
    for xi, (bv, lv) in enumerate(zip(base_vals, lca_vals)):
        if not np.isnan(bv) and not np.isnan(lv) and lv < bv:
            ax.annotate(f"-{(bv-lv)*100:.1f}", (xi, max(bv, lv) + 0.01), ha="center", fontsize=7, color="#1a7f37")
axes[0].set_ylabel("WER (↓)")
axes[0].legend(loc="upper right", fontsize=9)
fig.suptitle("图 4：扰动条件下 Base 与 LCA 的 Whisper WER（数字为 LCA 相对 Base 的 WER 下降，单位百分点）", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fig4_perturbed_asr_wer.png", dpi=200)
plt.close(fig)
print("wrote fig4_perturbed_asr_wer.png")

# ---------------- Figure 5: V0-V4 cumulative robustness with 95% CI ----------------
vs = pd.read_csv("output/experiments/exp5_lca_component_factorial_20260603_seed42/reports/statistical_tests_20260610/factorial_variant_summary.csv")
order = ["V0", "V1", "V2", "V3", "V4"]
vs = vs.set_index("variant").loc[order].reset_index()
labels = ["V0\n全深度\n对照", "V1\n+random-L", "V2\n仅ChannelSim", "V3\nrandom-L\n+ChannelSim", "V4\n完整LCA\n(+一致性)"]
means = vs["mean_rob_imp"].values
lo = vs["ci_lo"].values
hi = vs["ci_hi"].values
yerr = np.vstack([means - lo, hi - means])

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#c0c4cc", "#c0c4cc", "#7fb3d5", "#5499c7", "#e8743b"]
bars = ax.bar(np.arange(len(order)), means, yerr=yerr, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("mel 鲁棒性改进 (Base 退化 − LCA 退化, ↑)")
ax.set_title("图 5：LCA 三组件对扰动鲁棒性的累积贡献（n=256，误差棒为 95% CI）", fontsize=10)
ax.grid(axis="y", alpha=0.3)
for i, m in enumerate(means):
    ax.annotate(f"{m:+.4f}", (i, hi[i] + 0.0003), ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig5_v0v4_cumulative_robustness.png", dpi=200)
plt.close(fig)
print("wrote fig5_v0v4_cumulative_robustness.png")
