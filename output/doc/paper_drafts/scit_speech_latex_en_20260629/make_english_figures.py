from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def add_box(ax, xy, w, h, text, fc="#f7f7f7", ec="#333333", fontsize=9):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.03",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return box


def add_arrow(ax, start, end, text=None, color="#333333", rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.035, text, ha="center", va="bottom", fontsize=8, color=color)


def save_system_overview() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.04, 0.60), 0.15, 0.16, "Speech\nwaveform $x$", fc="#eef5ff")
    add_box(ax, (0.25, 0.60), 0.16, 0.16, "Encoder\n$E_\\theta$", fc="#eef5ff")
    add_box(ax, (0.47, 0.60), 0.17, 0.16, "RVQ indices\n$I_{1:L}$", fc="#fff4dc")
    add_box(ax, (0.72, 0.60), 0.20, 0.16, "Shared decoder\n$D(C^*(I_{1:L}))$", fc="#eef5ff")
    add_box(ax, (0.79, 0.31), 0.13, 0.12, "Output\n$\\hat{x}$", fc="#eef5ff")

    add_arrow(ax, (0.19, 0.68), (0.25, 0.68))
    add_arrow(ax, (0.41, 0.68), (0.47, 0.68), "$L\\in\\{1,2,3\\}$")
    add_arrow(ax, (0.64, 0.68), (0.72, 0.68), "index packets")
    add_arrow(ax, (0.82, 0.60), (0.845, 0.43))

    add_box(ax, (0.45, 0.28), 0.21, 0.15, "Index-channel\nperturbations", fc="#ffecea", ec="#b64040")
    add_arrow(ax, (0.555, 0.60), (0.555, 0.43), "coverage / substitution /\npacket loss / burst loss", "#b64040")
    add_arrow(ax, (0.66, 0.355), (0.72, 0.62), color="#b64040", rad=-0.15)

    add_box(ax, (0.20, 0.20), 0.27, 0.16, "Offline shared knowledge\nRVQ codebooks $C^*$ + decoder $D$\nnot transmitted at runtime", fc="#edf8ed", ec="#3b7f3b", fontsize=8)
    add_arrow(ax, (0.335, 0.36), (0.54, 0.60), color="#3b7f3b", rad=-0.18)
    add_arrow(ax, (0.47, 0.28), (0.75, 0.60), color="#3b7f3b", rad=0.12)

    add_box(ax, (0.04, 0.16), 0.14, 0.14, "HuBERT\nteacher", fc="#eeeeee", ec="#777777", fontsize=8)
    add_arrow(ax, (0.18, 0.23), (0.37, 0.60), "training only", "#777777", rad=-0.18)

    ax.text(
        0.50,
        0.91,
        "SCIT-Speech: shared RVQ codebook index transmission at 500/1000/1500 bps",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )
    ax.text(
        0.50,
        0.08,
        "$K=1024$, $f_q=50$ Hz, $n_q=3$, and raw/net index payload $R(L)=L f_q\\lceil\\log_2K\\rceil=500L$ bps.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig1_system_overview.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_system_overview.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_perturbed_asr() -> None:
    base_dir = ROOT / "experiments" / "exp11_perturbed_asr_wer_20260609" / "eval_perturbed_asr_wer"
    clean = pd.read_csv(base_dir / "test-clean_100" / "metrics" / "perturbed_asr_wer_results.csv")
    other = pd.read_csv(base_dir / "test-other_100" / "metrics" / "perturbed_asr_wer_results.csv")
    clean["split"] = "test-clean"
    other["split"] = "test-other"
    df = pd.concat([clean, other], ignore_index=True)

    cond_names = {
        "dropout-high": "prev-frame\ncoverage",
        "substitution-high": "random\nsubstitution",
    }
    df = df[df["channel"].isin(cond_names)].copy()
    agg = df.groupby(["split", "L", "channel", "model"], as_index=False)["wer_vs_gt"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharey=True)
    channels = ["dropout-high", "substitution-high"]
    labels = [f"L={L}\n{cond_names[ch]}" for ch in channels for L in [1, 2, 3]]
    x = np.arange(len(labels))
    width = 0.36

    for ax, split in zip(axes, ["test-clean", "test-other"]):
        base_vals = []
        lca_vals = []
        for ch in channels:
            for L in [1, 2, 3]:
                base_v = agg[
                    (agg["split"] == split)
                    & (agg["L"] == L)
                    & (agg["channel"] == ch)
                    & (agg["model"] == "base")
                ]["wer_vs_gt"]
                lca_v = agg[
                    (agg["split"] == split)
                    & (agg["L"] == L)
                    & (agg["channel"] == ch)
                    & (agg["model"] == "lca")
                ]["wer_vs_gt"]
                base_vals.append(float(base_v.iloc[0]))
                lca_vals.append(float(lca_v.iloc[0]))

        ax.bar(x - width / 2, base_vals, width, label="Base", color="#b8bdc7", edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, lca_vals, width, label="LCA", color="#e36f3d", edgecolor="black", linewidth=0.5)
        for idx, (bv, lv) in enumerate(zip(base_vals, lca_vals)):
            delta = (lv - bv) * 100.0
            color = "#1b7f3a" if delta < 0 else "#8a1f1f"
            text = f"{delta:+.1f} pp"
            ax.text(idx, max(bv, lv) + 0.018, text, ha="center", va="bottom", fontsize=7, color=color)
        ax.set_title(split)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, max(base_vals + lca_vals) + 0.09)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Whisper WER (lower is better)")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_perturbed_asr_wer.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig4_perturbed_asr_wer.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_lca_ablation() -> None:
    path = (
        ROOT
        / "experiments"
        / "exp5_lca_component_factorial_20260603_seed42"
        / "reports"
        / "statistical_tests_20260610"
        / "factorial_variant_summary.csv"
    )
    df = pd.read_csv(path)
    order = ["V0", "V1", "V2", "V3", "V4"]
    df = df.set_index("variant").loc[order].reset_index()

    labels = [
        "V0\nfull-depth\ncontrol",
        "V1\n+random L",
        "V2\nChannelSim\nonly",
        "V3\nrandom L\n+ChannelSim",
        "V4\nfull LCA\n(+consistency)",
    ]
    means = df["mean_rob_imp"].to_numpy()
    lo = df["ci_lo"].to_numpy()
    hi = df["ci_hi"].to_numpy()
    yerr = np.vstack([means - lo, hi - means])

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = ["#c8cbd2", "#c8cbd2", "#86b8d9", "#4f96c6", "#e36f3d"]
    ax.bar(np.arange(len(order)), means, yerr=yerr, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mel robustness improvement (higher is better)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for i, (m, h) in enumerate(zip(means, hi)):
        ax.text(i, h + 0.00035, f"{m:+.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_lca_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig6_lca_ablation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    save_system_overview()
    save_perturbed_asr()
    save_lca_ablation()
