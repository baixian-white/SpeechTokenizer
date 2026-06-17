"""Figure 3: Perturbation-robustness curves on LibriSpeech test-other (full, n=2939).

Aggregates exp8 (dropout/substitution mid/high) and exp9 (packet-loss 1p/3p/5p,
burst-loss 2f/5f/10f) plus the shared clean baseline, then plots:

  rows: mel-L1 (lower better) and STOI (higher better)
  cols: dropout, substitution, packet loss, burst loss
  per cell: Base vs LCA, L in {1,2,3}, x-axis = perturbation strength.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP8 = ROOT / "output" / "experiments" / "exp8_librispeech_test_full_perturb_20260606" / "eval_perturb_full_lca_nosave"
EXP9 = ROOT / "output" / "experiments" / "exp9_librispeech_test_full_packet_burst_20260608" / "eval_packet_burst_full_lca_nosave"
OUT = ROOT / "output" / "doc" / "paper_drafts" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
MATERIALS = ROOT / "output" / "doc" / "scit_speech_method_cn_draft_20260609论文素材"

DATASET = "test-other"

PERTURB_AXES = {
    "dropout": dict(
        title="Dropout (independent)",
        clean_label="0%",
        levels=[("clean", "0%"), ("dropout-mid", "mid"), ("dropout-high", "high")],
        xtick_pos=[0, 1, 2],
        source="exp8",
    ),
    "substitution": dict(
        title="Substitution (independent)",
        clean_label="0%",
        levels=[("clean", "0%"), ("substitution-mid", "mid"), ("substitution-high", "high")],
        xtick_pos=[0, 1, 2],
        source="exp8",
    ),
    "packet": dict(
        title="Packet loss",
        clean_label="0%",
        levels=[("clean", "0%"), ("packet-loss-1p", "1%"), ("packet-loss-3p", "3%"), ("packet-loss-5p", "5%")],
        xtick_pos=[0, 1, 2, 3],
        source="exp9",
    ),
    "burst": dict(
        title="Burst loss",
        clean_label="0",
        levels=[("clean", "0"), ("burst-2f", "2f"), ("burst-5f", "5f"), ("burst-10f", "10f")],
        xtick_pos=[0, 1, 2, 3],
        source="exp9",
    ),
}

L_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"}
MODEL_STYLE = {
    "base": dict(linestyle="--", marker="o", lw=1.6, ms=6, alpha=0.85),
    "lca":  dict(linestyle="-",  marker="*", lw=2.0, ms=10, alpha=1.0),
}


def load(source: str, dataset: str) -> pd.DataFrame:
    if source == "exp8":
        p = EXP8 / dataset / "metrics" / "base_vs_lca_perturb_results.csv"
    else:
        p = EXP9 / dataset / "metrics" / "packet_burst_results.csv"
    return pd.read_csv(p, low_memory=False)


def aggregate(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    sub = df[df["channel"].isin(channels)].copy()
    g = sub.groupby(["model", "L", "channel"])[["mel_l1", "stoi"]].mean().reset_index()
    return g


def plot():
    df8 = load("exp8", DATASET)
    df9 = load("exp9", DATASET)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.0), sharey="row")
    metrics = [("mel_l1", "mel-L1 (lower is better)"), ("stoi", "STOI (higher is better)")]

    for col, (pkey, cfg) in enumerate(PERTURB_AXES.items()):
        df = df8 if cfg["source"] == "exp8" else df9
        channels = [lvl for lvl, _ in cfg["levels"]]
        agg = aggregate(df, channels)

        # build a position lookup from channel to xtick
        ch_to_x = {lvl: pos for (lvl, _), pos in zip(cfg["levels"], cfg["xtick_pos"])}

        for row, (metric_col, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for L in [1, 2, 3]:
                for model in ["base", "lca"]:
                    rows = agg[(agg["model"] == model) & (agg["L"] == L)].copy()
                    if rows.empty:
                        continue
                    rows["x"] = rows["channel"].map(ch_to_x)
                    rows = rows.sort_values("x")
                    style = MODEL_STYLE[model]
                    label = f"{'LCA' if model == 'lca' else 'Base'}  L={L}"
                    ax.plot(
                        rows["x"], rows[metric_col],
                        color=L_COLORS[L],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        markersize=style["ms"],
                        linewidth=style["lw"],
                        alpha=style["alpha"],
                        label=label if (row == 0 and col == 0) else None,
                        markeredgecolor="black" if model == "lca" else None,
                        markeredgewidth=0.5 if model == "lca" else 0,
                    )
            ax.set_xticks(cfg["xtick_pos"])
            ax.set_xticklabels([lab for _, lab in cfg["levels"]], fontsize=9)
            ax.grid(True, ls="--", alpha=0.35)
            if row == 0:
                ax.set_title(cfg["title"], fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row == 1:
                ax.set_xlabel("perturbation strength", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=6, frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=10,
    )

    fig.suptitle(
        f"Figure 3. Perturbation robustness on LibriSpeech {DATASET} (n=2939). "
        "LCA (solid) vs Base (dashed) at L ∈ {1,2,3}.",
        fontsize=12, y=0.995,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.13, wspace=0.18, hspace=0.30)

    png = OUT / "fig3_perturbation_robustness.png"
    pdf = OUT / "fig3_perturbation_robustness.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}\nSaved: {pdf}")

    if MATERIALS.exists():
        for fname in (png.name, pdf.name):
            target = MATERIALS / fname
            target.write_bytes((OUT / fname).read_bytes())
            print(f"Synced: {target}")


if __name__ == "__main__":
    plot()
