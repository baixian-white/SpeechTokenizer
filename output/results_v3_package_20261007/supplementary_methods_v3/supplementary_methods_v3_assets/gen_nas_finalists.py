"""Plot search-time proxy quality and encoder cost for the eight final NAS candidates.

Run from any directory with ``py gen_nas_finalists.py``. The Stage-4
measurements are not metrics of the subsequently trained SCIT codec.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "output" / "experiments" / "exp1_nas_distill_run1_seed42"
SOURCE = RUN / "metrics" / "stage4_final.csv"
BEST = RUN / "artifacts" / "best_architecture" / "best_candidate_raw.json"
OUTPUT = HERE / "nas_finalists.png"

INK = "#202629"
MUTED = "#556068"
OTHER = "#5A6A73"
SELECTED = "#0072B2"


def load_candidates():
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["stage"] == "stage4_final"
            and row["encoder_type"] == "nas-candidate"
            and row["status"] == "completed"
        ]
    assert len(rows) == 8, f"Expected eight completed Stage-4 NAS candidates, got {len(rows)}"
    assert len({row["candidate_id"] for row in rows}) == 8

    fields = (
        "stage_rank", "stage_score", "encoder_macs_g", "encoder_params",
        "proxy_mel_loss", "teacher_latent_smooth_l1",
    )
    for row in rows:
        for key in fields:
            row[key] = float(row[key])
            assert math.isfinite(row[key]), (row["candidate_id"], key)
    assert sorted(int(row["stage_rank"]) for row in rows) == list(range(1, 9))

    selected_id = json.loads(BEST.read_text(encoding="utf-8"))["candidate_id"]
    selected = next(row for row in rows if row["candidate_id"] == selected_id)
    assert selected["stage_rank"] == 1
    assert selected["stage_score"] == min(row["stage_score"] for row in rows)
    return rows, selected


def draw():
    rows, selected = load_candidates()
    others = [row for row in rows if row is not selected]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.1))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.81, wspace=0.32)

    panels = [
        (
            axes[0], "a  Reconstruction proxy", "encoder_macs_g",
            "Encoder MACs (G / 1-s input)", "proxy_mel_loss", "Proxy mel-L1",
            (0.517, 0.634), (1.465, 1.685), (-87, 31),
        ),
        (
            axes[1], "b  Teacher alignment proxy", "encoder_params",
            "Encoder parameters (millions)", "teacher_latent_smooth_l1",
            "Latent SmoothL1", (8.38, 8.99), (1.86, 2.195), (-90, 30),
        ),
    ]
    for ax, title, xkey, xlabel, ykey, ylabel, xlim, ylim, offset in panels:
        factor = 1e-6 if xkey == "encoder_params" else 1.0
        ax.scatter(
            [row[xkey] * factor for row in others],
            [row[ykey] for row in others],
            marker="o", s=46, facecolor="white", edgecolor=OTHER,
            linewidth=1.2, zorder=3,
        )
        x, y = selected[xkey] * factor, selected[ykey]
        ax.scatter(
            [x], [y], marker="*", s=140, facecolor=SELECTED,
            edgecolor=INK, linewidth=0.45, zorder=5,
        )
        ax.annotate(
            "selected\n000896", xy=(x, y), xytext=offset,
            textcoords="offset points", ha="left", va="center",
            fontsize=7.4, color=SELECTED,
            arrowprops={"arrowstyle": "-", "color": SELECTED, "lw": 0.65},
        )
        ax.set(
            title=title, xlabel=xlabel, ylabel=ylabel,
            xlim=xlim, ylim=ylim,
        )
        ax.title.set_ha("left")
        ax.title.set_x(0)
        ax.tick_params(axis="both", direction="out", length=3, colors=INK)
        ax.grid(color="#D7DEE1", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        ax.text(
            0.98, 0.97, "lower is better", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.2, color=MUTED,
        )

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white",
                   markeredgecolor=OTHER, markeredgewidth=1.2,
                   markersize=6.5, label="Other Stage-4 candidates"),
            Line2D([0], [0], marker="*", linestyle="", markerfacecolor=SELECTED,
                   markeredgecolor=INK, markeredgewidth=0.45,
                   markersize=10, label="Selected by combined score"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2,
        frameon=False, fontsize=8.2, handletextpad=0.45,
    )
    fig.savefig(OUTPUT, dpi=300, facecolor="white")
    plt.close(fig)
    print(
        f"{OUTPUT} (8 Stage-4 NAS candidates; selected {selected['candidate_id']}; "
        f"score {selected['stage_score']:.5f})"
    )


if __name__ == "__main__":
    draw()
