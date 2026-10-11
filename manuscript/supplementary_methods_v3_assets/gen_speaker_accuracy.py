#!/usr/bin/env python3
"""Plot formal exp23 per-speaker test accuracy for three classifier seeds."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS_ROOT = Path(__file__).resolve().parent / "speaker_formal_metrics"
SEEDS = (41, 42, 43)
N_SPEAKERS = 110
TEST_GROUPS_PER_SPEAKER = 15
BOTTOM_DECILE_SIZE = 11
COLORS = ("#0072B2", "#D55E00", "#009E73")  # Okabe-Ito


def load_scores(root: Path) -> list[tuple[int, np.ndarray, float, float]]:
    runs = []
    expected_ids = None
    for seed in SEEDS:
        path = root / f"seed{seed}" / "metrics" / "test_metrics.json"
        with path.open(encoding="utf-8") as source:
            record = json.load(source)
        scores = record["per_speaker_accuracy"]
        ids = set(scores)
        if len(ids) != N_SPEAKERS or (expected_ids is not None and ids != expected_ids):
            raise ValueError(f"{path}: expected the same {N_SPEAKERS} speakers in every seed")
        expected_ids = ids
        values = np.asarray([scores[key] for key in sorted(ids, key=int)], dtype=float)
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{path}: invalid per-speaker accuracy")
        correct_counts = values * TEST_GROUPS_PER_SPEAKER
        if not np.allclose(correct_counts, np.rint(correct_counts), atol=1e-10):
            raise ValueError(f"{path}: accuracy is not a multiple of 1/15")
        ranked = np.sort(values)
        top1 = float(record["top1"])
        worst = float(record["worst_decile_accuracy"])
        if not np.isclose(values.mean(), top1, atol=1e-10):
            raise ValueError(f"{path}: per-speaker mean disagrees with saved Top-1")
        if not np.isclose(ranked[:BOTTOM_DECILE_SIZE].mean(), worst, atol=1e-10):
            raise ValueError(f"{path}: lowest-11 mean disagrees with saved worst decile")
        runs.append((seed, ranked * 100, top1 * 100, worst * 100))
    return runs


def plot(runs: list[tuple[int, np.ndarray, float, float]], output: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(3, 1, figsize=(6.75, 3.65), sharex=True, sharey=True)
    ranks = np.arange(1, N_SPEAKERS + 1)
    for ax, (seed, accuracies, top1, worst), color in zip(axes, runs, COLORS):
        ax.axvspan(0.5, BOTTOM_DECILE_SIZE + 0.5, color="#F1F1F1", zorder=0)
        ax.axvline(BOTTOM_DECILE_SIZE + 0.5, color="#888888", lw=0.65, ls=":", zorder=1)
        ax.plot(ranks, accuracies, color=color, lw=1.1, drawstyle="steps-mid", zorder=2)
        ax.scatter(ranks[BOTTOM_DECILE_SIZE:], accuracies[BOTTOM_DECILE_SIZE:],
                   s=8, facecolors="white", edgecolors=color, linewidths=0.6, zorder=3)
        ax.scatter(ranks[:BOTTOM_DECILE_SIZE], accuracies[:BOTTOM_DECILE_SIZE],
                   s=13, color=color, linewidths=0, zorder=4)
        ax.text(0.985, 0.1, f"Seed {seed}  |  Top-1 {top1:.2f}%  |  lowest 11 {worst:.2f}%",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#303030")
        ax.set_ylim(45, 103)
        ax.set_yticks([50, 75, 100])
        ax.grid(axis="y", color="#D7D7D7", alpha=0.75, lw=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888888")
        ax.spines[["left", "bottom"]].set_linewidth(0.65)
        ax.tick_params(axis="both", width=0.65, length=2.5, color="#666666")

    axes[0].text(6, 99, "Lowest 11", ha="center", va="top", fontsize=7.5,
                 color="#535353")
    axes[-1].set_xlim(0.5, N_SPEAKERS + 0.5)
    axes[-1].set_xticks([1, 11, 25, 50, 75, 100, 110])
    axes[-1].set_xlabel("Speaker rank within seed (ascending accuracy)")
    fig.supylabel("Per-speaker Top-1 accuracy (%)", x=0.025, fontsize=9)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.985, bottom=0.15, hspace=0.16)

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "speaker_accuracy.png", dpi=300, metadata={
        "Software": "gen_speaker_accuracy.py",
    })
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    runs = load_scores(args.metrics_root)
    plot(runs, args.output)
    for seed, ranked, top1, worst in runs:
        print(f"Seed {seed}: 110 speakers, 15 test groups/speaker; "
              f"Top-1 {top1:.2f}%, lowest 11 mean {worst:.2f}%, "
              f"range {ranked[0]:.2f}-{ranked[-1]:.2f}%")


if __name__ == "__main__":
    main()
