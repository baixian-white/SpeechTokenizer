"""Plot paired L=3 SCIT-LCA minus SCIT changes for the five Results conditions."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("paired_robustness.png")
METRICS = ("mel_l1", "stoi", "pesq_wb")
CONDITIONS = (
    (
        "Clean",
        "exp8_librispeech_test_full_perturb_20260606/"
        "eval_perturb_full_lca_nosave/test-clean/metrics/"
        "base_vs_lca_perturb_results.csv",
        "clean",
        ((0.9075, 0.8711), (0.8853, 0.8922), (1.9538, 1.9397)),
    ),
    (
        "10% erasure",
        "exp8_librispeech_test_full_perturb_20260606/"
        "eval_perturb_full_lca_nosave/test-clean/metrics/"
        "base_vs_lca_perturb_results.csv",
        "dropout-high",
        ((0.9986, 0.9511), (0.8457, 0.8580), (1.5364, 1.5549)),
    ),
    (
        "3% substitution",
        "exp8_librispeech_test_full_perturb_20260606/"
        "eval_perturb_full_lca_nosave/test-clean/metrics/"
        "base_vs_lca_perturb_results.csv",
        "substitution-high",
        ((0.9716, 0.9266), (0.8640, 0.8723), (1.7027, 1.7150)),
    ),
    (
        "5% packet (100 ms)",
        "exp9_librispeech_test_full_packet_burst_20260608/"
        "eval_packet_burst_full_lca_nosave/test-clean/metrics/"
        "packet_burst_results.csv",
        "packet-loss-5p",
        ((0.9336, 0.8959), (0.8727, 0.8805), (1.8096, 1.8067)),
    ),
    (
        "One 200-ms burst",
        "exp9_librispeech_test_full_packet_burst_20260608/"
        "eval_packet_burst_full_lca_nosave/test-clean/metrics/"
        "packet_burst_results.csv",
        "burst-10f",
        ((0.9271, 0.8902), (0.8752, 0.8826), (1.8674, 1.8584)),
    ),
)


def paired_differences():
    selected = {}
    for label, relative_path, channel, expected_means in CONDITIONS:
        path = ROOT / "output" / "experiments" / relative_path
        if path not in selected:
            selected[path] = {}
        selected[path][channel] = (label, expected_means)

    pairs = {label: {} for label, *_ in CONDITIONS}
    for path, channels in selected.items():
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                channel = row["channel"]
                if row["L"] != "3" or channel not in channels:
                    continue
                label, _ = channels[channel]
                key = (row["sample_id"], int(row["channel_seed"]))
                entry = pairs[label].setdefault(key, {})
                model = row["model"]
                if model not in ("base", "lca") or model in entry:
                    raise ValueError(f"Unexpected or duplicate model in {path}: {channel} {key} {model}")
                entry[model] = tuple(float(row[metric]) for metric in METRICS)

    differences = {metric: [] for metric in METRICS}
    reference_ids = None
    for label, _, _, expected_means in CONDITIONS:
        condition_pairs = pairs[label]
        if len(condition_pairs) != 2620:
            raise ValueError(f"{label}: expected 2620 pairs, got {len(condition_pairs)}")
        sample_ids = {sample_id for sample_id, _ in condition_pairs}
        if len(sample_ids) != 2620 or (reference_ids is not None and sample_ids != reference_ids):
            raise ValueError(f"{label}: missing or repeated utterance IDs")
        reference_ids = sample_ids
        base = []
        lca = []
        for key in sorted(condition_pairs):
            entry = condition_pairs[key]
            if set(entry) != {"base", "lca"}:
                raise ValueError(f"{label}: unmatched base/LCA pair for {key}")
            base.append(entry["base"])
            lca.append(entry["lca"])
        base = np.asarray(base, dtype=float)
        lca = np.asarray(lca, dtype=float)
        if not np.isfinite(base).all() or not np.isfinite(lca).all():
            raise ValueError(f"{label}: non-finite metric")
        for index, metric in enumerate(METRICS):
            observed = (base[:, index].mean(), lca[:, index].mean())
            if any(f"{actual:.4f}" != f"{expected:.4f}" for actual, expected in zip(observed, expected_means[index])):
                raise ValueError(
                    f"{label} {metric}: means {observed} disagree with Table 2 {expected_means[index]}"
                )
            delta = lca[:, index] - base[:, index]
            differences[metric].append(delta)
            print(
                f"{label:22s} {metric:7s} n={len(delta)} "
                f"base={observed[0]:.4f} lca={observed[1]:.4f} "
                f"delta={delta.mean():+.4f} "
                f"median={np.median(delta):+.4f} "
                f"p05={np.percentile(delta, 5):+.4f} "
                f"p95={np.percentile(delta, 95):+.4f}"
            )
    return differences


def draw(differences):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.65), sharey=True)
    positions = np.arange(len(CONDITIONS))
    for ax, metric, title in zip(
        axes,
        METRICS,
        ("mel-L1 (lower is better)", "STOI (higher is better)", "PESQ-WB (higher is better)"),
    ):
        values = differences[metric]
        central = [
            array[
                (array >= np.percentile(array, 1))
                & (array <= np.percentile(array, 99))
            ]
            for array in values
        ]
        violin = ax.violinplot(
            central,
            positions=positions,
            orientation="horizontal",
            widths=0.74,
            showextrema=False,
            points=120,
        )
        for body in violin["bodies"]:
            body.set_facecolor("#0072B2")
            body.set_edgecolor("#0072B2")
            body.set_alpha(0.28)
            body.set_linewidth(0.55)
        boxes = ax.boxplot(
            values,
            positions=positions,
            orientation="horizontal",
            widths=0.25,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            zorder=3,
        )
        for box in boxes["boxes"]:
            box.set(facecolor="white", edgecolor="#145374", linewidth=0.85)
        for line in boxes["whiskers"] + boxes["caps"]:
            line.set(color="#145374", linewidth=0.8)
        for line in boxes["medians"]:
            line.set(color="#173241", linewidth=1.3)
        ax.scatter(
            [array.mean() for array in values],
            positions,
            marker="D",
            s=18,
            color="#D55E00",
            edgecolors="white",
            linewidths=0.45,
            zorder=5,
        )
        ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--", zorder=1)
        ax.axhline(2.5, color="#bdbdbd", linewidth=0.6, linestyle=":", zorder=0)
        ax.set_title(title, pad=8)
        ax.set_xlabel("SCIT-LCA minus SCIT")
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#d6d6d6", linewidth=0.5, alpha=0.65)
        ax.set_axisbelow(True)
        ax.set_ylim(4.58, -0.58)
        lower = min(np.percentile(array, 1) for array in values)
        upper = max(np.percentile(array, 99) for array in values)
        padding = 0.07 * (upper - lower)
        ax.set_xlim(lower - padding, upper + padding)
    axes[0].set_yticks(positions, [label for label, *_ in CONDITIONS])
    axes[0].tick_params(axis="y", labelsize=7.6)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Paired utterance-level changes at L=3 (n=2,620 per condition)", fontsize=9.5, y=0.98)
    fig.legend(
        handles=[
            Patch(facecolor="#0072B2", alpha=0.28, label="Violin: central 98%"),
            Line2D([0], [0], color="#145374", linewidth=4, label="Box: IQR, 5th-95th whiskers"),
            Line2D([0], [0], marker="D", linestyle="", markersize=4.6, color="#D55E00", label="Full-sample mean"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=6.9,
        frameon=False,
    )
    fig.subplots_adjust(left=0.205, right=0.985, top=0.83, bottom=0.19, wspace=0.29)
    fig.savefig(OUTPUT, dpi=300)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    draw(paired_differences())
