"""Build Results v2 figures and tables from archived measurements."""

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results_v2_assets"
EXP = ROOT / "output" / "experiments"
SPEAKER = Path("D:/tmp/speechtokenizer-exp23/tmp/exp23_formal")
NAS = EXP / "exp1_nas_distill_run1_seed42" / "metrics" / "stage1_profile.csv"
QUALITY = (
    EXP / "exp12_baseline_comparison_test300_20260610_seed42"
    / "runs/test-clean_300/metrics/per_method_summary.csv"
)
INDEX = (
    EXP / "exp8_librispeech_test_full_perturb_20260606"
    / "eval_perturb_full_lca_nosave/test-clean/metrics/base_vs_lca_perturb_results.csv"
)
PACKET = (
    EXP / "exp9_librispeech_test_full_packet_burst_20260608"
    / "eval_packet_burst_full_lca_nosave/test-clean/metrics/packet_burst_results.csv"
)
METRICS = ("mel_l1", "stoi", "pesq_wb")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tex(name, content):
    (OUT / name).write_text(content.strip() + "\n", encoding="utf-8")


def build_nas():
    rows = read_csv(NAS)
    reference = next(row for row in rows if row["candidate_id"] == "hand_encoder")
    selected = next(row for row in rows if row["candidate_id"] == "nas_seed42_000896")
    candidates = [
        row for row in rows
        if row["encoder_type"] == "nas-candidate" and row["status"] == "completed"
    ]
    assert len(candidates) == 8192
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.55),
                             gridspec_kw={"width_ratios": [1.04, 1]})
    fig.subplots_adjust(left=0.082, right=0.985, bottom=0.20, top=0.85, wspace=0.36)
    ax = axes[0]
    ax.scatter(
        [float(row["encoder_params"]) / 1e6 for row in candidates],
        [float(row["encoder_macs"]) / 1e9 for row in candidates],
        s=6, c="#858D93", alpha=0.12, linewidths=0, rasterized=True,
        label="8,192 configuration candidates",
    )
    ax.scatter(float(reference["encoder_params"]) / 1e6,
               float(reference["encoder_macs"]) / 1e9,
               s=65, c="#20282D", marker="^", zorder=5)
    ax.scatter(float(selected["encoder_params"]) / 1e6,
               float(selected["encoder_macs"]) / 1e9,
               s=95, c="#D55E00", marker="*", edgecolors="#20282D",
               linewidths=0.5, zorder=6)
    ax.annotate("Reference", (float(reference["encoder_params"]) / 1e6,
                              float(reference["encoder_macs"]) / 1e9),
                xytext=(-9, 11), textcoords="offset points", ha="right", fontsize=8)
    ax.annotate("Selected", (float(selected["encoder_params"]) / 1e6,
                             float(selected["encoder_macs"]) / 1e9),
                xytext=(9, -17), textcoords="offset points", fontsize=8,
                color="#A34200")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7, 85)
    ax.set_ylim(0.40, 8)
    ax.xaxis.set_major_locator(FixedLocator([8, 16, 32, 64]))
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
    ax.minorticks_off()
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Encoder parameters (M, log scale)")
    ax.set_ylabel("Encoder MACs (G, log scale)")
    ax.set_title("a  Profiled configuration space", loc="left", pad=14, weight="bold")

    fields = ("encoder_params", "encoder_macs", "encoder_rtf_mean")
    fractions = [float(selected[key]) / float(reference[key]) for key in fields]
    ax = axes[1]
    positions = [2, 1, 0]
    ax.barh(positions, [1, 1, 1], color="#E5E8EA", height=0.54, label="Reference = 1")
    ax.barh(positions, fractions, color=["#0072B2", "#009E73", "#D55E00"],
            height=0.54)
    for y, value in zip(positions, fractions):
        ax.text(value + 0.035, y, f"{100 * (1 - value):.1f}% lower",
                va="center", fontsize=8, color="#20282D")
    ax.set_yticks(positions, ["Parameters", "MACs", "Encoder RTF"])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlim(0, 1.07)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Fraction of reference encoder cost")
    ax.set_title("b  Selected encoder", loc="left", pad=14, weight="bold")
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.20), frameon=False)
    fig.savefig(OUT / "nas_resources.pdf")
    fig.savefig(OUT / "nas_resources.png", dpi=300)
    plt.close(fig)
    return {
        "configuration_count": len(candidates),
        "reference": {key: float(reference[key]) for key in fields},
        "selected": {key: float(selected[key]) for key in fields},
        "selected_candidate": selected["candidate_id"],
        "reduction_percent": dict(zip(fields, [100 * (1 - x) for x in fractions])),
    }


def build_quality():
    rows = read_csv(QUALITY)
    specifications = [
        ("scit_base", "L=3", r"SCIT", "1.5"),
        ("dac", "n_q_3", r"DAC ($n_q=3$)", "1.5"),
        ("encodec", "bw1.5kbps_n_cb2", "EnCodec", "1.5"),
        ("opus", "opus_6000bps", "Opus", "6.0"),
        ("pcm", "16bit_16khz_passthrough", "PCM", "256"),
    ]
    lines = []
    evidence = []
    for method, setting, label, rate in specifications:
        row = next(r for r in rows if r["method"] == method and r["codec_setting"] == setting)
        assert int(row["n_samples"]) == 300
        values = []
        for metric in METRICS:
            mean, lower, upper = (float(row[f"{metric}_{key}"])
                                  for key in ("mean", "ci_low", "ci_high"))
            rendered = f"{mean:.3f}"
            if method == "scit_base":
                rendered = r"\textbf{" + rendered + "}"
            if method == "opus" and metric == "mel_l1":
                rendered += r"$^\dagger$"
            values.append(f"{rendered} [{lower:.3f}, {upper:.3f}]")
        lines.append(label + " & " + rate + " & " + " & ".join(values) + r"\\")
        evidence.append(row)
    content = r"""
\begin{table*}[t]
\centering
\caption{Clean-channel reconstruction on 300 LibriSpeech test-clean utterances. Entries are means [95\% utterance-bootstrap intervals]. The first three codecs share the nominal 1.5-kbit/s operating point; bold means identify the best in that group. Opus and PCM are higher-rate references. Lower mel-L1 and higher STOI/PESQ-WB are better.}
\label{tab:rv2-quality}
\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrr}
\toprule
Codec & kbit/s & mel-L1 & STOI & PESQ-WB\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\parbox{\textwidth}{\footnotesize $^\dagger$Full-band mel-L1 is sensitive to low-rate Opus bandwidth reduction and is not a standalone cross-codec perceptual ranking. Intervals use 10,000 resamples and seed 42; displayed values are rounded to three decimals.}
\end{table*}
"""
    write_tex("quality_table.tex", content)
    return evidence


def aggregate_conditions(path, conditions):
    grouped = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["L"] == "3" and row["channel"] in conditions:
                grouped[row["model"], row["channel"]].append(row)
    result = {}
    for (model, channel), rows in grouped.items():
        assert len(rows) == len({row["sample_id"] for row in rows}) == 2620
        means = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            assert all(math.isfinite(value) for value in values)
            means[metric] = math.fsum(values) / len(values)
        result[model, channel] = means
    for channel in conditions:
        pairs = [
            {row["sample_id"]: row["channel_seed"] for row in grouped[model, channel]}
            for model in ("base", "lca")
        ]
        assert pairs[0] == pairs[1]
    return result


def build_robustness():
    index = aggregate_conditions(INDEX, {"clean", "dropout-high", "substitution-high"})
    packet = aggregate_conditions(PACKET, {"packet-loss-5p", "burst-10f"})
    all_means = index | packet
    specifications = [
        ("clean", "Clean"),
        ("dropout-high", r"10\% index erasure"),
        ("substitution-high", r"3\% index substitution"),
        ("packet-loss-5p", r"5\% packet loss (100 ms)"),
        ("burst-10f", "One 200-ms burst"),
    ]
    lines = []
    evidence = []
    for channel, label in specifications:
        base, lca = all_means["base", channel], all_means["lca", channel]
        values = []
        for metric in METRICS:
            for value in (base[metric], lca[metric]):
                best = min(base[metric], lca[metric]) if metric == "mel_l1" else max(base[metric], lca[metric])
                rendered = f"{value:.4f}"
                values.append(r"\textbf{" + rendered + "}" if value == best else rendered)
        lines.append(label + " & " + " & ".join(values) + r"\\")
        evidence.append({
            "condition": channel, "n": 2620, "base": base, "lca": lca,
            "delta": {metric: lca[metric] - base[metric] for metric in METRICS},
        })
    content = r"""
\begin{table*}[t]
\centering
\caption{SCIT and SCIT-LCA on all 2,620 LibriSpeech test-clean utterances at $L=3$. Means are computed from paired utterances with matched perturbation seeds. Bold entries indicate the favourable direction within each condition, not statistical significance. Erasure and packet/burst masks use the previous original index, with the first frame protected.}
\label{tab:rv2-robustness}
\small\setlength{\tabcolsep}{6pt}
\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{2}{c}{mel-L1 $\downarrow$} & \multicolumn{2}{c}{STOI $\uparrow$} & \multicolumn{2}{c}{PESQ-WB $\uparrow$}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Condition & SCIT & SCIT-LCA & SCIT & SCIT-LCA & SCIT & SCIT-LCA\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\parbox{\textwidth}{\footnotesize The burst condition removes one randomly placed span of ten latent frames per utterance; it is not a fixed percentage of packet loss. These token-domain perturbations do not emulate a complete packet-loss concealment receiver.}
\end{table*}
"""
    write_tex("robustness_table.tex", content)
    return evidence


def build_speaker():
    bootstrap = json.loads((SPEAKER / "aggregate/bootstrap_summary.json").read_text(encoding="utf-8"))
    intervals = {row["seed"]: row for row in bootstrap["bootstrap_by_seed"]}
    metrics = {seed: json.loads((SPEAKER / f"seed{seed}/metrics/test_metrics.json").read_text(encoding="utf-8"))
               for seed in (41, 42, 43)}
    keys = ("top1", "macro_f1", "top5", "ece_15bin", "worst_decile_accuracy")
    lines = []
    for seed, values in metrics.items():
        assert len(values["per_speaker_accuracy"]) == 110
        interval = intervals[seed]
        assert abs(interval["point_estimate"] - values["top1"]) < 1e-12
        label = str(seed) + (" (delivery)" if seed == 42 else "")
        cells = [f"{100 * values[key]:.2f}" for key in keys]
        cells.append(f"[{100 * interval['ci_low']:.2f}, {100 * interval['ci_high']:.2f}]")
        lines.append(label + " & " + " & ".join(cells) + r"\\")
    mean_cells = []
    for key in keys:
        values = [100 * metrics[seed][key] for seed in metrics]
        rendered = f"{statistics.mean(values):.2f}"
        if key in ("top1", "macro_f1"):
            rendered += r" $\pm$ " + f"{statistics.stdev(values):.2f}"
        mean_cells.append(r"\textbf{" + rendered + "}")
    lines.append(r"\textbf{Mean}" + " & " + " & ".join(mean_cells) + r" & --\\")
    content = r"""
\begin{table*}[t]
\centering
\caption{Closed-set speaker identification from a frozen SCIT-LCA payload ($L=3$, 110 speakers, 1,650 test utterance groups). All entries are percentages. Mean $\pm$ SD uses the sample standard deviation across three training seeds. Worst 10\% is the mean accuracy of the 11 lowest-accuracy speakers. The last column reports within-seed hierarchical-bootstrap intervals, not uncertainty on the three-seed mean.}
\label{tab:rv2-speaker}
\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrr}
\toprule
Seed & Top-1 & Macro-F1 & Top-5 & ECE & Worst 10\% & Top-1 95\% CI\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\parbox{\textwidth}{\footnotesize Seed 42 was predeclared for delivery. Intervals are retained from the archived analysis; its bootstrap replicate count and RNG seed remain an author-confirmation item in Methods and Supplementary Methods S7.}
\end{table*}
"""
    write_tex("speaker_table.tex", content)
    return {"test_metrics": metrics, "bootstrap": bootstrap}


def main():
    OUT.mkdir(exist_ok=True)
    protected = [
        HERE / "manuscript.tex", HERE / "manuscript.pdf",
        HERE / "results_zh.md", HERE / "methods_main_zh.md",
        HERE / "supplementary_methods_body.tex",
    ]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    evidence = {
        "generated_from_archived_records": True,
        "nas": build_nas(), "quality": build_quality(),
        "robustness": build_robustness(), "speaker": build_speaker(),
        "sources": {str(path): digest(path) for path in (NAS, QUALITY, INDEX, PACKET)},
        "protected_original_hashes": hashes,
    }
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "nas_candidates": evidence["nas"]["configuration_count"],
        "nas_reduction_percent": evidence["nas"]["reduction_percent"],
        "quality_rows": len(evidence["quality"]),
        "robustness_conditions": len(evidence["robustness"]),
        "matched_utterances_per_condition": 2620,
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
