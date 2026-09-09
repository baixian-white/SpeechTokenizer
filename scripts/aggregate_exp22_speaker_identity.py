"""Aggregate Exp22 speaker-identity runs across random seeds."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import ensure_run_layout, write_csv, write_text


SEED_RE = re.compile(r"_seed(\d+)")

PROBE_METRICS = ("top1_accuracy", "top5_accuracy", "macro_f1")
IDENTITY_METRICS = (
    "top1_accuracy",
    "verified_rate",
    "correct_score_mean",
    "max_impostor_score_mean",
    "margin_mean",
    "eer",
    "tar_at_far",
)

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def parse_seed(path: Path) -> int | None:
    for part in [path.name, *[parent.name for parent in path.parents]]:
        match = SEED_RE.search(part)
        if match:
            return int(match.group(1))
    return None


def parse_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def format_number(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def ci95_from_std(std_value: float | None, n: int) -> float | None:
    if std_value is None or n < 2:
        return None
    t_value = T_CRITICAL_95.get(n - 1, 1.96)
    return float(t_value * std_value / math.sqrt(n))


def read_run_rows(paths: Sequence[Path], kind: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(paths):
        seed = parse_seed(path)
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["seed"] = seed
                row["run_dir"] = str(path.parents[1])
                row["summary_path"] = str(path)
                row["kind"] = kind
                rows.append(row)
    return rows


def glob_paths(patterns: Sequence[str]) -> List[Path]:
    paths = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern))
    return sorted({path.resolve() for path in paths if path.exists()})


def aggregate_rows(rows: Iterable[dict], group_keys: Sequence[str], metrics: Sequence[str]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in group_keys)
        grouped[key].append(row)

    output = []
    for key, items in sorted(grouped.items()):
        out = {name: value for name, value in zip(group_keys, key)}
        seeds = sorted({item.get("seed") for item in items if item.get("seed") is not None})
        out["n_seeds"] = len(seeds)
        out["seeds"] = " ".join(str(seed) for seed in seeds)
        out["run_count"] = len(items)
        for meta_key in ("feature_dim", "speaker_count", "test_count", "n"):
            values = [str(item.get(meta_key, "")).strip() for item in items if str(item.get(meta_key, "")).strip()]
            if values and all(value == values[0] for value in values):
                out[meta_key] = values[0]

        for metric in metrics:
            values = [parse_float(item.get(metric)) for item in items]
            clean = [value for value in values if value is not None]
            out[f"{metric}_mean"] = mean(clean) if clean else None
            out[f"{metric}_std"] = stdev(clean) if len(clean) >= 2 else None
            out[f"{metric}_ci95"] = ci95_from_std(out[f"{metric}_std"], len(clean))
        output.append(out)
    return output


def write_aggregate_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    clean_rows = []
    for row in rows:
        clean_rows.append({field: format_number(row.get(field)) for field in fields})
    write_csv(path, clean_rows, fields)


def metric_cell(row: dict, metric: str) -> str:
    avg = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if avg is None:
        return ""
    if std is None:
        return f"{avg:.3f}"
    return f"{avg:.3f} +/- {std:.3f}"


def write_markdown_report(path: Path, probe_rows: Sequence[dict], identity_rows: Sequence[dict]) -> None:
    lines = [
        "# Exp22 Paper-Grade Speaker Identity Aggregate",
        "",
        "This report aggregates completed Exp22 speaker identity runs across seeds.",
        "",
    ]
    if probe_rows:
        lines.extend(
            [
                "## Codes/Latent Speaker Probe",
                "",
                "| model | feature | L | seeds | top1 | top5 | macro-F1 |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in probe_rows:
            lines.append(
                "| {model} | {feature_kind} | {L} | {n_seeds} | {top1} | {top5} | {macro_f1} |".format(
                    model=row.get("model", ""),
                    feature_kind=row.get("feature_kind", ""),
                    L=row.get("L", ""),
                    n_seeds=row.get("n_seeds", ""),
                    top1=metric_cell(row, "top1_accuracy"),
                    top5=metric_cell(row, "top5_accuracy"),
                    macro_f1=metric_cell(row, "macro_f1"),
                )
            )
        lines.append("")
    if identity_rows:
        lines.extend(
            [
                "## ECAPA Speaker Preservation",
                "",
                "| model | L | seeds | top1 | verified | EER | TAR@FAR=0.01 | margin |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in identity_rows:
            lines.append(
                "| {model} | {L} | {n_seeds} | {top1} | {verified} | {eer} | {tar} | {margin} |".format(
                    model=row.get("model", ""),
                    L=row.get("L", ""),
                    n_seeds=row.get("n_seeds", ""),
                    top1=metric_cell(row, "top1_accuracy"),
                    verified=metric_cell(row, "verified_rate"),
                    eer=metric_cell(row, "eer"),
                    tar=metric_cell(row, "tar_at_far"),
                    margin=metric_cell(row, "margin_mean"),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- Probe metrics answer whether speaker identity is linearly recoverable from SCIT codes/features.",
            "- ECAPA preservation metrics answer whether decoded waveforms remain recognizable as the same speaker.",
            "- Table cells show mean +/- std when at least two seeds are available; single-seed cells show the raw value.",
            "- CSV files additionally include a small-sample 95% t-interval half-width for each metric.",
            "- MFCC preservation runs are retained as dependency-light sanity checks, not paper-grade speaker verification.",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--probe-glob",
        action="append",
        default=[],
        help="Glob to speaker_probe_summary.csv files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--identity-glob",
        action="append",
        default=[],
        help="Glob to speaker_identity_summary.csv files. Can be passed multiple times.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    run_dir = ensure_run_layout(args.run_dir)

    probe_patterns = args.probe_glob or [
        str(PROJECT_ROOT / "output/experiments/exp22_speaker_probe_vctk110_20260708_seed*/metrics/speaker_probe_summary.csv")
    ]
    identity_patterns = args.identity_glob or [
        str(PROJECT_ROOT / "output/experiments/exp22_speaker_identity_ecapa_vctk110_20260708_seed*/metrics/speaker_identity_summary.csv")
    ]

    probe_source = read_run_rows(glob_paths(probe_patterns), kind="probe")
    identity_source = read_run_rows(glob_paths(identity_patterns), kind="identity")
    probe_rows = aggregate_rows(probe_source, ("model", "feature_kind", "L"), PROBE_METRICS)
    identity_rows = aggregate_rows(identity_source, ("model", "L"), IDENTITY_METRICS)

    probe_fields = [
        "model",
        "feature_kind",
        "L",
        "feature_dim",
        "speaker_count",
        "test_count",
        "n_seeds",
        "seeds",
        "run_count",
    ] + [f"{metric}_{suffix}" for metric in PROBE_METRICS for suffix in ("mean", "std", "ci95")]
    identity_fields = [
        "model",
        "L",
        "n",
        "n_seeds",
        "seeds",
        "run_count",
    ] + [f"{metric}_{suffix}" for metric in IDENTITY_METRICS for suffix in ("mean", "std", "ci95")]

    write_csv(run_dir / "metrics" / "speaker_probe_seed_runs.csv", probe_source, sorted(probe_source[0].keys()) if probe_source else ["kind"])
    write_csv(
        run_dir / "metrics" / "speaker_identity_seed_runs.csv",
        identity_source,
        sorted(identity_source[0].keys()) if identity_source else ["kind"],
    )
    write_aggregate_csv(run_dir / "metrics" / "speaker_probe_multiseed_summary.csv", probe_rows, probe_fields)
    write_aggregate_csv(run_dir / "metrics" / "speaker_identity_ecapa_multiseed_summary.csv", identity_rows, identity_fields)
    write_markdown_report(run_dir / "reports" / "exp22_paper_grade_summary.md", probe_rows, identity_rows)
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
