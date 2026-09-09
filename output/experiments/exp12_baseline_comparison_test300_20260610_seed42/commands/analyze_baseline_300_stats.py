"""Aggregate per-method audio-quality metrics and run paired pairwise tests
for the exp12 baseline comparison on test-clean_300 / test-other_300.

For each split this script reads
  runs/<split>/metrics/audio_quality_results.csv
and writes
  runs/<split>/metrics/per_method_summary.csv
  runs/<split>/metrics/lca_vs_baselines_pairwise.csv
  runs/<split>/reports/baseline_300_summary.md

Pairwise comparisons (SCIT-Speech-LCA vs baselines), paired by sample_id:
  500  bps : scit_lca L=1 vs dac n_q_1
  1000 bps : scit_lca L=2 vs dac n_q_2
  1500 bps : scit_lca L=3 vs dac n_q_3
  1500 bps : scit_lca L=3 vs encodec bw1.5kbps_n_cb2
  cross    : scit_lca L=3 vs opus 6 kbps   (rate-mismatched, flagged 'cross-rate')

Bootstrap: 10000 resamples, seed=42, paired by sample_id.
Wilcoxon: scipy.stats.wilcoxon when available; otherwise a normal-approximation
fallback using z = T / sqrt(n*(n+1)*(2n+1)/6) where T is the signed-rank sum
(zero-difference samples are dropped, as in the standard Wilcoxon procedure).

Verified from
runs/test-clean_300_smoke/metrics/audio_quality_results.csv:
  DAC      codec_setting = 'n_q_<int>'                        (e.g. n_q_3)
  EnCodec  codec_setting = 'bw<bw>kbps_n_cb<int>'             (e.g. bw1.5kbps_n_cb2)
  SCIT     codec_setting = 'L=<int>'                          (e.g. L=3)
  PCM      codec_setting = '16bit_16khz_passthrough'
The user-described 'encodec_<bw>' format is NOT what the runner emits; we use
the verified format above.

Run:
  python analyze_baseline_300_stats.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_stats = None
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

EXPERIMENT_ROOT = Path(
    r"h:/H-CODE/speechtokenizer/output/experiments/"
    r"exp12_baseline_comparison_test300_20260610_seed42"
).resolve()

SPLITS = ("test-clean_300", "test-other_300")

METRICS: Tuple[str, ...] = (
    "wave_l1",
    "mel_l1",
    "si_snr_db",
    "corr",
    "stoi",
    "pesq_wb",
)

# Higher-is-better convention used only for human-readable summaries.
HIGHER_IS_BETTER: Dict[str, bool] = {
    "wave_l1": False,
    "mel_l1": False,
    "si_snr_db": True,
    "corr": True,
    "stoi": True,
    "pesq_wb": True,
}

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
CI_LOW_Q = 2.5
CI_HIGH_Q = 97.5


# ---------------------------------------------------------------------------
# Bootstrap & test helpers
# ---------------------------------------------------------------------------

def bootstrap_ci_mean(
    values: np.ndarray,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using a non-parametric bootstrap."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if values.size == 1:
        v = float(values[0])
        return (v, v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [CI_LOW_Q, CI_HIGH_Q])
    return (float(values.mean()), float(lo), float(hi))


def bootstrap_ci_paired_diff(
    diffs: np.ndarray,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the mean of paired differences."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if diffs.size == 1:
        v = float(diffs[0])
        return (v, v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [CI_LOW_Q, CI_HIGH_Q])
    return (float(diffs.mean()), float(lo), float(hi))


def wilcoxon_signed_rank(diffs: np.ndarray) -> Dict[str, object]:
    """Return Wilcoxon signed-rank statistic and (approximate) p-value.

    Drops zero differences (standard Wilcoxon procedure). When scipy is
    available we use ``scipy.stats.wilcoxon``; otherwise we fall back to a
    normal-approximation using
        z = T / sqrt(n*(n+1)*(2n+1)/6)
    with T the signed-rank sum (mid-ranks for ties), and report a two-sided
    p-value via the standard-normal survival function approximation.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    nonzero = diffs[diffs != 0.0]
    n = int(nonzero.size)
    out: Dict[str, object] = {
        "n_used": n,
        "n_zero_diffs": int(diffs.size - n),
        "method": "n/a",
        "statistic": float("nan"),
        "p_value": float("nan"),
    }
    if n == 0:
        out["method"] = "skipped_all_zero"
        return out
    if _HAVE_SCIPY:
        try:
            res = _scipy_stats.wilcoxon(
                nonzero, zero_method="wilcox", alternative="two-sided"
            )
            out["method"] = "scipy.stats.wilcoxon"
            out["statistic"] = float(res.statistic)
            out["p_value"] = float(res.pvalue)
            return out
        except Exception as exc:  # pragma: no cover
            out["method"] = f"scipy_failed:{exc!r}_fallback_normal_approx"
    else:
        out["method"] = "normal_approx"

    abs_vals = np.abs(nonzero)
    # Mid-ranks for ties
    order = np.argsort(abs_vals, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sorted_abs = abs_vals[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    signs = np.sign(nonzero)
    signed_sum = float(np.sum(signs * ranks))  # T
    var = n * (n + 1) * (2 * n + 1) / 6.0
    z = signed_sum / math.sqrt(var) if var > 0 else 0.0
    # Two-sided p-value via standard-normal survival approximation
    p = math.erfc(abs(z) / math.sqrt(2.0))
    out["statistic"] = signed_sum
    out["p_value"] = float(p)
    if out["method"] == "n/a":
        out["method"] = "normal_approx"
    return out


# ---------------------------------------------------------------------------
# Per-method aggregation
# ---------------------------------------------------------------------------

def compute_per_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean + 95% bootstrap CI per (method, codec_setting) for each metric."""
    rows: List[Dict[str, object]] = []
    grouped = df.groupby(["method", "codec_setting"], sort=True, dropna=False)
    for (method, codec_setting), sub in grouped:
        row: Dict[str, object] = {
            "method": method,
            "codec_setting": codec_setting,
            "n_samples": int(len(sub)),
        }
        for metric in METRICS:
            if metric not in sub.columns:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_ci_low"] = float("nan")
                row[f"{metric}_ci_high"] = float("nan")
                continue
            mean, lo, hi = bootstrap_ci_mean(sub[metric].to_numpy(dtype=float))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pairwise comparisons
# ---------------------------------------------------------------------------

def _select(df: pd.DataFrame, method: str, codec_setting: str) -> pd.DataFrame:
    sub = df[(df["method"] == method) & (df["codec_setting"] == codec_setting)]
    if sub.empty:
        return sub
    # Defend against duplicate sample_ids within a single (method, codec_setting)
    sub = sub.drop_duplicates(subset=["sample_id"], keep="first")
    return sub


def compare_pair(
    df: pd.DataFrame,
    a_method: str,
    a_setting: str,
    b_method: str,
    b_setting: str,
    rate_label: str,
    cross_rate: bool,
) -> Optional[Dict[str, object]]:
    """Paired comparison row for a single (a vs b) at a single rate label."""
    a = _select(df, a_method, a_setting)
    b = _select(df, b_method, b_setting)
    if a.empty or b.empty:
        return {
            "rate_label": rate_label,
            "comparison": f"{a_method}/{a_setting} vs {b_method}/{b_setting}",
            "cross_rate": "yes" if cross_rate else "no",
            "n_samples": 0,
            "n_dropped_unmatched": int(a.shape[0] + b.shape[0]),
            "note": "missing_one_or_both_groups",
        }

    a_set = set(a["sample_id"].tolist())
    b_set = set(b["sample_id"].tolist())
    common = sorted(a_set & b_set)
    dropped = (len(a_set) + len(b_set)) - 2 * len(common)
    if not common:
        return {
            "rate_label": rate_label,
            "comparison": f"{a_method}/{a_setting} vs {b_method}/{b_setting}",
            "cross_rate": "yes" if cross_rate else "no",
            "n_samples": 0,
            "n_dropped_unmatched": int(dropped),
            "note": "no_paired_sample_ids",
        }

    a_p = a.set_index("sample_id").loc[common]
    b_p = b.set_index("sample_id").loc[common]

    out: Dict[str, object] = {
        "rate_label": rate_label,
        "comparison": f"{a_method}/{a_setting} vs {b_method}/{b_setting}",
        "cross_rate": "yes" if cross_rate else "no",
        "n_samples": int(len(common)),
        "n_dropped_unmatched": int(dropped),
        "note": "",
    }
    for metric in METRICS:
        if metric not in a_p.columns or metric not in b_p.columns:
            continue
        a_vals = a_p[metric].to_numpy(dtype=float)
        b_vals = b_p[metric].to_numpy(dtype=float)
        diffs = a_vals - b_vals  # A (LCA) minus B (baseline)
        diffs = diffs[~np.isnan(diffs)]
        if diffs.size == 0:
            continue
        mean_diff, lo, hi = bootstrap_ci_paired_diff(diffs)
        wil = wilcoxon_signed_rank(diffs)
        out[f"{metric}_mean_diff"] = mean_diff
        out[f"{metric}_ci_low"] = lo
        out[f"{metric}_ci_high"] = hi
        out[f"{metric}_wilcoxon_stat"] = wil["statistic"]
        out[f"{metric}_wilcoxon_p"] = wil["p_value"]
        out[f"{metric}_wilcoxon_method"] = wil["method"]
        out[f"{metric}_wilcoxon_n_used"] = wil["n_used"]
        out[f"{metric}_wilcoxon_n_zero"] = wil["n_zero_diffs"]
    return out


def find_opus_setting(df: pd.DataFrame, target_kbps: float = 6.0) -> Optional[str]:
    """Find an Opus codec_setting that corresponds to ``target_kbps``.

    The exp12 runner emits 'opus_<bitrate>bps' (e.g. 'opus_6000bps'). Earlier
    designs emitted 'opus_<kbps>kbps'. We accept both. The matching is done by
    parsing the numeric kbps/bps tokens INSIDE the string and comparing in
    canonical kbps units, because naive substring checks yield false positives
    (e.g. '6000bps' is a substring of 'opus_16000bps').
    """
    import re
    opus = df[df["method"] == "opus"]
    if opus.empty:
        return None
    settings = sorted(opus["codec_setting"].astype(str).unique().tolist())
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(kbps|bps)\b", re.IGNORECASE)
    # Pass 1: token-aware regex, canonicalize bps -> kbps before comparison.
    for s in settings:
        for num, unit in pattern.findall(s):
            try:
                v = float(num)
            except ValueError:
                continue
            if unit.lower() == "bps":
                v = v / 1000.0
            if abs(v - target_kbps) < 1e-6:
                return s
    # Pass 2: only if regex finds nothing, allow exact full-string substring with
    # a leading boundary (underscore or start-of-string) to avoid embedded matches.
    target_kbps_token = f"{int(target_kbps)}kbps" if target_kbps == int(target_kbps) else f"{target_kbps:g}kbps"
    target_bps_token = f"{int(round(target_kbps * 1000))}bps"
    boundary = re.compile(r"(?:^|[_\-])(" + re.escape(target_kbps_token) + r"|" + re.escape(target_bps_token) + r")(?:$|[_\-])", re.IGNORECASE)
    for s in settings:
        if boundary.search(s):
            return s
    return None


def build_pairwise_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build the LCA-vs-baselines pairwise table for one split."""
    rows: List[Dict[str, object]] = []
    notes: List[str] = []

    rows.append(compare_pair(
        df, "scit_lca", "L=1", "dac", "n_q_1",
        rate_label="500_bps", cross_rate=False,
    ))
    rows.append(compare_pair(
        df, "scit_lca", "L=2", "dac", "n_q_2",
        rate_label="1000_bps", cross_rate=False,
    ))
    rows.append(compare_pair(
        df, "scit_lca", "L=3", "dac", "n_q_3",
        rate_label="1500_bps", cross_rate=False,
    ))
    rows.append(compare_pair(
        df, "scit_lca", "L=3", "encodec", "bw1.5kbps_n_cb2",
        rate_label="1500_bps", cross_rate=False,
    ))

    opus_setting = find_opus_setting(df, target_kbps=6.0)
    if opus_setting is None:
        notes.append("Opus 6 kbps row not found in CSV; skipping cross-rate comparison.")
        rows.append({
            "rate_label": "cross-rate",
            "comparison": "scit_lca/L=3 vs opus/<6kbps not found>",
            "cross_rate": "yes",
            "n_samples": 0,
            "n_dropped_unmatched": 0,
            "note": "opus_6kbps_setting_missing",
        })
    else:
        rows.append(compare_pair(
            df, "scit_lca", "L=3", "opus", opus_setting,
            rate_label="cross-rate", cross_rate=True,
        ))

    rows = [r for r in rows if r is not None]
    table = pd.DataFrame(rows)
    return table, notes


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(v: object, digits: int = 4) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        return f"{v:.{digits}f}"
    return str(v)


def _summary_table_md(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "_(no rows)_\n"
    cols = ["method", "codec_setting", "n_samples"]
    for m in METRICS:
        cols.append(f"{m}_mean")
        cols.append(f"{m}_ci_low")
        cols.append(f"{m}_ci_high")
    cols = [c for c in cols if c in summary.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for _, row in summary.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in ("method", "codec_setting"):
                cells.append(str(v))
            elif c == "n_samples":
                cells.append(str(int(v)))
            else:
                cells.append(_fmt(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _pairwise_table_md(pairwise: pd.DataFrame) -> str:
    if pairwise.empty:
        return "_(no pairwise rows)_\n"
    blocks: List[str] = []
    for _, row in pairwise.iterrows():
        blocks.append(
            f"### {row.get('comparison', '?')}  "
            f"(rate: {row.get('rate_label', '?')}, "
            f"cross-rate: {row.get('cross_rate', '?')}, "
            f"n_paired={int(row.get('n_samples', 0))}, "
            f"n_dropped_unmatched={int(row.get('n_dropped_unmatched', 0))})"
        )
        if row.get("note"):
            blocks.append(f"_note_: `{row['note']}`")
        if int(row.get("n_samples", 0)) <= 0:
            blocks.append("")
            continue
        blocks.append(
            "| metric | mean_diff (LCA - baseline) | 95% CI | "
            "Wilcoxon stat | p (approx) | n_used | n_zero | method |"
        )
        blocks.append("|---|---|---|---|---|---|---|---|")
        for m in METRICS:
            md_key = f"{m}_mean_diff"
            if md_key not in pairwise.columns:
                continue
            blocks.append(
                f"| {m} | {_fmt(row.get(md_key))} | "
                f"[{_fmt(row.get(f'{m}_ci_low'))}, {_fmt(row.get(f'{m}_ci_high'))}] | "
                f"{_fmt(row.get(f'{m}_wilcoxon_stat'))} | "
                f"{_fmt(row.get(f'{m}_wilcoxon_p'), digits=4)} | "
                f"{_fmt(row.get(f'{m}_wilcoxon_n_used'), digits=0)} | "
                f"{_fmt(row.get(f'{m}_wilcoxon_n_zero'), digits=0)} | "
                f"{row.get(f'{m}_wilcoxon_method', '')} |"
            )
        blocks.append("")
    return "\n".join(blocks) + "\n"


def _interpret_pairwise(pairwise: pd.DataFrame) -> str:
    if pairwise.empty:
        return "No pairwise rows produced.\n"
    lines: List[str] = []
    for _, row in pairwise.iterrows():
        if int(row.get("n_samples", 0)) <= 0:
            lines.append(f"- {row['comparison']}: skipped ({row.get('note', '')}).")
            continue
        verdicts: List[str] = []
        for m in METRICS:
            md_key = f"{m}_mean_diff"
            if md_key not in pairwise.columns:
                continue
            md = row.get(md_key)
            lo = row.get(f"{m}_ci_low")
            hi = row.get(f"{m}_ci_high")
            p = row.get(f"{m}_wilcoxon_p")
            if isinstance(md, float) and math.isnan(md):
                continue
            ci_excludes_zero = (
                isinstance(lo, float) and isinstance(hi, float)
                and not math.isnan(lo) and not math.isnan(hi)
                and (lo > 0 or hi < 0)
            )
            sig = (isinstance(p, float) and not math.isnan(p) and p < 0.05)
            if HIGHER_IS_BETTER[m]:
                better_lca = md > 0
            else:
                better_lca = md < 0
            tag = "LCA better" if better_lca else "baseline better"
            mark = "**" if (ci_excludes_zero and sig) else ""
            verdicts.append(f"{mark}{m}: {tag} (Δ={_fmt(md)}, p={_fmt(p)}){mark}")
        lines.append(f"- {row['comparison']}: " + "; ".join(verdicts))
    return "\n".join(lines) + "\n"


def write_summary_md(
    split: str,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    notes: Sequence[str],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# Baseline 300-sample summary — `{split}`\n")
    lines.append(
        "Bootstrap: 10000 resamples, seed=42, 95% percentile CI. "
        "Pairwise tests use Wilcoxon signed-rank "
        f"({'scipy.stats' if _HAVE_SCIPY else 'normal-approx fallback'}). "
        "Differences reported as **LCA minus baseline**.\n"
    )
    if notes:
        lines.append("\n_Notes_:\n")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
    lines.append("\n## Per-method summary\n")
    lines.append(_summary_table_md(summary))
    lines.append("\n## LCA vs baselines (paired by sample_id)\n")
    lines.append(_pairwise_table_md(pairwise))
    lines.append("\n## Plain-language interpretation\n")
    lines.append(
        "Markers `**...**` flag metrics where the bootstrap 95% CI excludes "
        "zero AND Wilcoxon p < 0.05. Higher-is-better: si_snr_db, corr, stoi, "
        "pesq_wb. Lower-is-better: wave_l1, mel_l1.\n"
    )
    lines.append(_interpret_pairwise(pairwise))
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-split orchestration
# ---------------------------------------------------------------------------

def process_split(split: str) -> Optional[Dict[str, Path]]:
    runs_dir = EXPERIMENT_ROOT / "runs" / split
    csv_path = runs_dir / "metrics" / "audio_quality_results.csv"
    if not csv_path.exists():
        print(f"[skip] {split}: missing {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    required = {"method", "codec_setting", "sample_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    summary = compute_per_method_summary(df)
    pairwise, notes = build_pairwise_table(df)

    metrics_dir = runs_dir / "metrics"
    reports_dir = runs_dir / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = metrics_dir / "per_method_summary.csv"
    pairwise_csv = metrics_dir / "lca_vs_baselines_pairwise.csv"
    md_path = reports_dir / "baseline_300_summary.md"

    summary.to_csv(summary_csv, index=False)

    # Prepend a comment row that documents the Wilcoxon implementation path.
    comment_line = (
        "# wilcoxon_implementation: "
        f"{'scipy.stats.wilcoxon' if _HAVE_SCIPY else 'normal-approx fallback (z = T / sqrt(n*(n+1)*(2n+1)/6))'}"
        f"; bootstrap_resamples={BOOTSTRAP_RESAMPLES}; seed={BOOTSTRAP_SEED}; "
        "differences = LCA - baseline\n"
    )
    pairwise_csv_text = pairwise.to_csv(index=False)
    pairwise_csv.write_text(comment_line + pairwise_csv_text, encoding="utf-8")

    write_summary_md(split, summary, pairwise, notes, md_path)

    print(f"[ok] {split}: wrote\n  {summary_csv}\n  {pairwise_csv}\n  {md_path}")
    return {"summary": summary_csv, "pairwise": pairwise_csv, "report": md_path}


def main() -> None:
    print(f"Experiment root: {EXPERIMENT_ROOT}")
    print(f"scipy available: {_HAVE_SCIPY}")
    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    main()
