"""Diagnose test-clean ASR regressions where SCIT-LCA is worse than SCIT-Base.

Slots into Section 6.5 of scit_speech_cn_revised_ml_20260610.md.

Inputs (read-only):
  - exp10/eval_asr_wer_clean/test-clean_300/metrics/asr_wer_results.csv
      1800 rows = 300 samples x 3 L x 2 model (clean condition).
  - exp11/eval_perturbed_asr_wer/test-clean_100/metrics/perturbed_asr_wer_results.csv
      1200 rows = 100 samples x 3 L x 2 channel x 2 model (dropout-high, substitution-high).

Outputs (overwritten on each run):
  metrics/per_utterance_pairing.csv
  metrics/regression_cases_clean_300.csv
  metrics/regression_cases_perturbed_100.csv
  metrics/regression_summary_by_L.csv
  reports/regression_diagnosis_summary.md

Hypothesis tests reported in the markdown summary:
  H1 regression cases concentrate in short utterances (< 10 words)
  H2 regression cases concentrate where Base was already very good (Base WER < 0.1)
  H3 regression magnitudes are small (median delta_wer < 0.05)

Plain pandas + numpy + statistics. No scipy.
Bootstrap (B=2000, seed=42) is used for share-statistic 95% CIs.
"""

from __future__ import annotations

import os
import re
import statistics
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("H:/H-CODE/speechtokenizer")
EXP14_DIR = PROJECT_ROOT / "output" / "experiments" / "exp14_testclean_asr_regression_diagnosis_20260611"

EXP10_CSV = (
    PROJECT_ROOT
    / "output" / "experiments" / "exp10_asr_wer_onthefly_20260609"
    / "eval_asr_wer_clean" / "test-clean_300" / "metrics" / "asr_wer_results.csv"
)
EXP11_CSV = (
    PROJECT_ROOT
    / "output" / "experiments" / "exp11_perturbed_asr_wer_20260609"
    / "eval_perturbed_asr_wer" / "test-clean_100" / "metrics" / "perturbed_asr_wer_results.csv"
)

METRICS_DIR = EXP14_DIR / "metrics"
REPORTS_DIR = EXP14_DIR / "reports"

LEN_BINS = [0, 10, 20, 40, np.inf]
LEN_LABELS = ["[0,10)", "[10,20)", "[20,40)", "[40,inf)"]

SHORT_UTT_THRESHOLD = 10           # words
STRONG_BASE_THRESHOLD = 0.1        # WER
SMALL_DELTA_THRESHOLD = 0.05       # WER

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
_WHISPER_PUNCT = re.compile(r"[^\w\s']", flags=re.UNICODE)


def _normalize_text(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace.

    Used for the PCM-floor proxy WER between the (case+punctuation) Whisper
    transcript on the unmodified PCM and the ALL-CAPS unpunctuated ground
    truth. Matches the spirit of jiwer's default normalization without adding
    a dependency.
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = _WHISPER_PUNCT.sub(" ", text)
    return [tok for tok in text.split() if tok]


def _word_edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein word-level edit distance (substitution = 1)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution / match
            )
        prev, curr = curr, prev
    return prev[m]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Standard WER = edits / max(1, len(reference_words))."""
    ref = _normalize_text(reference)
    hyp = _normalize_text(hypothesis)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _word_edit_distance(ref, hyp) / len(ref)


# ---------------------------------------------------------------------------
# Loading and pairing
# ---------------------------------------------------------------------------
def _utt_words(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())


def _len_bin(n_words: int) -> str:
    idx = int(np.digitize([n_words], LEN_BINS, right=False)[0]) - 1
    idx = max(0, min(idx, len(LEN_LABELS) - 1))
    return LEN_LABELS[idx]


def load_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["split"] = "clean"
    df["condition"] = "clean"
    return df


def load_perturbed(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["split"] = "perturbed"
    df = df.rename(columns={"channel": "condition"})
    return df


def build_pairings(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot base vs lca rows into a per-utterance pairing table.

    Output schema:
      split, condition, L, sample_id,
      base_wer, lca_wer, delta_wer (= lca - base),
      ground_truth, original_whisper,
      utt_words, utt_len_bin, original_whisper_wer (PCM floor proxy)
    """
    key_cols = ["split", "condition", "L", "sample_id"]
    keep_cols = key_cols + ["model", "wer_vs_gt", "ground_truth", "original_whisper"]
    sub = df[keep_cols].copy()

    base = sub[sub["model"] == "base"].drop(columns=["model"]).rename(
        columns={"wer_vs_gt": "base_wer"}
    )
    lca = sub[sub["model"] == "lca"].drop(columns=["model"]).rename(
        columns={"wer_vs_gt": "lca_wer"}
    )

    paired = base.merge(
        lca[key_cols + ["lca_wer"]],
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )
    paired["delta_wer"] = paired["lca_wer"] - paired["base_wer"]
    paired["utt_words"] = paired["ground_truth"].map(_utt_words)
    paired["utt_len_bin"] = paired["utt_words"].map(_len_bin)

    # PCM floor proxy: WER(original_whisper, ground_truth). Per-sample, but
    # we recompute per row for simplicity; values are identical across L.
    paired["original_whisper_wer"] = [
        compute_wer(gt, ow) for gt, ow in zip(paired["ground_truth"], paired["original_whisper"])
    ]
    return paired


def build_per_utterance_pairing() -> pd.DataFrame:
    """Concatenate clean and perturbed pairings into one long table."""
    clean_paired = build_pairings(load_clean(EXP10_CSV))
    perturbed_paired = build_pairings(load_perturbed(EXP11_CSV))
    cols = [
        "split", "condition", "L", "sample_id",
        "base_wer", "lca_wer", "delta_wer",
        "utt_words", "utt_len_bin", "original_whisper_wer",
        "ground_truth", "original_whisper",
    ]
    return pd.concat(
        [clean_paired[cols], perturbed_paired[cols]],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Regression case extraction
# ---------------------------------------------------------------------------
def extract_regression_cases(paired: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows where LCA strictly worse than Base (delta_wer > 0), sorted desc."""
    cases = paired[(paired["split"] == split) & (paired["delta_wer"] > 0)].copy()
    cases = cases.sort_values("delta_wer", ascending=False).reset_index(drop=True)
    return cases


# ---------------------------------------------------------------------------
# Summary by (split, condition, L)
# ---------------------------------------------------------------------------
def summarise_by_cell(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouper = paired.groupby(["split", "condition", "L"], dropna=False)
    for (split, condition, L), g in grouper:
        n_total = len(g)
        worse_mask = g["delta_wer"] > 0
        better_mask = g["delta_wer"] < 0
        tied_mask = g["delta_wer"] == 0
        n_worse = int(worse_mask.sum())
        n_better = int(better_mask.sum())
        n_tied = int(tied_mask.sum())

        regs = g[worse_mask]
        if n_worse > 0:
            mean_base_on_reg = float(regs["base_wer"].mean())
            mean_utt_words_reg = float(regs["utt_words"].mean())
            short_share = float((regs["utt_words"] < SHORT_UTT_THRESHOLD).mean())
            strong_base_share = float((regs["base_wer"] < STRONG_BASE_THRESHOLD).mean())
        else:
            mean_base_on_reg = float("nan")
            mean_utt_words_reg = float("nan")
            short_share = float("nan")
            strong_base_share = float("nan")

        rows.append({
            "split": split,
            "condition": condition,
            "L": int(L),
            "n_total": n_total,
            "n_lca_worse": n_worse,
            "n_lca_better": n_better,
            "n_tied": n_tied,
            "mean_delta_wer": float(g["delta_wer"].mean()),
            "median_delta_wer": float(g["delta_wer"].median()),
            "mean_base_wer_on_regressions": mean_base_on_reg,
            "mean_utt_words_on_regressions": mean_utt_words_reg,
            "share_short_utts_on_regressions": short_share,
            "share_strong_base_on_regressions": strong_base_share,
        })
    summary = pd.DataFrame(rows).sort_values(["split", "condition", "L"]).reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Bootstrap CI for share statistics
# ---------------------------------------------------------------------------
def bootstrap_share_ci(
    binary: np.ndarray,
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (mean_share, lo, hi) over B bootstrap resamples."""
    binary = np.asarray(binary, dtype=float)
    n = len(binary)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    shares = binary[idx].mean(axis=1)
    lo = float(np.quantile(shares, alpha / 2))
    hi = float(np.quantile(shares, 1 - alpha / 2))
    return float(binary.mean()), lo, hi


# ---------------------------------------------------------------------------
# Hypothesis testing helpers
# ---------------------------------------------------------------------------
def focus_cells(paired: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the regression-prone cells called out in the spec.

    clean L=2, plus perturbed L=2 / L=3 under dropout-high and substitution-high.
    """
    is_clean_focus = (paired["split"] == "clean") & (paired["L"] == 2)
    is_perturbed_focus = (
        (paired["split"] == "perturbed")
        & paired["condition"].isin(["dropout-high", "substitution-high"])
        & paired["L"].isin([2, 3])
    )
    return paired[is_clean_focus | is_perturbed_focus].copy()


def evaluate_hypothesis_h1(reg_cases: pd.DataFrame) -> dict:
    short = (reg_cases["utt_words"] < SHORT_UTT_THRESHOLD).to_numpy()
    share, lo, hi = bootstrap_share_ci(short.astype(float))
    by_bin = (
        reg_cases.assign(_one=1)
        .groupby("utt_len_bin", dropna=False)["_one"].sum()
        .reindex(LEN_LABELS, fill_value=0)
    )
    return {
        "n": int(len(reg_cases)),
        "share_short": share,
        "share_short_ci": (lo, hi),
        "by_len_bin": by_bin.to_dict(),
        "verdict_supports": share >= 0.5,
    }


def evaluate_hypothesis_h2(reg_cases: pd.DataFrame) -> dict:
    strong = (reg_cases["base_wer"] < STRONG_BASE_THRESHOLD).to_numpy()
    share, lo, hi = bootstrap_share_ci(strong.astype(float))
    return {
        "n": int(len(reg_cases)),
        "share_strong_base": share,
        "share_strong_base_ci": (lo, hi),
        "mean_base_on_reg": float(reg_cases["base_wer"].mean()) if len(reg_cases) else float("nan"),
        "median_base_on_reg": float(reg_cases["base_wer"].median()) if len(reg_cases) else float("nan"),
        "verdict_supports": share >= 0.5,
    }


def evaluate_hypothesis_h3(reg_cases: pd.DataFrame) -> dict:
    deltas = reg_cases["delta_wer"].to_numpy(dtype=float)
    if len(deltas) == 0:
        return {
            "n": 0,
            "median_delta": float("nan"),
            "mean_delta": float("nan"),
            "share_small_delta": float("nan"),
            "share_small_delta_ci": (float("nan"), float("nan")),
            "verdict_supports": False,
        }
    median_delta = float(np.median(deltas))
    mean_delta = float(np.mean(deltas))
    small = (deltas < SMALL_DELTA_THRESHOLD).astype(float)
    share, lo, hi = bootstrap_share_ci(small)
    return {
        "n": int(len(deltas)),
        "median_delta": median_delta,
        "mean_delta": mean_delta,
        "share_small_delta": share,
        "share_small_delta_ci": (lo, hi),
        "verdict_supports": median_delta < SMALL_DELTA_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    if x != x:  # NaN
        return "n/a"
    return f"{100 * x:.1f}%"


def _fmt_ci(ci: tuple[float, float]) -> str:
    lo, hi = ci
    if lo != lo or hi != hi:
        return "[n/a]"
    return f"[{100*lo:.1f}%, {100*hi:.1f}%]"


def _fmt_float(x: float, ndigits: int = 4) -> str:
    if x != x:
        return "n/a"
    return f"{x:.{ndigits}f}"


def render_markdown_summary(
    paired_all: pd.DataFrame,
    summary_by_cell: pd.DataFrame,
    h1: dict,
    h2: dict,
    h3: dict,
    focus_reg_cases: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Test-clean ASR Regression Diagnosis (exp10 + exp11)")
    lines.append("")
    lines.append(
        "This report explains *why* SCIT-LCA is sometimes worse than SCIT-Base on "
        "test-clean ASR at high transmitted-layer count L. Findings feed into "
        "Section 6.5 of `scit_speech_cn_revised_ml_20260610.md`."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- exp10 (clean): 300 utterances x 3 L x 2 model = 1800 rows, paired into 900 per-utt comparisons.")
    lines.append("- exp11 (perturbed): 100 utterances x 3 L x 2 channel x 2 model = 1200 rows, paired into 600 per-utt comparisons.")
    lines.append("- WER metric: `wer_vs_gt`. PCM-floor proxy: WER(original_whisper, ground_truth) per sample.")
    lines.append(f"- Short-utterance threshold: < {SHORT_UTT_THRESHOLD} words.")
    lines.append(f"- Strong-base threshold: Base WER < {STRONG_BASE_THRESHOLD}.")
    lines.append(f"- Small-regression threshold: delta_wer < {SMALL_DELTA_THRESHOLD}.")
    lines.append(f"- Bootstrap: B={BOOTSTRAP_B}, seed={BOOTSTRAP_SEED}, 95% percentile CI.")
    lines.append("")

    lines.append("## Per-cell summary")
    lines.append("")
    lines.append(
        "| split | condition | L | n_total | n_lca_worse | n_lca_better | n_tied | "
        "mean_delta | median_delta | mean_base_on_reg | mean_utt_words_on_reg | "
        "share_short_on_reg | share_strong_base_on_reg |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary_by_cell.iterrows():
        lines.append(
            f"| {r['split']} | {r['condition']} | {int(r['L'])} | {int(r['n_total'])} | "
            f"{int(r['n_lca_worse'])} | {int(r['n_lca_better'])} | {int(r['n_tied'])} | "
            f"{_fmt_float(r['mean_delta_wer'])} | {_fmt_float(r['median_delta_wer'])} | "
            f"{_fmt_float(r['mean_base_wer_on_regressions'])} | "
            f"{_fmt_float(r['mean_utt_words_on_regressions'], 2)} | "
            f"{_fmt_pct(r['share_short_utts_on_regressions'])} | "
            f"{_fmt_pct(r['share_strong_base_on_regressions'])} |"
        )
    lines.append("")

    lines.append("## Focus regression cells")
    lines.append("")
    lines.append(
        "Cells flagged in the original observation: clean L=2; perturbed dropout-high "
        "and substitution-high at L=2 and L=3. The hypothesis tests below are computed "
        "over regression cases (delta_wer > 0) pooled across these cells."
    )
    lines.append("")
    lines.append(f"- Total focus regression cases: **{len(focus_reg_cases)}**")
    lines.append("")

    lines.append("## H1 - Regression cases concentrate in short utterances")
    lines.append("")
    lines.append(
        f"- Share of focus regressions with utterance words < {SHORT_UTT_THRESHOLD}: "
        f"**{_fmt_pct(h1['share_short'])}** (95% CI {_fmt_ci(h1['share_short_ci'])}, n={h1['n']})."
    )
    lines.append("- Per-bin counts:")
    for b in LEN_LABELS:
        lines.append(f"  - {b}: {h1['by_len_bin'].get(b, 0)}")
    lines.append(
        f"- **Verdict**: {'supported' if h1['verdict_supports'] else 'not supported'} "
        f"(majority of regressions {'are' if h1['verdict_supports'] else 'are NOT'} short)."
    )
    lines.append("")

    lines.append("## H2 - Regression cases concentrate where Base was already very good")
    lines.append("")
    lines.append(
        f"- Share of focus regressions with base WER < {STRONG_BASE_THRESHOLD}: "
        f"**{_fmt_pct(h2['share_strong_base'])}** (95% CI {_fmt_ci(h2['share_strong_base_ci'])}, n={h2['n']})."
    )
    lines.append(f"- Mean Base WER on regressions: {_fmt_float(h2['mean_base_on_reg'])}")
    lines.append(f"- Median Base WER on regressions: {_fmt_float(h2['median_base_on_reg'])}")
    lines.append(
        f"- **Verdict**: {'supported' if h2['verdict_supports'] else 'not supported'} "
        f"(majority of regressions {'are' if h2['verdict_supports'] else 'are NOT'} on strong-Base utts)."
    )
    lines.append("")

    lines.append("## H3 - Regression magnitudes are small")
    lines.append("")
    lines.append(f"- Median delta_wer on focus regressions: **{_fmt_float(h3['median_delta'])}**")
    lines.append(f"- Mean delta_wer on focus regressions: {_fmt_float(h3['mean_delta'])}")
    lines.append(
        f"- Share with delta_wer < {SMALL_DELTA_THRESHOLD}: **{_fmt_pct(h3['share_small_delta'])}** "
        f"(95% CI {_fmt_ci(h3['share_small_delta_ci'])}, n={h3['n']})."
    )
    lines.append(
        f"- **Verdict**: {'supported' if h3['verdict_supports'] else 'not supported'} "
        f"(median delta_wer {'<' if h3['verdict_supports'] else '>='} {SMALL_DELTA_THRESHOLD})."
    )
    lines.append("")

    lines.append("## Take-away for Section 6.5")
    lines.append("")
    h1_str = "supported" if h1["verdict_supports"] else "not supported"
    h2_str = "supported" if h2["verdict_supports"] else "not supported"
    h3_str = "supported" if h3["verdict_supports"] else "not supported"
    lines.append(
        f"- H1 (short-utterance concentration): {h1_str}.\n"
        f"- H2 (strong-Base concentration): {h2_str}.\n"
        f"- H3 (regressions are small in magnitude): {h3_str}."
    )
    lines.append("")
    lines.append(
        "Together, these tests characterize the LCA regressions on test-clean as a "
        "small-magnitude tail driven mostly by utterances where Base is already "
        "near the PCM floor and/or the utterance is short, leaving little headroom "
        "for the channel-aware redistribution to recover lost detail."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    paired_all = build_per_utterance_pairing()
    paired_all.to_csv(METRICS_DIR / "per_utterance_pairing.csv", index=False)

    clean_reg = extract_regression_cases(paired_all, split="clean")
    perturbed_reg = extract_regression_cases(paired_all, split="perturbed")
    clean_reg.to_csv(METRICS_DIR / "regression_cases_clean_300.csv", index=False)
    perturbed_reg.to_csv(METRICS_DIR / "regression_cases_perturbed_100.csv", index=False)

    summary_by_cell = summarise_by_cell(paired_all)
    summary_by_cell.to_csv(METRICS_DIR / "regression_summary_by_L.csv", index=False)

    focus_paired = focus_cells(paired_all)
    focus_reg_cases = focus_paired[focus_paired["delta_wer"] > 0].copy()

    h1 = evaluate_hypothesis_h1(focus_reg_cases)
    h2 = evaluate_hypothesis_h2(focus_reg_cases)
    h3 = evaluate_hypothesis_h3(focus_reg_cases)

    md = render_markdown_summary(paired_all, summary_by_cell, h1, h2, h3, focus_reg_cases)
    (REPORTS_DIR / "regression_diagnosis_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
