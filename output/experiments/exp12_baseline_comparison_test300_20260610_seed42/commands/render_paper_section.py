"""Render the exp12 paper section by substituting placeholder tokens with
real numbers from the per-method summary and pairwise CSVs.

Pipeline (when --skip-analyze is NOT set):
  1. Run analyze_baseline_300_stats.py for both splits (test-clean_300 +
     test-other_300), producing per-method-summary + pairwise CSVs +
     baseline_300_summary.md under runs/<split>/{metrics,reports}/.
  2. Read per_method_summary.csv and lca_vs_baselines_pairwise.csv from
     each split.
  3. Substitute every <<EXP12_TC:KEY>> / <<EXP12_TO:KEY>> token in the
     draft template with the rendered value. Tokens that cannot be
     resolved (literal-wildcard placeholders, GLOBAL placeholders, mis-
     keyed entries, etc.) are left in place and reported as
     tokens_unresolved.
  4. Write the filled markdown next to the template (suffix .filled.md).
  5. Print a JSON status line and exit.

Token grammar (recognised forms):

    <<EXP12_<SPLIT>:<METHOD>_<METRIC>_mean>>
    <<EXP12_<SPLIT>:<METHOD>_<METRIC>_ci>>
    <<EXP12_<SPLIT>:<METHOD>_<METRIC>_mean (95% CI)>>      # template alias of *_ci
    <<EXP12_<SPLIT>:pair_<A>_vs_<B>_<metric>_diff>>
    <<EXP12_<SPLIT>:pair_<A>_vs_<B>_<metric>_p>>
    <<EXP12_<SPLIT>:diff_<X>_vs_<Y>_<metric>_ci95>>        # template alias
    <<EXP12_<SPLIT>:wilcoxon_<X>_vs_<Y>_<metric>_p>>       # template alias

  SPLIT ∈ {TC, TO}.  METHOD ∈ METHOD_TABLE keys (with friendly aliases
  for the variants used by the existing draft, e.g. ``dac_nq1`` ≡
  ``dac_n_q_1``, ``encodec_1p5kbps`` ≡ ``encodec_1.5k``).  METRIC ∈
  {mel_l1, stoi, pesq_wb, si_snr_db, wave_l1, corr} and the short alias
  ``mel`` ≡ ``mel_l1`` is also accepted for pairwise tokens (matches the
  spec example pair_lca_L3_vs_dac_n_q_3_mel_diff).

Run:
  python render_paper_section.py                    # full pipeline
  python render_paper_section.py --skip-analyze     # re-render only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
COMMANDS_DIR = THIS_FILE.parent
EXPERIMENT_ROOT = COMMANDS_DIR.parent
ANALYZE_SCRIPT = COMMANDS_DIR / "analyze_baseline_300_stats.py"

TEMPLATE_PATH = Path(
    r"h:/H-CODE/speechtokenizer/output/doc/paper_drafts/"
    r"scit_speech_cn_revised_ml_20260610.exp12_section_draft.md"
).resolve()
FILLED_PATH = Path(
    r"h:/H-CODE/speechtokenizer/output/doc/paper_drafts/"
    r"scit_speech_cn_revised_ml_20260610.exp12_section_filled.md"
).resolve()

SPLITS: Dict[str, str] = {
    "TC": "test-clean_300",
    "TO": "test-other_300",
}


# ---------------------------------------------------------------------------
# Method / metric registry
# ---------------------------------------------------------------------------

# Canonical method-key -> (method, codec_setting) as written by the runner.
# Opus rows are bitrate-resolved at runtime via find_opus_setting() below
# because the runner emits 'opus_<bps>bps' (not '<kbps>kbps').
METHOD_TABLE: Dict[str, Tuple[str, Optional[str]]] = {
    "scit_base_L1": ("scit_base", "L=1"),
    "scit_base_L2": ("scit_base", "L=2"),
    "scit_base_L3": ("scit_base", "L=3"),
    "scit_lca_L1":  ("scit_lca",  "L=1"),
    "scit_lca_L2":  ("scit_lca",  "L=2"),
    "scit_lca_L3":  ("scit_lca",  "L=3"),
    "dac_n_q_1":    ("dac",       "n_q_1"),
    "dac_n_q_2":    ("dac",       "n_q_2"),
    "dac_n_q_3":    ("dac",       "n_q_3"),
    "encodec_1.5k": ("encodec",   "bw1.5kbps_n_cb2"),
    "encodec_3k":   ("encodec",   "bw3.0kbps_n_cb4"),
    "encodec_6k":   ("encodec",   "bw6.0kbps_n_cb8"),
    # opus settings resolved dynamically per-split:
    "opus_6k":      ("opus", None),
    "opus_8k":      ("opus", None),
    "opus_12k":     ("opus", None),
    "pcm":          ("pcm",       "16bit_16khz_passthrough"),
}

# Friendly aliases used by the existing draft / spec examples.
METHOD_ALIASES: Dict[str, str] = {
    "dac_nq1":          "dac_n_q_1",
    "dac_nq2":          "dac_n_q_2",
    "dac_nq3":          "dac_n_q_3",
    "encodec_1p5kbps":  "encodec_1.5k",
    "encodec_3kbps":    "encodec_3k",
    "encodec_6kbps":    "encodec_6k",
    "opus_6kbps":       "opus_6k",
    "opus_8kbps":       "opus_8k",
    "opus_12kbps":      "opus_12k",
    # spec uses 'lca_L3' shorthand inside pair_ tokens (no 'scit_' prefix).
    "lca_L1":           "scit_lca_L1",
    "lca_L2":           "scit_lca_L2",
    "lca_L3":           "scit_lca_L3",
    "base_L1":          "scit_base_L1",
    "base_L2":          "scit_base_L2",
    "base_L3":          "scit_base_L3",
}

# Metric aliases.  The pairwise spec uses 'mel_diff' / 'mel_p' which we map
# to mel_l1.  Everything else is identity.
METRIC_CANON: Dict[str, str] = {
    "mel":      "mel_l1",
    "mel_l1":   "mel_l1",
    "stoi":     "stoi",
    "pesq_wb":  "pesq_wb",
    "pesq":     "pesq_wb",
    "si_snr":   "si_snr_db",
    "si_snr_db": "si_snr_db",
    "wave_l1":  "wave_l1",
    "wave":     "wave_l1",
    "corr":     "corr",
}

ALL_METRICS = ("wave_l1", "mel_l1", "si_snr_db", "corr", "stoi", "pesq_wb")


def canon_method(key: str) -> Optional[str]:
    if key in METHOD_TABLE:
        return key
    return METHOD_ALIASES.get(key)


def canon_metric(key: str) -> Optional[str]:
    return METRIC_CANON.get(key)


# ---------------------------------------------------------------------------
# Opus setting resolution (matches analyze_baseline_300_stats.find_opus_setting)
# ---------------------------------------------------------------------------

def resolve_opus_setting(df: pd.DataFrame, target_kbps: float) -> Optional[str]:
    opus = df[df["method"] == "opus"]
    if opus.empty:
        return None
    settings = sorted(opus["codec_setting"].astype(str).unique().tolist())
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(kbps|bps)\b", re.IGNORECASE)
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
    return None


def build_method_index(summary_df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """Materialize METHOD_TABLE for one split, resolving opus rows from the
    summary dataframe (which carries the codec_setting column)."""
    # Reconstruct an audio-quality-style frame from the per-method summary so
    # we can reuse resolve_opus_setting.  per_method_summary has columns
    # method + codec_setting which is enough.
    src = summary_df[["method", "codec_setting"]].drop_duplicates()
    resolved: Dict[str, Tuple[str, str]] = {}
    for key, (method, setting) in METHOD_TABLE.items():
        if setting is not None:
            resolved[key] = (method, setting)
            continue
        # opus_<kbps>k -> resolve dynamically
        m = re.match(r"opus_(\d+(?:\.\d+)?)k$", key)
        if not m:
            continue
        target = float(m.group(1))
        s = resolve_opus_setting(src, target)
        if s is not None:
            resolved[key] = (method, s)
    return resolved


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def fmt4(v) -> str:
    if v is None or _is_nan(v):
        return "NaN"
    return f"{float(v):.4f}"


def fmt_mean(v) -> str:
    return fmt4(v)


def fmt_ci(mean, lo, hi) -> str:
    return f"{fmt4(mean)} [{fmt4(lo)}, {fmt4(hi)}]"


def fmt_p(p) -> str:
    if p is None or _is_nan(p):
        return "p=NaN"
    p = float(p)
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.2f}"


# ---------------------------------------------------------------------------
# Per-method lookup
# ---------------------------------------------------------------------------

def lookup_method_metric(
    summary_df: pd.DataFrame,
    method_index: Dict[str, Tuple[str, str]],
    method_key: str,
    metric: str,
) -> Optional[Tuple[float, float, float]]:
    if method_key not in method_index:
        return None
    method, setting = method_index[method_key]
    sub = summary_df[
        (summary_df["method"] == method)
        & (summary_df["codec_setting"] == setting)
    ]
    if sub.empty:
        return None
    row = sub.iloc[0]
    mean = row.get(f"{metric}_mean")
    lo = row.get(f"{metric}_ci_low")
    hi = row.get(f"{metric}_ci_high")
    if mean is None or _is_nan(mean):
        return None
    return (float(mean), float(lo), float(hi))


# ---------------------------------------------------------------------------
# Pairwise lookup
# ---------------------------------------------------------------------------

def _comparison_label(method: str, setting: str) -> str:
    return f"{method}/{setting}"


def lookup_pair(
    pairwise_df: pd.DataFrame,
    method_index: Dict[str, Tuple[str, str]],
    a_key: str,
    b_key: str,
    metric: str,
) -> Optional[Dict[str, float]]:
    """Return {'mean_diff', 'ci_low', 'ci_high', 'p'} for the pair A-vs-B,
    where the difference is reported as A minus B.  The pairwise CSV stores
    LCA minus baseline for rows whose comparison is 'scit_lca/... vs
    <baseline>/...'.  If the requested pair is reversed (baseline first), we
    fetch the LCA-first row and negate mean_diff and CI bounds.
    """
    a = method_index.get(a_key)
    b = method_index.get(b_key)
    if a is None or b is None:
        return None
    a_label = _comparison_label(*a)
    b_label = _comparison_label(*b)
    forward = f"{a_label} vs {b_label}"
    reverse = f"{b_label} vs {a_label}"
    sign = 1.0
    sub = pairwise_df[pairwise_df["comparison"] == forward]
    if sub.empty:
        sub = pairwise_df[pairwise_df["comparison"] == reverse]
        sign = -1.0
    if sub.empty:
        return None
    row = sub.iloc[0]
    md = row.get(f"{metric}_mean_diff")
    lo = row.get(f"{metric}_ci_low")
    hi = row.get(f"{metric}_ci_high")
    p = row.get(f"{metric}_wilcoxon_p")
    if md is None or _is_nan(md):
        return None
    md = sign * float(md)
    if sign < 0:
        # CI bounds also flip; the new low is -old_high, new high is -old_low.
        new_lo = -float(hi) if hi is not None and not _is_nan(hi) else float("nan")
        new_hi = -float(lo) if lo is not None and not _is_nan(lo) else float("nan")
        lo, hi = new_lo, new_hi
    return {
        "mean_diff": md,
        "ci_low": float(lo) if lo is not None else float("nan"),
        "ci_high": float(hi) if hi is not None else float("nan"),
        "p": float(p) if p is not None and not _is_nan(p) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------

# Match: pair_<A>_vs_<B>_<metric>_(diff|p)
_PAIR_RE = re.compile(r"^pair_(?P<a>.+)_vs_(?P<b>.+)_(?P<metric>[a-z_]+)_(?P<kind>diff|p)$")
# Match: diff_<X>_vs_<Y>_<metric>_ci95   (template alias)
_DIFF_RE = re.compile(r"^diff_(?P<x>.+)_vs_(?P<y>.+)_(?P<metric>[a-z_0-9]+)_ci95$")
# Match: wilcoxon_<X>_vs_<Y>_<metric>_p  (template alias)
_WIL_RE  = re.compile(r"^wilcoxon_(?P<x>.+)_vs_(?P<y>.+)_(?P<metric>[a-z_0-9]+)_p$")
# Match: <method>_<metric>_(mean|ci)        OR
#        <method>_<metric>_mean (95% CI)    (template alias of *_ci)
_SINGLE_RE = re.compile(
    r"^(?P<rest>.+?)_(?P<kind>mean|ci)(?:\s*\(95%\s*CI\))?$",
    re.IGNORECASE,
)


def _split_method_metric(rest: str) -> Optional[Tuple[str, str]]:
    """Given e.g. 'scit_lca_L1_mel_l1' or 'dac_nq1_pesq_wb', split into
    (method_key, metric).  We try every metric suffix until one matches a
    known method key (after alias resolution)."""
    for metric_alias, _ in sorted(METRIC_CANON.items(), key=lambda kv: -len(kv[0])):
        suffix = "_" + metric_alias
        if rest.endswith(suffix):
            method_key = rest[: -len(suffix)]
            if canon_method(method_key) is not None:
                return (canon_method(method_key), METRIC_CANON[metric_alias])
    return None


def resolve_token(
    split: str,
    key: str,
    summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    method_index: Dict[str, Tuple[str, str]],
) -> Optional[str]:
    """Return the rendered string for one token, or None if unresolvable."""
    key = key.strip()

    # ---- pairwise: pair_<A>_vs_<B>_<metric>_diff  /  _p
    m = _PAIR_RE.match(key)
    if m:
        a_key = canon_method(m.group("a"))
        b_key = canon_method(m.group("b"))
        metric = canon_metric(m.group("metric"))
        if a_key is None or b_key is None or metric is None:
            return None
        info = lookup_pair(pairwise_df, method_index, a_key, b_key, metric)
        if info is None:
            return None
        if m.group("kind") == "diff":
            return fmt_ci(info["mean_diff"], info["ci_low"], info["ci_high"])
        return fmt_p(info["p"])

    # ---- pairwise alias: diff_<X>_vs_<Y>_<metric>_ci95
    m = _DIFF_RE.match(key)
    if m:
        x_key = canon_method(m.group("x"))
        y_key = canon_method(m.group("y"))
        metric = canon_metric(m.group("metric"))
        if x_key is None or y_key is None or metric is None:
            return None
        info = lookup_pair(pairwise_df, method_index, x_key, y_key, metric)
        if info is None:
            return None
        return fmt_ci(info["mean_diff"], info["ci_low"], info["ci_high"])

    # ---- pairwise alias: wilcoxon_<X>_vs_<Y>_<metric>_p
    m = _WIL_RE.match(key)
    if m:
        x_key = canon_method(m.group("x"))
        y_key = canon_method(m.group("y"))
        metric = canon_metric(m.group("metric"))
        if x_key is None or y_key is None or metric is None:
            return None
        info = lookup_pair(pairwise_df, method_index, x_key, y_key, metric)
        if info is None:
            return None
        return fmt_p(info["p"])

    # ---- single: <method>_<metric>_mean | _ci | _mean (95% CI)
    m = _SINGLE_RE.match(key)
    if m:
        rest = m.group("rest")
        kind = m.group("kind").lower()
        # If the original key carried the "(95% CI)" suffix, kind is still
        # 'mean' but we want CI rendering.
        wants_ci = kind == "ci" or "(95% CI)" in key.upper().replace("  ", " ")
        split_pair = _split_method_metric(rest)
        if split_pair is None:
            return None
        method_key, metric = split_pair
        triple = lookup_method_metric(summary_df, method_index, method_key, metric)
        if triple is None:
            return None
        mean, lo, hi = triple
        if wants_ci:
            return fmt_ci(mean, lo, hi)
        return fmt_mean(mean)

    return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_analyze() -> None:
    """Invoke analyze_baseline_300_stats.py for both splits."""
    if not ANALYZE_SCRIPT.exists():
        raise FileNotFoundError(f"analyze script missing: {ANALYZE_SCRIPT}")
    cmd = [sys.executable, str(ANALYZE_SCRIPT)]
    print(f"[render] running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(EXPERIMENT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(
            f"analyze_baseline_300_stats.py exited with code {res.returncode}"
        )


def load_split_tables(split_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_csv = split_dir / "metrics" / "per_method_summary.csv"
    pairwise_csv = split_dir / "metrics" / "lca_vs_baselines_pairwise.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"missing per_method_summary.csv: {summary_csv}")
    if not pairwise_csv.exists():
        raise FileNotFoundError(f"missing lca_vs_baselines_pairwise.csv: {pairwise_csv}")
    summary_df = pd.read_csv(summary_csv)
    # The pairwise CSV is prefixed with a single '#'-comment line written by
    # analyze_baseline_300_stats.py; pandas handles that with comment='#'.
    pairwise_df = pd.read_csv(pairwise_csv, comment="#")
    return summary_df, pairwise_df


# Token regex.  Captures SPLIT (TC|TO|...) and KEY (everything up to '>>').
TOKEN_RE = re.compile(r"<<EXP12_(?P<split>[A-Z]+):(?P<key>[^>]+)>>")


def render_template(
    template_text: str,
    tables_by_split: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[str, str]]]],
) -> Tuple[str, int, List[str]]:
    replaced = 0
    unresolved: List[str] = []

    def _sub(match: re.Match) -> str:
        nonlocal replaced
        split = match.group("split")
        key = match.group("key")
        full = match.group(0)
        if split not in SPLITS:
            unresolved.append(full)
            return full
        summary_df, pairwise_df, method_index = tables_by_split[split]
        rendered = resolve_token(split, key, summary_df, pairwise_df, method_index)
        if rendered is None:
            unresolved.append(full)
            return full
        replaced += 1
        return rendered

    out = TOKEN_RE.sub(_sub, template_text)
    return out, replaced, unresolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-analyze", action="store_true",
        help="skip running analyze_baseline_300_stats.py and use existing CSVs",
    )
    parser.add_argument(
        "--template", type=Path, default=TEMPLATE_PATH,
        help="path to the placeholder draft template",
    )
    parser.add_argument(
        "--out", type=Path, default=FILLED_PATH,
        help="path to write the filled markdown",
    )
    args = parser.parse_args()

    if not args.skip_analyze:
        run_analyze()

    if not args.template.exists():
        raise FileNotFoundError(f"template not found: {args.template}")

    tables_by_split: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[str, str]]]] = {}
    row_counts: Dict[str, int] = {}
    for split_code, split_name in SPLITS.items():
        split_dir = EXPERIMENT_ROOT / "runs" / split_name
        summary_df, pairwise_df = load_split_tables(split_dir)
        method_index = build_method_index(summary_df)
        tables_by_split[split_code] = (summary_df, pairwise_df, method_index)
        row_counts[split_code] = int(len(summary_df))

    template_text = args.template.read_text(encoding="utf-8")
    filled_text, replaced, unresolved = render_template(template_text, tables_by_split)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(filled_text, encoding="utf-8")

    status = {
        "test_clean_rows": row_counts.get("TC", 0),
        "test_other_rows": row_counts.get("TO", 0),
        "tokens_replaced": replaced,
        "tokens_unresolved": sorted(set(unresolved)),
        "filled_path": str(args.out),
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
