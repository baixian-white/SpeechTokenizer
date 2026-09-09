"""Patch the remaining unresolved <<EXP12_*:diff_*>> and <<EXP12_*:wilcoxon_*>> tokens
in scit_speech_cn_revised_ml_20260610.exp12_section_filled.md by reading the production
pairwise CSVs and substituting real values. Idempotent.

Skips the literal placeholder examples on lines 3 and 9 of the document (those are
documentation prose, not real tokens — they appear inside backticks as illustrative
syntax).
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRWISE = {
    "TC": ROOT / "runs" / "test-clean_300" / "metrics" / "lca_vs_baselines_pairwise.csv",
    "TO": ROOT / "runs" / "test-other_300" / "metrics" / "lca_vs_baselines_pairwise.csv",
}
DRAFT = Path(r"h:/H-CODE/speechtokenizer/output/doc/paper_drafts/scit_speech_cn_revised_ml_20260610.exp12_section_filled.md")

# Map "baseline-key" tokens used in the draft to the actual (method, codec_setting) row in pairwise CSV.
BASELINE_MAP = {
    "dac_nq1": ("dac", "n_q_1", 1),
    "dac_nq2": ("dac", "n_q_2", 2),
    "dac_nq3": ("dac", "n_q_3", 3),
    "encodec_1p5kbps": ("encodec", "bw1.5kbps_n_cb2", 3),
    "opus_6kbps": ("opus", "opus_6000bps", 3),
}

# Metric keys exposed in pairwise CSV.
METRICS = {"mel_l1", "stoi", "pesq_wb", "si_snr_db", "wave_l1", "corr"}


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "nan"
    if p < 1e-3:
        return "p<0.001"
    return f"p={p:.3g}"


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def load_pairwise(split: str) -> pd.DataFrame:
    df = pd.read_csv(PAIRWISE[split], comment="#")
    return df


def find_pairwise_row(df: pd.DataFrame, baseline_key: str, scit_L: int) -> pd.Series | None:
    method, codec_setting, default_L = BASELINE_MAP.get(baseline_key, (None, None, None))
    if method is None:
        return None
    L_str = f"L={scit_L}"
    candidates = df[df["comparison"].str.contains(f"scit_lca/{L_str} vs {method}/{codec_setting}", regex=False, na=False)]
    if candidates.empty:
        return None
    return candidates.iloc[0]


TOKEN_DIFF_RE = re.compile(r"<<EXP12_(TC|TO):diff_([a-z0-9_]+?)_vs_scit_lca_L([1-3])_([a-z0-9_]+?)_ci95>>")
TOKEN_P_RE = re.compile(r"<<EXP12_(TC|TO):wilcoxon_([a-z0-9_]+?)_vs_scit_lca_L([1-3])_([a-z0-9_]+?)_p>>")


def patch():
    cache = {}
    text = DRAFT.read_text(encoding="utf-8")
    unresolved = []
    replacements = 0

    def get_df(split: str) -> pd.DataFrame:
        if split not in cache:
            cache[split] = load_pairwise(split)
        return cache[split]

    def replace_diff(m: re.Match) -> str:
        nonlocal replacements
        split, baseline_key, L_str, metric = m.group(1), m.group(2), m.group(3), m.group(4)
        if metric not in METRICS:
            unresolved.append((m.group(0), f"unknown metric '{metric}'"))
            return m.group(0)
        df = get_df(split)
        row = find_pairwise_row(df, baseline_key, int(L_str))
        if row is None:
            unresolved.append((m.group(0), "no matching pairwise row"))
            return m.group(0)
        try:
            mean = float(row[f"{metric}_mean_diff"]) ; lo = float(row[f"{metric}_ci_low"]) ; hi = float(row[f"{metric}_ci_high"])
        except KeyError as e:
            unresolved.append((m.group(0), f"missing column {e}"))
            return m.group(0)
        replacements += 1
        return fmt_ci(mean, lo, hi)

    def replace_p(m: re.Match) -> str:
        nonlocal replacements
        split, baseline_key, L_str, metric = m.group(1), m.group(2), m.group(3), m.group(4)
        if metric not in METRICS:
            unresolved.append((m.group(0), f"unknown metric '{metric}'"))
            return m.group(0)
        df = get_df(split)
        row = find_pairwise_row(df, baseline_key, int(L_str))
        if row is None:
            unresolved.append((m.group(0), "no matching pairwise row"))
            return m.group(0)
        try:
            p = float(row[f"{metric}_wilcoxon_p"])
        except KeyError as e:
            unresolved.append((m.group(0), f"missing column {e}"))
            return m.group(0)
        replacements += 1
        return fmt_p(p)

    new_text = TOKEN_DIFF_RE.sub(replace_diff, text)
    new_text = TOKEN_P_RE.sub(replace_p, new_text)
    DRAFT.write_text(new_text, encoding="utf-8")
    print({"replacements": replacements, "unresolved": unresolved, "draft": str(DRAFT)})


if __name__ == "__main__":
    patch()
