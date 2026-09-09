"""Compute the PCM Whisper WER/CER lower bound on test-clean_300 / test-other_300.

Reuses exp10's `original_whisper` field (= Whisper base.en transcript of PCM passthrough)
and the same jiwer normalization chain used in scripts/evaluate_asr_wer_onthefly.py.
"""
from pathlib import Path
import jiwer
import pandas as pd

CSV_TC = "output/experiments/exp10_asr_wer_onthefly_20260609/eval_asr_wer_clean/test-clean_300/metrics/asr_wer_results.csv"
CSV_TO = "output/experiments/exp10_asr_wer_onthefly_20260609/eval_asr_wer_clean/test-other_300/metrics/asr_wer_results.csv"

WORD = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])
CHAR = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])


def per_sample_pcm_wer(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    pcm = df.drop_duplicates("sample_id")[["sample_id", "ground_truth", "original_whisper"]].copy()
    pcm = pcm.dropna(subset=["ground_truth", "original_whisper"])
    rows = []
    for _, r in pcm.iterrows():
        gt = str(r["ground_truth"]).strip()
        ow = str(r["original_whisper"]).strip()
        try:
            wer = float(jiwer.wer([gt], [ow], reference_transform=WORD, hypothesis_transform=WORD))
            cer = float(jiwer.cer([gt], [ow], reference_transform=CHAR, hypothesis_transform=CHAR))
        except Exception:
            wer, cer = float("nan"), float("nan")
        rows.append({"sample_id": r["sample_id"], "wer_pcm": wer, "cer_pcm": cer})
    return pd.DataFrame(rows)


def corpus_pcm_wer(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    pcm = df.drop_duplicates("sample_id")[["sample_id", "ground_truth", "original_whisper"]].copy()
    pcm = pcm.dropna(subset=["ground_truth", "original_whisper"])
    refs = [str(s).strip() for s in pcm["ground_truth"].tolist()]
    hyps = [str(s).strip() for s in pcm["original_whisper"].tolist()]
    wer_corpus = float(jiwer.wer(refs, hyps, reference_transform=WORD, hypothesis_transform=WORD))
    cer_corpus = float(jiwer.cer(refs, hyps, reference_transform=CHAR, hypothesis_transform=CHAR))
    return {"n": len(refs), "wer_corpus": wer_corpus, "cer_corpus": cer_corpus}


outputs = {}
for name, csv_path in [("test-clean_300", CSV_TC), ("test-other_300", CSV_TO)]:
    per = per_sample_pcm_wer(csv_path)
    out_csv = Path(f"output/experiments/exp10_asr_wer_onthefly_20260609/pcm_lower_bound/{name}_pcm_per_sample.csv")
    per.to_csv(out_csv, index=False)
    corpus = corpus_pcm_wer(csv_path)
    macro_wer = float(per["wer_pcm"].mean())
    macro_cer = float(per["cer_pcm"].mean())
    outputs[name] = {
        "n": corpus["n"],
        "wer_corpus": corpus["wer_corpus"],  # reference-length-weighted
        "cer_corpus": corpus["cer_corpus"],
        "wer_macro_per_sample_mean": macro_wer,  # equally-weighted per sample
        "cer_macro_per_sample_mean": macro_cer,
    }

# Markdown summary
md_lines = [
    "# PCM Whisper Lower Bound (test-clean_300 / test-other_300)",
    "",
    "Reuses exp10's `original_whisper` field (Whisper base.en transcript of PCM-passthrough audio) with the same jiwer normalization chain (`ToLowerCase + RemovePunctuation + RemoveMultipleSpaces + Strip`) as `scripts/evaluate_asr_wer_onthefly.py`.",
    "",
    "| split | n | WER (corpus) | WER (macro per-sample) | CER (corpus) | CER (macro per-sample) |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, m in outputs.items():
    md_lines.append(
        f"| {name} | {m['n']} | {m['wer_corpus']:.4f} | {m['wer_macro_per_sample_mean']:.4f} | {m['cer_corpus']:.4f} | {m['cer_macro_per_sample_mean']:.4f} |"
    )
md_lines.append("")
md_lines.append("Per-sample CSVs: `*_pcm_per_sample.csv`.")

Path("output/experiments/exp10_asr_wer_onthefly_20260609/pcm_lower_bound/pcm_lower_bound_summary.md").write_text(
    "\n".join(md_lines), encoding="utf-8"
)

import json
print(json.dumps(outputs, indent=2))
