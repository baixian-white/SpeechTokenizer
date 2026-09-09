# PCM Whisper Lower Bound (test-clean_300 / test-other_300)

Reuses exp10's `original_whisper` field (Whisper base.en transcript of PCM-passthrough audio) with the same jiwer normalization chain (`ToLowerCase + RemovePunctuation + RemoveMultipleSpaces + Strip`) as `scripts/evaluate_asr_wer_onthefly.py`.

| split | n | WER (corpus) | WER (macro per-sample) | CER (corpus) | CER (macro per-sample) |
|---|---:|---:|---:|---:|---:|
| test-clean_300 | 300 | 0.0359 | 0.0486 | 0.0129 | 0.0185 |
| test-other_300 | 300 | 0.1552 | 0.1915 | 0.0709 | 0.0923 |

Per-sample CSVs: `*_pcm_per_sample.csv`.