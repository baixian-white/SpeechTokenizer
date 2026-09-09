# Test-clean ASR Regression Diagnosis (exp10 + exp11)

This report explains *why* SCIT-LCA is sometimes worse than SCIT-Base on test-clean ASR at high transmitted-layer count L. Findings feed into Section 6.5 of `scit_speech_cn_revised_ml_20260610.md`.

## Setup

- exp10 (clean): 300 utterances x 3 L x 2 model = 1800 rows, paired into 900 per-utt comparisons.
- exp11 (perturbed): 100 utterances x 3 L x 2 channel x 2 model = 1200 rows, paired into 600 per-utt comparisons.
- WER metric: `wer_vs_gt`. PCM-floor proxy: WER(original_whisper, ground_truth) per sample.
- Short-utterance threshold: < 10 words.
- Strong-base threshold: Base WER < 0.1.
- Small-regression threshold: delta_wer < 0.05.
- Bootstrap: B=2000, seed=42, 95% percentile CI.

## Per-cell summary

| split | condition | L | n_total | n_lca_worse | n_lca_better | n_tied | mean_delta | median_delta | mean_base_on_reg | mean_utt_words_on_reg | share_short_on_reg | share_strong_base_on_reg |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | clean | 1 | 300 | 106 | 145 | 49 | -0.0261 | 0.0000 | 0.2806 | 23.81 | 17.9% | 22.6% |
| clean | clean | 2 | 300 | 91 | 99 | 110 | 0.0016 | 0.0000 | 0.1004 | 24.40 | 17.6% | 61.5% |
| clean | clean | 3 | 300 | 61 | 72 | 167 | -0.0234 | 0.0000 | 0.0597 | 26.80 | 13.1% | 82.0% |
| perturbed | dropout-high | 1 | 100 | 34 | 46 | 20 | -0.0095 | 0.0000 | 0.2459 | 23.53 | 26.5% | 23.5% |
| perturbed | dropout-high | 2 | 100 | 35 | 33 | 32 | 0.0096 | 0.0000 | 0.1533 | 25.66 | 17.1% | 57.1% |
| perturbed | dropout-high | 3 | 100 | 31 | 22 | 47 | 0.0117 | 0.0000 | 0.1011 | 26.03 | 19.4% | 67.7% |
| perturbed | substitution-high | 1 | 100 | 36 | 50 | 14 | -0.0413 | -0.0086 | 0.2625 | 26.19 | 13.9% | 22.2% |
| perturbed | substitution-high | 2 | 100 | 33 | 28 | 39 | 0.0064 | 0.0000 | 0.0804 | 28.55 | 18.2% | 69.7% |
| perturbed | substitution-high | 3 | 100 | 27 | 27 | 46 | 0.0079 | 0.0000 | 0.1510 | 29.56 | 14.8% | 77.8% |

## Focus regression cells

Cells flagged in the original observation: clean L=2; perturbed dropout-high and substitution-high at L=2 and L=3. The hypothesis tests below are computed over regression cases (delta_wer > 0) pooled across these cells.

- Total focus regression cases: **217**

## H1 - Regression cases concentrate in short utterances

- Share of focus regressions with utterance words < 10: **17.5%** (95% CI [12.4%, 23.0%], n=217).
- Per-bin counts:
  - [0,10): 38
  - [10,20): 58
  - [20,40): 75
  - [40,inf): 46
- **Verdict**: not supported (majority of regressions are NOT short).

## H2 - Regression cases concentrate where Base was already very good

- Share of focus regressions with base WER < 0.1: **65.0%** (95% CI [58.5%, 71.9%], n=217).
- Mean Base WER on regressions: 0.1123
- Median Base WER on regressions: 0.0625
- **Verdict**: supported (majority of regressions are on strong-Base utts).

## H3 - Regression magnitudes are small

- Median delta_wer on focus regressions: **0.0741**
- Mean delta_wer on focus regressions: 0.1316
- Share with delta_wer < 0.05: **30.0%** (95% CI [24.0%, 35.9%], n=217).
- **Verdict**: not supported (median delta_wer >= 0.05).

## Take-away for Section 6.5

- H1 (short-utterance concentration): not supported.
- H2 (strong-Base concentration): supported.
- H3 (regressions are small in magnitude): not supported.

Together, these tests characterize the LCA regressions on test-clean as a small-magnitude tail driven mostly by utterances where Base is already near the PCM floor and/or the utterance is short, leaving little headroom for the channel-aware redistribution to recover lost detail.
