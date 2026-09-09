# On-the-fly ASR/WER summary

- Whisper: `base.en`
- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | n | WER vs GT ↓ | CER vs GT ↓ | WER vs original Whisper ↓ |
|---|---:|---:|---:|---:|---:|
| base | 1 | 300 | 0.6986 | 0.4671 | 0.6735 |
| base | 2 | 300 | 0.4791 | 0.2867 | 0.4355 |
| base | 3 | 300 | 0.3929 | 0.2201 | 0.3395 |
| lca | 1 | 300 | 0.6941 | 0.4454 | 0.6608 |
| lca | 2 | 300 | 0.4625 | 0.2772 | 0.4143 |
| lca | 3 | 300 | 0.3665 | 0.2123 | 0.2929 |

## LCA - Base deltas

| L | ΔWER vs GT | ΔCER vs GT | ΔWER vs original Whisper |
|---:|---:|---:|---:|
| 1 | -0.0045 | -0.0217 | -0.0128 |
| 2 | -0.0166 | -0.0095 | -0.0213 |
| 3 | -0.0263 | -0.0077 | -0.0466 |
