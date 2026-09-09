# Perturbed ASR/WER summary

- Whisper: `base.en`
- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | channel | n | WER vs GT ↓ | CER vs GT ↓ | WER vs original Whisper ↓ | actual drop | actual sub |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | dropout-high | 1 | 0.2857 | 0.1456 | 0.2963 | 0.0977 | 0.0000 |
| base | 1 | substitution-high | 1 | 0.1786 | 0.0696 | 0.1852 | 0.0000 | 0.0268 |
| base | 2 | dropout-high | 1 | 0.1429 | 0.0949 | 0.1481 | 0.1034 | 0.0000 |
| base | 2 | substitution-high | 1 | 0.1071 | 0.0696 | 0.0741 | 0.0000 | 0.0316 |
| base | 3 | dropout-high | 1 | 0.1429 | 0.0443 | 0.1111 | 0.1041 | 0.0000 |
| base | 3 | substitution-high | 1 | 0.1429 | 0.0443 | 0.1111 | 0.0000 | 0.0281 |
| lca | 1 | dropout-high | 1 | 0.3571 | 0.1962 | 0.3704 | 0.0977 | 0.0000 |
| lca | 1 | substitution-high | 1 | 0.3571 | 0.1709 | 0.4074 | 0.0000 | 0.0268 |
| lca | 2 | dropout-high | 1 | 0.0714 | 0.0190 | 0.1111 | 0.1034 | 0.0000 |
| lca | 2 | substitution-high | 1 | 0.0714 | 0.0253 | 0.0000 | 0.0000 | 0.0316 |
| lca | 3 | dropout-high | 1 | 0.0714 | 0.0380 | 0.0370 | 0.1041 | 0.0000 |
| lca | 3 | substitution-high | 1 | 0.0714 | 0.0253 | 0.0741 | 0.0000 | 0.0281 |

## LCA - Base deltas

| L | channel | ΔWER vs GT | ΔCER vs GT | ΔWER vs original Whisper |
|---:|---|---:|---:|---:|
| 1 | dropout-high | +0.0714 | +0.0506 | +0.0741 |
| 1 | substitution-high | +0.1786 | +0.1013 | +0.2222 |
| 2 | dropout-high | -0.0714 | -0.0759 | -0.0370 |
| 2 | substitution-high | -0.0357 | -0.0443 | -0.0741 |
| 3 | dropout-high | -0.0714 | -0.0063 | -0.0741 |
| 3 | substitution-high | -0.0714 | -0.0190 | -0.0370 |
