# Perturbed ASR/WER summary

- Whisper: `base.en`
- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | channel | n | WER vs GT ↓ | CER vs GT ↓ | WER vs original Whisper ↓ | actual drop | actual sub |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | dropout-high | 100 | 0.3321 | 0.1801 | 0.3228 | 0.1022 | 0.0000 |
| base | 1 | substitution-high | 100 | 0.3685 | 0.1981 | 0.3553 | 0.0000 | 0.0296 |
| base | 2 | dropout-high | 100 | 0.1514 | 0.0704 | 0.1210 | 0.1013 | 0.0000 |
| base | 2 | substitution-high | 100 | 0.1464 | 0.0617 | 0.1164 | 0.0000 | 0.0303 |
| base | 3 | dropout-high | 100 | 0.0971 | 0.0398 | 0.0726 | 0.1001 | 0.0000 |
| base | 3 | substitution-high | 100 | 0.1199 | 0.0494 | 0.0871 | 0.0000 | 0.0297 |
| lca | 1 | dropout-high | 100 | 0.3226 | 0.1721 | 0.3001 | 0.1022 | 0.0000 |
| lca | 1 | substitution-high | 100 | 0.3272 | 0.1757 | 0.3182 | 0.0000 | 0.0296 |
| lca | 2 | dropout-high | 100 | 0.1610 | 0.0783 | 0.1420 | 0.1013 | 0.0000 |
| lca | 2 | substitution-high | 100 | 0.1528 | 0.0792 | 0.1335 | 0.0000 | 0.0303 |
| lca | 3 | dropout-high | 100 | 0.1088 | 0.0527 | 0.0832 | 0.1001 | 0.0000 |
| lca | 3 | substitution-high | 100 | 0.1278 | 0.0529 | 0.0946 | 0.0000 | 0.0297 |

## LCA - Base deltas

| L | channel | ΔWER vs GT | ΔCER vs GT | ΔWER vs original Whisper |
|---:|---|---:|---:|---:|
| 1 | dropout-high | -0.0095 | -0.0080 | -0.0227 |
| 1 | substitution-high | -0.0413 | -0.0224 | -0.0371 |
| 2 | dropout-high | +0.0096 | +0.0079 | +0.0209 |
| 2 | substitution-high | +0.0064 | +0.0175 | +0.0171 |
| 3 | dropout-high | +0.0117 | +0.0129 | +0.0106 |
| 3 | substitution-high | +0.0079 | +0.0036 | +0.0075 |
