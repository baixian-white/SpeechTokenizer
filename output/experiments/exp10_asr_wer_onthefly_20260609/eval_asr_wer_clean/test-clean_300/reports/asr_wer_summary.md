# On-the-fly ASR/WER summary

- Whisper: `base.en`
- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | n | WER vs GT ↓ | CER vs GT ↓ | WER vs original Whisper ↓ |
|---|---:|---:|---:|---:|---:|
| base | 1 | 300 | 0.3350 | 0.1908 | 0.3236 |
| base | 2 | 300 | 0.1304 | 0.0658 | 0.1191 |
| base | 3 | 300 | 0.1167 | 0.0618 | 0.0973 |
| lca | 1 | 300 | 0.3089 | 0.1749 | 0.3024 |
| lca | 2 | 300 | 0.1320 | 0.0695 | 0.1192 |
| lca | 3 | 300 | 0.0933 | 0.0441 | 0.0800 |

## LCA - Base deltas

| L | ΔWER vs GT | ΔCER vs GT | ΔWER vs original Whisper |
|---:|---:|---:|---:|
| 1 | -0.0261 | -0.0160 | -0.0212 |
| 2 | +0.0016 | +0.0037 | +0.0001 |
| 3 | -0.0234 | -0.0178 | -0.0174 |
