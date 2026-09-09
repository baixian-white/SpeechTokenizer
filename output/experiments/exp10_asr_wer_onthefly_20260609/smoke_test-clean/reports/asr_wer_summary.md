# On-the-fly ASR/WER summary

- Whisper: `base.en`
- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | n | WER vs GT ↓ | CER vs GT ↓ | WER vs original Whisper ↓ |
|---|---:|---:|---:|---:|---:|
| base | 1 | 1 | 0.1786 | 0.0759 | 0.1852 |
| base | 2 | 1 | 0.0714 | 0.0380 | 0.0741 |
| base | 3 | 1 | 0.0714 | 0.0380 | 0.0370 |
| lca | 1 | 1 | 0.1786 | 0.0823 | 0.2222 |
| lca | 2 | 1 | 0.0714 | 0.0253 | 0.0000 |
| lca | 3 | 1 | 0.0714 | 0.0253 | 0.0741 |

## LCA - Base deltas

| L | ΔWER vs GT | ΔCER vs GT | ΔWER vs original Whisper |
|---:|---:|---:|---:|
| 1 | +0.0000 | +0.0063 | +0.0370 |
| 2 | +0.0000 | -0.0127 | -0.0741 |
| 3 | +0.0000 | -0.0127 | +0.0370 |
