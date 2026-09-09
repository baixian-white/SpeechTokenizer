# Perturbed ASR/WER summary

- Whisper: `base.en`
- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | channel | n | WER vs GT ↓ | CER vs GT ↓ | WER vs original Whisper ↓ | actual drop | actual sub |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | dropout-high | 100 | 0.6690 | 0.4466 | 0.6600 | 0.0981 | 0.0000 |
| base | 1 | substitution-high | 100 | 0.6556 | 0.4179 | 0.6392 | 0.0000 | 0.0297 |
| base | 2 | dropout-high | 100 | 0.5088 | 0.3212 | 0.4898 | 0.0965 | 0.0000 |
| base | 2 | substitution-high | 100 | 0.4810 | 0.2976 | 0.4616 | 0.0000 | 0.0306 |
| base | 3 | dropout-high | 100 | 0.4852 | 0.3113 | 0.4699 | 0.0995 | 0.0000 |
| base | 3 | substitution-high | 100 | 0.3691 | 0.2201 | 0.3396 | 0.0000 | 0.0294 |
| lca | 1 | dropout-high | 100 | 0.5921 | 0.3799 | 0.5756 | 0.0981 | 0.0000 |
| lca | 1 | substitution-high | 100 | 0.6409 | 0.4165 | 0.6187 | 0.0000 | 0.0297 |
| lca | 2 | dropout-high | 100 | 0.4449 | 0.2596 | 0.4221 | 0.0965 | 0.0000 |
| lca | 2 | substitution-high | 100 | 0.4348 | 0.2613 | 0.4069 | 0.0000 | 0.0306 |
| lca | 3 | dropout-high | 100 | 0.4066 | 0.2321 | 0.3690 | 0.0995 | 0.0000 |
| lca | 3 | substitution-high | 100 | 0.3205 | 0.1868 | 0.2908 | 0.0000 | 0.0294 |

## LCA - Base deltas

| L | channel | ΔWER vs GT | ΔCER vs GT | ΔWER vs original Whisper |
|---:|---|---:|---:|---:|
| 1 | dropout-high | -0.0769 | -0.0667 | -0.0844 |
| 1 | substitution-high | -0.0147 | -0.0014 | -0.0205 |
| 2 | dropout-high | -0.0639 | -0.0616 | -0.0677 |
| 2 | substitution-high | -0.0462 | -0.0364 | -0.0547 |
| 3 | dropout-high | -0.0786 | -0.0791 | -0.1008 |
| 3 | substitution-high | -0.0486 | -0.0333 | -0.0489 |
