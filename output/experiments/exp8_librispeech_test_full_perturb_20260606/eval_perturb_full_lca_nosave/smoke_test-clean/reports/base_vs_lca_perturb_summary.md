# Full LibriSpeech perturbation evaluation summary

- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`
- Channel seed: `42`

| model | L | channel | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ | actual_drop | actual_sub |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 1 | 1.0740 | 0.8320 | 1.3601 | -4.554 | 0.0000 | 0.0000 |
| base | 2 | clean | 1 | 0.8591 | 0.8948 | 2.0431 | +0.166 | 0.0000 | 0.0000 |
| base | 3 | clean | 1 | 0.7998 | 0.9111 | 2.3584 | +1.905 | 0.0000 | 0.0000 |
| lca | 1 | clean | 1 | 1.0157 | 0.8461 | 1.3939 | -5.682 | 0.0000 | 0.0000 |
| lca | 2 | clean | 1 | 0.8170 | 0.9002 | 1.9822 | -0.589 | 0.0000 | 0.0000 |
| lca | 3 | clean | 1 | 0.7522 | 0.9212 | 2.3767 | +1.100 | 0.0000 | 0.0000 |

## LCA - Base deltas

Negative mel-L1 means LCA is better; positive STOI/PESQ/SI-SNR means LCA is better.

| L | channel | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---|---:|---:|---:|---:|
| 1 | clean | -0.0583 | +0.0140 | +0.0338 | -1.128 |
| 2 | clean | -0.0421 | +0.0054 | -0.0610 | -0.755 |
| 3 | clean | -0.0476 | +0.0102 | +0.0184 | -0.805 |
