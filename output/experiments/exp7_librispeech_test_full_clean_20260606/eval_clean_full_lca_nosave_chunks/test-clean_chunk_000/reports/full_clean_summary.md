# Full LibriSpeech clean evaluation summary

- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 500 | 1.148 | 0.804 | 1.322 | -7.68 |
| base | 2 | 500 | 0.925 | 0.869 | 1.775 | -2.09 |
| base | 3 | 500 | 0.866 | 0.889 | 2.026 | -0.60 |
| lca | 1 | 500 | 1.123 | 0.815 | 1.350 | -7.71 |
| lca | 2 | 500 | 0.898 | 0.875 | 1.760 | -2.28 |
| lca | 3 | 500 | 0.834 | 0.895 | 2.016 | -0.40 |

| L | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---:|---:|---:|---:|
| 1 | -0.025 | +0.011 | +0.028 | -0.03 |
| 2 | -0.027 | +0.005 | -0.015 | -0.19 |
| 3 | -0.032 | +0.006 | -0.009 | +0.20 |
