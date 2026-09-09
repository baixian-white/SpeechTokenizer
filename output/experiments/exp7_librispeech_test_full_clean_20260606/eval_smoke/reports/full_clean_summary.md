# Full LibriSpeech clean evaluation summary

- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 1 | 1.074 | 0.832 | 1.360 | -4.55 |
| base | 2 | 1 | 0.859 | 0.895 | 2.043 | +0.17 |
| base | 3 | 1 | 0.800 | 0.911 | 2.358 | +1.91 |
| lca | 1 | 1 | 1.016 | 0.846 | 1.394 | -5.68 |
| lca | 2 | 1 | 0.817 | 0.900 | 1.982 | -0.59 |
| lca | 3 | 1 | 0.752 | 0.921 | 2.377 | +1.10 |

| L | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---:|---:|---:|---:|
| 1 | -0.058 | +0.014 | +0.034 | -1.13 |
| 2 | -0.042 | +0.005 | -0.061 | -0.75 |
| 3 | -0.048 | +0.010 | +0.018 | -0.81 |
