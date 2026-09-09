# Full LibriSpeech clean evaluation summary

- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 2939 | 1.397 | 0.769 | 1.260 | -9.07 |
| base | 2 | 2939 | 1.146 | 0.832 | 1.577 | -3.37 |
| base | 3 | 2939 | 1.072 | 0.854 | 1.760 | -1.72 |
| lca | 1 | 2939 | 1.384 | 0.782 | 1.284 | -9.37 |
| lca | 2 | 2939 | 1.116 | 0.839 | 1.567 | -3.73 |
| lca | 3 | 2939 | 1.039 | 0.862 | 1.746 | -1.75 |

| L | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---:|---:|---:|---:|
| 1 | -0.013 | +0.013 | +0.024 | -0.31 |
| 2 | -0.030 | +0.008 | -0.010 | -0.36 |
| 3 | -0.033 | +0.007 | -0.015 | -0.03 |
