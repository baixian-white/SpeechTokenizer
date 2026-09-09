# Full LibriSpeech clean evaluation summary

- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 2620 | 1.226 | 0.801 | 1.308 | -7.24 |
| base | 2 | 2620 | 0.976 | 0.864 | 1.726 | -2.01 |
| base | 3 | 2620 | 0.907 | 0.885 | 1.954 | -0.50 |
| lca | 1 | 2620 | 1.205 | 0.813 | 1.331 | -7.60 |
| lca | 2 | 2620 | 0.946 | 0.871 | 1.706 | -2.30 |
| lca | 3 | 2620 | 0.871 | 0.892 | 1.940 | -0.48 |

| L | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---:|---:|---:|---:|
| 1 | -0.021 | +0.013 | +0.023 | -0.36 |
| 2 | -0.030 | +0.007 | -0.019 | -0.29 |
| 3 | -0.036 | +0.007 | -0.014 | +0.01 |
