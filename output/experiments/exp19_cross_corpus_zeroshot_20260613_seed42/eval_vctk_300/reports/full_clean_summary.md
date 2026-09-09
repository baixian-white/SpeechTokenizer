# Full LibriSpeech clean evaluation summary

- Base: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
- LCA: `output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42\checkpoints\SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 300 | 1.288 | 0.759 | 1.477 | -5.49 |
| base | 2 | 300 | 1.029 | 0.805 | 1.903 | +0.71 |
| base | 3 | 300 | 0.978 | 0.821 | 2.106 | +2.17 |
| lca | 1 | 300 | 1.534 | 0.763 | 1.567 | -5.71 |
| lca | 2 | 300 | 1.055 | 0.809 | 1.919 | +0.16 |
| lca | 3 | 300 | 0.919 | 0.825 | 2.091 | +1.87 |

| L | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---:|---:|---:|---:|
| 1 | +0.246 | +0.004 | +0.090 | -0.22 |
| 2 | +0.026 | +0.004 | +0.016 | -0.56 |
| 3 | -0.059 | +0.004 | -0.015 | -0.30 |
