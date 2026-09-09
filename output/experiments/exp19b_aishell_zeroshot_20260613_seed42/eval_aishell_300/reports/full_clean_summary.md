# Full LibriSpeech clean evaluation summary

- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 300 | 1.271 | 0.682 | 1.146 | -8.45 |
| base | 2 | 300 | 1.045 | 0.756 | 1.343 | -3.22 |
| base | 3 | 300 | 0.983 | 0.784 | 1.492 | -1.46 |
| lca | 1 | 300 | 1.293 | 0.698 | 1.171 | -10.02 |
| lca | 2 | 300 | 1.020 | 0.768 | 1.373 | -3.72 |
| lca | 3 | 300 | 0.941 | 0.795 | 1.521 | -1.52 |

| L | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---:|---:|---:|---:|
| 1 | +0.023 | +0.016 | +0.025 | -1.57 |
| 2 | -0.026 | +0.012 | +0.029 | -0.50 |
| 3 | -0.042 | +0.011 | +0.029 | -0.07 |
