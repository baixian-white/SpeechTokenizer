# Full LibriSpeech perturbation evaluation summary

- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`
- Channel seed: `42`

| model | L | channel | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ | actual_drop | actual_sub |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 2939 | 1.3966 | 0.7694 | 1.2603 | -9.065 | 0.0000 | 0.0000 |
| base | 1 | dropout-high | 2939 | 1.4420 | 0.7441 | 1.2032 | -10.690 | 0.0993 | 0.0000 |
| base | 1 | dropout-mid | 2939 | 1.4197 | 0.7565 | 1.2263 | -9.927 | 0.0497 | 0.0000 |
| base | 1 | substitution-high | 2939 | 1.4544 | 0.7525 | 1.2221 | -9.409 | 0.0000 | 0.0298 |
| base | 1 | substitution-mid | 2939 | 1.4168 | 0.7637 | 1.2465 | -9.197 | 0.0000 | 0.0100 |
| base | 2 | clean | 2939 | 1.1455 | 0.8317 | 1.5768 | -3.365 | 0.0000 | 0.0000 |
| base | 2 | dropout-high | 2939 | 1.2105 | 0.7977 | 1.3837 | -5.600 | 0.0997 | 0.0000 |
| base | 2 | dropout-mid | 2939 | 1.1787 | 0.8145 | 1.4597 | -4.521 | 0.0499 | 0.0000 |
| base | 2 | substitution-high | 2939 | 1.2143 | 0.8119 | 1.4516 | -3.925 | 0.0000 | 0.0299 |
| base | 2 | substitution-mid | 2939 | 1.1687 | 0.8247 | 1.5274 | -3.537 | 0.0000 | 0.0100 |
| base | 3 | clean | 2939 | 1.0722 | 0.8545 | 1.7605 | -1.724 | 0.0000 | 0.0000 |
| base | 3 | dropout-high | 2939 | 1.1448 | 0.8177 | 1.4847 | -4.205 | 0.0997 | 0.0000 |
| base | 3 | dropout-mid | 2939 | 1.1096 | 0.8355 | 1.5899 | -3.005 | 0.0498 | 0.0000 |
| base | 3 | substitution-high | 2939 | 1.1423 | 0.8334 | 1.5803 | -2.349 | 0.0000 | 0.0301 |
| base | 3 | substitution-mid | 2939 | 1.0952 | 0.8476 | 1.6924 | -1.919 | 0.0000 | 0.0099 |
| lca | 1 | clean | 2939 | 1.3840 | 0.7820 | 1.2842 | -9.371 | 0.0000 | 0.0000 |
| lca | 1 | dropout-high | 2939 | 1.4236 | 0.7604 | 1.2255 | -11.000 | 0.0993 | 0.0000 |
| lca | 1 | dropout-mid | 2939 | 1.4037 | 0.7711 | 1.2498 | -10.167 | 0.0497 | 0.0000 |
| lca | 1 | substitution-high | 2939 | 1.4262 | 0.7658 | 1.2480 | -9.794 | 0.0000 | 0.0298 |
| lca | 1 | substitution-mid | 2939 | 1.3984 | 0.7767 | 1.2717 | -9.487 | 0.0000 | 0.0100 |
| lca | 2 | clean | 2939 | 1.1159 | 0.8393 | 1.5667 | -3.730 | 0.0000 | 0.0000 |
| lca | 2 | dropout-high | 2939 | 1.1730 | 0.8102 | 1.3899 | -5.868 | 0.0997 | 0.0000 |
| lca | 2 | dropout-mid | 2939 | 1.1450 | 0.8246 | 1.4610 | -4.818 | 0.0499 | 0.0000 |
| lca | 2 | substitution-high | 2939 | 1.1711 | 0.8209 | 1.4601 | -4.332 | 0.0000 | 0.0299 |
| lca | 2 | substitution-mid | 2939 | 1.1339 | 0.8329 | 1.5260 | -3.921 | 0.0000 | 0.0100 |
| lca | 3 | clean | 2939 | 1.0392 | 0.8617 | 1.7457 | -1.753 | 0.0000 | 0.0000 |
| lca | 3 | dropout-high | 2939 | 1.1036 | 0.8301 | 1.4922 | -4.134 | 0.0997 | 0.0000 |
| lca | 3 | dropout-mid | 2939 | 1.0721 | 0.8457 | 1.5938 | -2.949 | 0.0498 | 0.0000 |
| lca | 3 | substitution-high | 2939 | 1.0994 | 0.8419 | 1.5890 | -2.421 | 0.0000 | 0.0301 |
| lca | 3 | substitution-mid | 2939 | 1.0585 | 0.8553 | 1.6867 | -1.979 | 0.0000 | 0.0099 |

## LCA - Base deltas

Negative mel-L1 means LCA is better; positive STOI/PESQ/SI-SNR means LCA is better.

| L | channel | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---|---:|---:|---:|---:|
| 1 | clean | -0.0127 | +0.0126 | +0.0239 | -0.306 |
| 1 | dropout-high | -0.0184 | +0.0163 | +0.0223 | -0.310 |
| 1 | dropout-mid | -0.0161 | +0.0145 | +0.0235 | -0.240 |
| 1 | substitution-high | -0.0283 | +0.0133 | +0.0259 | -0.384 |
| 1 | substitution-mid | -0.0185 | +0.0130 | +0.0251 | -0.290 |
| 2 | clean | -0.0296 | +0.0076 | -0.0101 | -0.365 |
| 2 | dropout-high | -0.0375 | +0.0125 | +0.0062 | -0.269 |
| 2 | dropout-mid | -0.0337 | +0.0101 | +0.0012 | -0.297 |
| 2 | substitution-high | -0.0432 | +0.0090 | +0.0085 | -0.407 |
| 2 | substitution-mid | -0.0349 | +0.0082 | -0.0014 | -0.384 |
| 3 | clean | -0.0330 | +0.0072 | -0.0147 | -0.029 |
| 3 | dropout-high | -0.0412 | +0.0124 | +0.0075 | +0.070 |
| 3 | dropout-mid | -0.0375 | +0.0102 | +0.0039 | +0.056 |
| 3 | substitution-high | -0.0429 | +0.0085 | +0.0087 | -0.072 |
| 3 | substitution-mid | -0.0368 | +0.0077 | -0.0057 | -0.060 |
