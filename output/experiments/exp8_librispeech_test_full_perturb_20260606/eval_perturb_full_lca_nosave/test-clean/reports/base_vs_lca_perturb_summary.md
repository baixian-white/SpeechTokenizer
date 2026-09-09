# Full LibriSpeech perturbation evaluation summary

- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`
- Channel seed: `42`

| model | L | channel | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ | actual_drop | actual_sub |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 2620 | 1.2262 | 0.8006 | 1.3075 | -7.240 | 0.0000 | 0.0000 |
| base | 1 | dropout-high | 2620 | 1.2861 | 0.7719 | 1.2217 | -8.973 | 0.0991 | 0.0000 |
| base | 1 | dropout-mid | 2620 | 1.2572 | 0.7858 | 1.2560 | -8.136 | 0.0497 | 0.0000 |
| base | 1 | substitution-high | 2620 | 1.2764 | 0.7822 | 1.2564 | -7.621 | 0.0000 | 0.0302 |
| base | 1 | substitution-mid | 2620 | 1.2424 | 0.7946 | 1.2887 | -7.379 | 0.0000 | 0.0100 |
| base | 2 | clean | 2620 | 0.9757 | 0.8642 | 1.7257 | -2.012 | 0.0000 | 0.0000 |
| base | 2 | dropout-high | 2620 | 1.0585 | 0.8271 | 1.4268 | -4.339 | 0.0998 | 0.0000 |
| base | 2 | dropout-mid | 2620 | 1.0180 | 0.8453 | 1.5399 | -3.200 | 0.0495 | 0.0000 |
| base | 2 | substitution-high | 2620 | 1.0367 | 0.8433 | 1.5449 | -2.621 | 0.0000 | 0.0300 |
| base | 2 | substitution-mid | 2620 | 0.9955 | 0.8571 | 1.6562 | -2.218 | 0.0000 | 0.0101 |
| base | 3 | clean | 2620 | 0.9075 | 0.8853 | 1.9538 | -0.496 | 0.0000 | 0.0000 |
| base | 3 | dropout-high | 2620 | 0.9986 | 0.8457 | 1.5364 | -3.194 | 0.1000 | 0.0000 |
| base | 3 | dropout-mid | 2620 | 0.9544 | 0.8653 | 1.6933 | -1.843 | 0.0496 | 0.0000 |
| base | 3 | substitution-high | 2620 | 0.9716 | 0.8640 | 1.7027 | -1.168 | 0.0000 | 0.0301 |
| base | 3 | substitution-mid | 2620 | 0.9286 | 0.8783 | 1.8584 | -0.731 | 0.0000 | 0.0100 |
| lca | 1 | clean | 2620 | 1.2054 | 0.8132 | 1.3305 | -7.600 | 0.0000 | 0.0000 |
| lca | 1 | dropout-high | 2620 | 1.2559 | 0.7887 | 1.2463 | -9.280 | 0.0991 | 0.0000 |
| lca | 1 | dropout-mid | 2620 | 1.2314 | 0.8005 | 1.2805 | -8.440 | 0.0497 | 0.0000 |
| lca | 1 | substitution-high | 2620 | 1.2445 | 0.7956 | 1.2809 | -8.014 | 0.0000 | 0.0302 |
| lca | 1 | substitution-mid | 2620 | 1.2180 | 0.8075 | 1.3127 | -7.743 | 0.0000 | 0.0100 |
| lca | 2 | clean | 2620 | 0.9461 | 0.8712 | 1.7063 | -2.300 | 0.0000 | 0.0000 |
| lca | 2 | dropout-high | 2620 | 1.0181 | 0.8393 | 1.4400 | -4.506 | 0.0998 | 0.0000 |
| lca | 2 | dropout-mid | 2620 | 0.9825 | 0.8552 | 1.5412 | -3.423 | 0.0495 | 0.0000 |
| lca | 2 | substitution-high | 2620 | 0.9973 | 0.8517 | 1.5460 | -2.945 | 0.0000 | 0.0300 |
| lca | 2 | substitution-mid | 2620 | 0.9626 | 0.8647 | 1.6437 | -2.509 | 0.0000 | 0.0101 |
| lca | 3 | clean | 2620 | 0.8711 | 0.8922 | 1.9397 | -0.482 | 0.0000 | 0.0000 |
| lca | 3 | dropout-high | 2620 | 0.9511 | 0.8580 | 1.5549 | -3.061 | 0.1000 | 0.0000 |
| lca | 3 | dropout-mid | 2620 | 0.9117 | 0.8751 | 1.7033 | -1.760 | 0.0496 | 0.0000 |
| lca | 3 | substitution-high | 2620 | 0.9266 | 0.8723 | 1.7150 | -1.198 | 0.0000 | 0.0301 |
| lca | 3 | substitution-mid | 2620 | 0.8890 | 0.8857 | 1.8529 | -0.721 | 0.0000 | 0.0100 |

## LCA - Base deltas

Negative mel-L1 means LCA is better; positive STOI/PESQ/SI-SNR means LCA is better.

| L | channel | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---|---:|---:|---:|---:|
| 1 | clean | -0.0207 | +0.0125 | +0.0230 | -0.360 |
| 1 | dropout-high | -0.0302 | +0.0168 | +0.0246 | -0.307 |
| 1 | dropout-mid | -0.0258 | +0.0148 | +0.0245 | -0.304 |
| 1 | substitution-high | -0.0319 | +0.0134 | +0.0245 | -0.393 |
| 1 | substitution-mid | -0.0244 | +0.0128 | +0.0240 | -0.364 |
| 2 | clean | -0.0296 | +0.0070 | -0.0194 | -0.288 |
| 2 | dropout-high | -0.0405 | +0.0122 | +0.0132 | -0.167 |
| 2 | dropout-mid | -0.0354 | +0.0099 | +0.0013 | -0.224 |
| 2 | substitution-high | -0.0393 | +0.0084 | +0.0012 | -0.323 |
| 2 | substitution-mid | -0.0329 | +0.0076 | -0.0126 | -0.291 |
| 3 | clean | -0.0364 | +0.0069 | -0.0141 | +0.014 |
| 3 | dropout-high | -0.0475 | +0.0123 | +0.0185 | +0.133 |
| 3 | dropout-mid | -0.0427 | +0.0097 | +0.0100 | +0.083 |
| 3 | substitution-high | -0.0451 | +0.0082 | +0.0122 | -0.030 |
| 3 | substitution-mid | -0.0396 | +0.0075 | -0.0055 | +0.009 |
