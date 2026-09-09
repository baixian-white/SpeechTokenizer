# Full LibriSpeech perturbation evaluation summary

- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`
- Channel seed: `42`

| model | L | channel | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ | actual_drop | actual_sub |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 1 | 1.0740 | 0.8320 | 1.3601 | -4.554 | 0.0000 | 0.0000 |
| base | 1 | dropout-high | 1 | 1.1117 | 0.8248 | 1.3010 | -6.150 | 0.0843 | 0.0000 |
| base | 1 | dropout-mid | 1 | 1.1235 | 0.8122 | 1.3014 | -5.316 | 0.0594 | 0.0000 |
| base | 1 | substitution-high | 1 | 1.1115 | 0.8104 | 1.2932 | -5.608 | 0.0000 | 0.0307 |
| base | 1 | substitution-mid | 1 | 1.0740 | 0.8320 | 1.3707 | -4.480 | 0.0000 | 0.0038 |
| base | 2 | clean | 1 | 0.8591 | 0.8948 | 2.0431 | +0.166 | 0.0000 | 0.0000 |
| base | 2 | dropout-high | 1 | 0.9756 | 0.8522 | 1.5664 | -2.804 | 0.0987 | 0.0000 |
| base | 2 | dropout-mid | 1 | 0.9296 | 0.8709 | 1.7365 | -0.676 | 0.0575 | 0.0000 |
| base | 2 | substitution-high | 1 | 0.9305 | 0.8764 | 1.7264 | -0.241 | 0.0000 | 0.0307 |
| base | 2 | substitution-mid | 1 | 0.8786 | 0.8809 | 1.8509 | -0.586 | 0.0000 | 0.0077 |
| base | 3 | clean | 1 | 0.7998 | 0.9111 | 2.3584 | +1.905 | 0.0000 | 0.0000 |
| base | 3 | dropout-high | 1 | 0.9204 | 0.8775 | 1.8220 | +0.134 | 0.1111 | 0.0000 |
| base | 3 | dropout-mid | 1 | 0.8524 | 0.8921 | 2.0306 | +0.508 | 0.0504 | 0.0000 |
| base | 3 | substitution-high | 1 | 0.8924 | 0.8918 | 1.9457 | +1.564 | 0.0000 | 0.0294 |
| base | 3 | substitution-mid | 1 | 0.8189 | 0.9097 | 2.2718 | +1.810 | 0.0000 | 0.0077 |
| lca | 1 | clean | 1 | 1.0157 | 0.8461 | 1.3939 | -5.682 | 0.0000 | 0.0000 |
| lca | 1 | dropout-high | 1 | 1.0609 | 0.8342 | 1.3449 | -7.105 | 0.0843 | 0.0000 |
| lca | 1 | dropout-mid | 1 | 1.0624 | 0.8292 | 1.3526 | -6.185 | 0.0594 | 0.0000 |
| lca | 1 | substitution-high | 1 | 1.0579 | 0.8213 | 1.3054 | -6.086 | 0.0000 | 0.0307 |
| lca | 1 | substitution-mid | 1 | 1.0187 | 0.8447 | 1.3971 | -5.747 | 0.0000 | 0.0038 |
| lca | 2 | clean | 1 | 0.8170 | 0.9002 | 1.9822 | -0.589 | 0.0000 | 0.0000 |
| lca | 2 | dropout-high | 1 | 0.9121 | 0.8702 | 1.5971 | -4.249 | 0.0987 | 0.0000 |
| lca | 2 | dropout-mid | 1 | 0.8812 | 0.8842 | 1.7591 | -0.840 | 0.0575 | 0.0000 |
| lca | 2 | substitution-high | 1 | 0.8947 | 0.8759 | 1.7049 | -1.284 | 0.0000 | 0.0307 |
| lca | 2 | substitution-mid | 1 | 0.8369 | 0.8881 | 1.8348 | -0.394 | 0.0000 | 0.0077 |
| lca | 3 | clean | 1 | 0.7522 | 0.9212 | 2.3767 | +1.100 | 0.0000 | 0.0000 |
| lca | 3 | dropout-high | 1 | 0.8454 | 0.8915 | 1.8454 | -0.241 | 0.1111 | 0.0000 |
| lca | 3 | dropout-mid | 1 | 0.7993 | 0.9037 | 2.0880 | +0.482 | 0.0504 | 0.0000 |
| lca | 3 | substitution-high | 1 | 0.8117 | 0.9016 | 2.0763 | +0.641 | 0.0000 | 0.0294 |
| lca | 3 | substitution-mid | 1 | 0.7694 | 0.9183 | 2.3261 | +0.841 | 0.0000 | 0.0077 |

## LCA - Base deltas

Negative mel-L1 means LCA is better; positive STOI/PESQ/SI-SNR means LCA is better.

| L | channel | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---|---:|---:|---:|---:|
| 1 | clean | -0.0583 | +0.0140 | +0.0338 | -1.128 |
| 1 | dropout-high | -0.0508 | +0.0094 | +0.0439 | -0.955 |
| 1 | dropout-mid | -0.0611 | +0.0171 | +0.0512 | -0.869 |
| 1 | substitution-high | -0.0536 | +0.0109 | +0.0123 | -0.478 |
| 1 | substitution-mid | -0.0554 | +0.0127 | +0.0264 | -1.267 |
| 2 | clean | -0.0421 | +0.0054 | -0.0610 | -0.755 |
| 2 | dropout-high | -0.0635 | +0.0180 | +0.0307 | -1.445 |
| 2 | dropout-mid | -0.0484 | +0.0133 | +0.0227 | -0.163 |
| 2 | substitution-high | -0.0358 | -0.0005 | -0.0214 | -1.044 |
| 2 | substitution-mid | -0.0417 | +0.0072 | -0.0161 | +0.191 |
| 3 | clean | -0.0476 | +0.0102 | +0.0184 | -0.805 |
| 3 | dropout-high | -0.0750 | +0.0140 | +0.0233 | -0.375 |
| 3 | dropout-mid | -0.0531 | +0.0116 | +0.0575 | -0.027 |
| 3 | substitution-high | -0.0807 | +0.0098 | +0.1306 | -0.923 |
| 3 | substitution-mid | -0.0495 | +0.0086 | +0.0543 | -0.969 |
