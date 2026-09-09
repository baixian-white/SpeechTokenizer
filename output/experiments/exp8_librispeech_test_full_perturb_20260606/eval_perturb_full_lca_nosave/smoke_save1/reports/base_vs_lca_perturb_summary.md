# Full LibriSpeech perturbation evaluation summary

- Base: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- LCA: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`
- Channel seed: `42`

| model | L | channel | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ | actual_drop | actual_sub |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 1 | 1.0740 | 0.8320 | 1.3601 | -4.554 | 0.0000 | 0.0000 |
| base | 1 | dropout-high | 1 | 1.1585 | 0.8008 | 1.2631 | -6.950 | 0.1015 | 0.0000 |
| base | 1 | dropout-mid | 1 | 1.1145 | 0.8167 | 1.3061 | -6.142 | 0.0556 | 0.0000 |
| base | 1 | substitution-high | 1 | 1.0936 | 0.8200 | 1.3250 | -4.851 | 0.0000 | 0.0172 |
| base | 1 | substitution-mid | 1 | 1.0789 | 0.8302 | 1.3516 | -4.600 | 0.0000 | 0.0057 |
| base | 2 | clean | 1 | 0.8591 | 0.8948 | 2.0431 | +0.166 | 0.0000 | 0.0000 |
| base | 2 | dropout-high | 1 | 0.9534 | 0.8627 | 1.6662 | -0.891 | 0.0872 | 0.0000 |
| base | 2 | dropout-mid | 1 | 0.8997 | 0.8788 | 1.8977 | -0.653 | 0.0584 | 0.0000 |
| base | 2 | substitution-high | 1 | 0.8940 | 0.8814 | 1.8655 | -1.011 | 0.0000 | 0.0259 |
| base | 2 | substitution-mid | 1 | 0.8752 | 0.8794 | 1.8846 | -0.198 | 0.0000 | 0.0125 |
| base | 3 | clean | 1 | 0.7998 | 0.9111 | 2.3584 | +1.905 | 0.0000 | 0.0000 |
| base | 3 | dropout-high | 1 | 0.8995 | 0.8726 | 1.7955 | -1.101 | 0.1003 | 0.0000 |
| base | 3 | dropout-mid | 1 | 0.8337 | 0.9016 | 2.1607 | +1.586 | 0.0390 | 0.0000 |
| base | 3 | substitution-high | 1 | 0.8641 | 0.8909 | 2.0708 | +1.302 | 0.0000 | 0.0294 |
| base | 3 | substitution-mid | 1 | 0.8240 | 0.9044 | 2.1984 | +1.722 | 0.0000 | 0.0121 |
| lca | 1 | clean | 1 | 1.0157 | 0.8461 | 1.3939 | -5.682 | 0.0000 | 0.0000 |
| lca | 1 | dropout-high | 1 | 1.0865 | 0.8183 | 1.3040 | -8.814 | 0.1015 | 0.0000 |
| lca | 1 | dropout-mid | 1 | 1.0464 | 0.8388 | 1.3643 | -7.436 | 0.0556 | 0.0000 |
| lca | 1 | substitution-high | 1 | 1.0373 | 0.8367 | 1.3510 | -5.970 | 0.0000 | 0.0172 |
| lca | 1 | substitution-mid | 1 | 1.0195 | 0.8440 | 1.3938 | -5.686 | 0.0000 | 0.0057 |
| lca | 2 | clean | 1 | 0.8170 | 0.9002 | 1.9822 | -0.589 | 0.0000 | 0.0000 |
| lca | 2 | dropout-high | 1 | 0.9114 | 0.8733 | 1.6232 | -1.474 | 0.0872 | 0.0000 |
| lca | 2 | dropout-mid | 1 | 0.8565 | 0.8914 | 1.8713 | -1.179 | 0.0584 | 0.0000 |
| lca | 2 | substitution-high | 1 | 0.8510 | 0.8892 | 1.8382 | -2.200 | 0.0000 | 0.0259 |
| lca | 2 | substitution-mid | 1 | 0.8364 | 0.8834 | 1.8881 | -1.266 | 0.0000 | 0.0125 |
| lca | 3 | clean | 1 | 0.7522 | 0.9212 | 2.3767 | +1.100 | 0.0000 | 0.0000 |
| lca | 3 | dropout-high | 1 | 0.8326 | 0.8941 | 1.8332 | -1.768 | 0.1003 | 0.0000 |
| lca | 3 | dropout-mid | 1 | 0.7883 | 0.9128 | 2.1204 | +0.769 | 0.0390 | 0.0000 |
| lca | 3 | substitution-high | 1 | 0.8034 | 0.9102 | 2.1679 | +0.797 | 0.0000 | 0.0294 |
| lca | 3 | substitution-mid | 1 | 0.7719 | 0.9102 | 2.2498 | +0.963 | 0.0000 | 0.0121 |

## LCA - Base deltas

Negative mel-L1 means LCA is better; positive STOI/PESQ/SI-SNR means LCA is better.

| L | channel | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---|---:|---:|---:|---:|
| 1 | clean | -0.0583 | +0.0140 | +0.0338 | -1.128 |
| 1 | dropout-high | -0.0720 | +0.0175 | +0.0409 | -1.864 |
| 1 | dropout-mid | -0.0681 | +0.0222 | +0.0582 | -1.294 |
| 1 | substitution-high | -0.0563 | +0.0167 | +0.0260 | -1.120 |
| 1 | substitution-mid | -0.0594 | +0.0138 | +0.0422 | -1.086 |
| 2 | clean | -0.0421 | +0.0054 | -0.0610 | -0.755 |
| 2 | dropout-high | -0.0420 | +0.0107 | -0.0431 | -0.582 |
| 2 | dropout-mid | -0.0431 | +0.0126 | -0.0264 | -0.526 |
| 2 | substitution-high | -0.0431 | +0.0079 | -0.0272 | -1.190 |
| 2 | substitution-mid | -0.0388 | +0.0039 | +0.0034 | -1.068 |
| 3 | clean | -0.0476 | +0.0102 | +0.0184 | -0.805 |
| 3 | dropout-high | -0.0669 | +0.0216 | +0.0377 | -0.668 |
| 3 | dropout-mid | -0.0454 | +0.0111 | -0.0403 | -0.817 |
| 3 | substitution-high | -0.0606 | +0.0192 | +0.0972 | -0.505 |
| 3 | substitution-mid | -0.0521 | +0.0058 | +0.0515 | -0.758 |
