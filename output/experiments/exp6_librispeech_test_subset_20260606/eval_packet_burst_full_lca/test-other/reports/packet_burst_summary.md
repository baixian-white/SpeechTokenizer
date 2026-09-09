# Packet/Burst Evaluation Summary

- Base: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
- LCA: `output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42\checkpoints\SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | condition | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ | actual loss |
|---|---:|---|---:|---:|---:|---:|---:|
| base | 1 | burst-10f | 1.402 | 0.760 | 1.251 | -9.58 | 0.041 |
| base | 1 | burst-2f | 1.394 | 0.765 | 1.254 | -8.92 | 0.008 |
| base | 1 | burst-5f | 1.397 | 0.762 | 1.253 | -9.17 | 0.020 |
| base | 1 | clean | 1.392 | 0.767 | 1.260 | -8.78 | 0.000 |
| base | 1 | packet-loss-1p | 1.396 | 0.764 | 1.254 | -8.95 | 0.013 |
| base | 1 | packet-loss-3p | 1.402 | 0.760 | 1.247 | -9.24 | 0.032 |
| base | 1 | packet-loss-5p | 1.407 | 0.759 | 1.246 | -9.46 | 0.049 |
| base | 2 | burst-10f | 1.152 | 0.820 | 1.536 | -3.86 | 0.041 |
| base | 2 | burst-2f | 1.139 | 0.827 | 1.569 | -3.31 | 0.008 |
| base | 2 | burst-5f | 1.144 | 0.824 | 1.553 | -3.51 | 0.020 |
| base | 2 | clean | 1.136 | 0.829 | 1.578 | -3.19 | 0.000 |
| base | 2 | packet-loss-1p | 1.138 | 0.827 | 1.563 | -3.31 | 0.007 |
| base | 2 | packet-loss-3p | 1.147 | 0.822 | 1.538 | -3.64 | 0.032 |
| base | 2 | packet-loss-5p | 1.157 | 0.817 | 1.514 | -4.51 | 0.052 |
| base | 3 | burst-10f | 1.077 | 0.844 | 1.719 | -2.50 | 0.041 |
| base | 3 | burst-2f | 1.066 | 0.849 | 1.748 | -1.85 | 0.008 |
| base | 3 | burst-5f | 1.071 | 0.848 | 1.740 | -2.03 | 0.020 |
| base | 3 | clean | 1.062 | 0.852 | 1.774 | -1.65 | 0.000 |
| base | 3 | packet-loss-1p | 1.066 | 0.850 | 1.757 | -1.86 | 0.010 |
| base | 3 | packet-loss-3p | 1.074 | 0.846 | 1.714 | -2.12 | 0.031 |
| base | 3 | packet-loss-5p | 1.080 | 0.843 | 1.688 | -2.36 | 0.046 |
| lca | 1 | burst-10f | 1.402 | 0.774 | 1.271 | -9.59 | 0.041 |
| lca | 1 | burst-2f | 1.393 | 0.779 | 1.274 | -9.26 | 0.008 |
| lca | 1 | burst-5f | 1.397 | 0.775 | 1.272 | -9.29 | 0.020 |
| lca | 1 | clean | 1.391 | 0.780 | 1.279 | -9.13 | 0.000 |
| lca | 1 | packet-loss-1p | 1.395 | 0.778 | 1.274 | -9.25 | 0.013 |
| lca | 1 | packet-loss-3p | 1.401 | 0.774 | 1.268 | -9.62 | 0.032 |
| lca | 1 | packet-loss-5p | 1.405 | 0.773 | 1.265 | -9.51 | 0.049 |
| lca | 2 | burst-10f | 1.133 | 0.830 | 1.530 | -4.03 | 0.041 |
| lca | 2 | burst-2f | 1.122 | 0.837 | 1.553 | -3.53 | 0.008 |
| lca | 2 | burst-5f | 1.125 | 0.834 | 1.542 | -3.70 | 0.020 |
| lca | 2 | clean | 1.118 | 0.838 | 1.564 | -3.44 | 0.000 |
| lca | 2 | packet-loss-1p | 1.120 | 0.837 | 1.550 | -3.58 | 0.007 |
| lca | 2 | packet-loss-3p | 1.128 | 0.832 | 1.530 | -3.86 | 0.032 |
| lca | 2 | packet-loss-5p | 1.137 | 0.827 | 1.503 | -4.35 | 0.052 |
| lca | 3 | burst-10f | 1.049 | 0.853 | 1.695 | -2.33 | 0.041 |
| lca | 3 | burst-2f | 1.038 | 0.858 | 1.718 | -1.71 | 0.008 |
| lca | 3 | burst-5f | 1.043 | 0.856 | 1.706 | -1.91 | 0.020 |
| lca | 3 | clean | 1.035 | 0.860 | 1.743 | -1.51 | 0.000 |
| lca | 3 | packet-loss-1p | 1.038 | 0.858 | 1.724 | -1.74 | 0.010 |
| lca | 3 | packet-loss-3p | 1.046 | 0.854 | 1.700 | -2.07 | 0.031 |
| lca | 3 | packet-loss-5p | 1.053 | 0.851 | 1.674 | -2.20 | 0.046 |
