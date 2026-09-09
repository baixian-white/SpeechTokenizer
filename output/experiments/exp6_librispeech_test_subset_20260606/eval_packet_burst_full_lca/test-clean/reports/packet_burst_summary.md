# Packet/Burst Evaluation Summary

- Base: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
- LCA: `output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42\checkpoints\SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

| model | L | condition | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ | actual loss |
|---|---:|---|---:|---:|---:|---:|---:|
| base | 1 | burst-10f | 1.258 | 0.790 | 1.275 | -8.25 | 0.038 |
| base | 1 | burst-2f | 1.247 | 0.797 | 1.281 | -7.87 | 0.008 |
| base | 1 | burst-5f | 1.251 | 0.793 | 1.276 | -8.20 | 0.019 |
| base | 1 | clean | 1.243 | 0.798 | 1.286 | -7.74 | 0.000 |
| base | 1 | packet-loss-1p | 1.246 | 0.797 | 1.284 | -7.85 | 0.006 |
| base | 1 | packet-loss-3p | 1.256 | 0.792 | 1.273 | -8.03 | 0.029 |
| base | 1 | packet-loss-5p | 1.262 | 0.788 | 1.268 | -8.32 | 0.049 |
| base | 2 | burst-10f | 1.006 | 0.853 | 1.621 | -3.00 | 0.038 |
| base | 2 | burst-2f | 0.994 | 0.860 | 1.650 | -2.56 | 0.008 |
| base | 2 | burst-5f | 0.999 | 0.857 | 1.643 | -2.72 | 0.019 |
| base | 2 | clean | 0.990 | 0.862 | 1.674 | -2.42 | 0.000 |
| base | 2 | packet-loss-1p | 0.996 | 0.859 | 1.652 | -2.65 | 0.010 |
| base | 2 | packet-loss-3p | 1.004 | 0.855 | 1.619 | -2.87 | 0.027 |
| base | 2 | packet-loss-5p | 1.015 | 0.849 | 1.582 | -3.28 | 0.048 |
| base | 3 | burst-10f | 0.939 | 0.873 | 1.833 | -1.74 | 0.038 |
| base | 3 | burst-2f | 0.926 | 0.882 | 1.869 | -1.16 | 0.008 |
| base | 3 | burst-5f | 0.931 | 0.877 | 1.839 | -1.44 | 0.019 |
| base | 3 | clean | 0.921 | 0.884 | 1.902 | -0.99 | 0.000 |
| base | 3 | packet-loss-1p | 0.927 | 0.880 | 1.869 | -1.14 | 0.010 |
| base | 3 | packet-loss-3p | 0.937 | 0.877 | 1.814 | -1.67 | 0.029 |
| base | 3 | packet-loss-5p | 0.948 | 0.870 | 1.772 | -2.43 | 0.052 |
| lca | 1 | burst-10f | 1.233 | 0.803 | 1.295 | -8.67 | 0.038 |
| lca | 1 | burst-2f | 1.223 | 0.810 | 1.299 | -8.15 | 0.008 |
| lca | 1 | burst-5f | 1.227 | 0.806 | 1.297 | -8.42 | 0.019 |
| lca | 1 | clean | 1.220 | 0.811 | 1.305 | -8.06 | 0.000 |
| lca | 1 | packet-loss-1p | 1.222 | 0.809 | 1.302 | -8.18 | 0.006 |
| lca | 1 | packet-loss-3p | 1.231 | 0.806 | 1.293 | -8.37 | 0.029 |
| lca | 1 | packet-loss-5p | 1.236 | 0.802 | 1.285 | -8.77 | 0.049 |
| lca | 2 | burst-10f | 0.974 | 0.860 | 1.610 | -3.22 | 0.038 |
| lca | 2 | burst-2f | 0.961 | 0.867 | 1.633 | -2.81 | 0.008 |
| lca | 2 | burst-5f | 0.966 | 0.864 | 1.626 | -2.93 | 0.019 |
| lca | 2 | clean | 0.958 | 0.869 | 1.655 | -2.68 | 0.000 |
| lca | 2 | packet-loss-1p | 0.963 | 0.866 | 1.638 | -2.88 | 0.010 |
| lca | 2 | packet-loss-3p | 0.971 | 0.862 | 1.609 | -3.14 | 0.027 |
| lca | 2 | packet-loss-5p | 0.983 | 0.857 | 1.578 | -3.53 | 0.048 |
| lca | 3 | burst-10f | 0.901 | 0.881 | 1.819 | -1.64 | 0.038 |
| lca | 3 | burst-2f | 0.886 | 0.889 | 1.854 | -1.03 | 0.008 |
| lca | 3 | burst-5f | 0.892 | 0.885 | 1.824 | -1.30 | 0.019 |
| lca | 3 | clean | 0.882 | 0.890 | 1.882 | -0.86 | 0.000 |
| lca | 3 | packet-loss-1p | 0.888 | 0.888 | 1.856 | -1.00 | 0.010 |
| lca | 3 | packet-loss-3p | 0.898 | 0.884 | 1.805 | -1.53 | 0.029 |
| lca | 3 | packet-loss-5p | 0.908 | 0.877 | 1.765 | -2.11 | 0.052 |
