# LibriSpeech test subset evaluation summary

- run: `exp6_librispeech_test_subset_20260606`
- data: `test-clean` 100 samples and `test-other` 100 samples sampled with seed 42
- models: SCIT-Speech-Base vs full LCA (`SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`)
- metrics: intrusive objective metrics; WER/CER not included

## 1. Clean Held-Out Test Subset

### test-clean

| Model | L | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 100 | 1.243 | 0.798 | 1.286 | -7.74 |
| base | 2 | 100 | 0.990 | 0.862 | 1.674 | -2.42 |
| base | 3 | 100 | 0.921 | 0.884 | 1.902 | -0.99 |
| lca | 1 | 100 | 1.220 | 0.811 | 1.305 | -8.06 |
| lca | 2 | 100 | 0.958 | 0.869 | 1.655 | -2.68 |
| lca | 3 | 100 | 0.882 | 0.890 | 1.882 | -0.86 |

| L | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---:|---:|---:|---:|
| 1 | -0.024 | +0.013 | +0.019 | -0.33 |
| 2 | -0.032 | +0.007 | -0.019 | -0.26 |
| 3 | -0.038 | +0.007 | -0.020 | +0.12 |

### test-other

| Model | L | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 100 | 1.392 | 0.767 | 1.260 | -8.78 |
| base | 2 | 100 | 1.136 | 0.829 | 1.578 | -3.19 |
| base | 3 | 100 | 1.062 | 0.852 | 1.774 | -1.65 |
| lca | 1 | 100 | 1.391 | 0.780 | 1.279 | -9.13 |
| lca | 2 | 100 | 1.118 | 0.838 | 1.564 | -3.44 |
| lca | 3 | 100 | 1.035 | 0.860 | 1.743 | -1.51 |

| L | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |
|---:|---:|---:|---:|---:|
| 1 | -0.001 | +0.014 | +0.019 | -0.35 |
| 2 | -0.018 | +0.010 | -0.014 | -0.25 |
| 3 | -0.027 | +0.008 | -0.031 | +0.14 |

## 2. Index-Level Dropout/Substitution Robustness

Values are mel-L1 degradation from clean. `robust_imp` is Base degradation minus LCA degradation; positive means LCA is more robust.

### test-clean

| L | Channel | Base deg | LCA deg | robust_imp | Base mel | LCA mel |
|---:|---|---:|---:|---:|---:|---:|
| 1 | dropout-mid | +0.030 | +0.025 | +0.005 | 1.273 | 1.244 |
| 1 | dropout-high | +0.059 | +0.050 | +0.009 | 1.303 | 1.270 |
| 1 | substitution-mid | +0.015 | +0.011 | +0.004 | 1.259 | 1.231 |
| 1 | substitution-high | +0.052 | +0.040 | +0.012 | 1.295 | 1.260 |
| 2 | dropout-mid | +0.044 | +0.038 | +0.006 | 1.034 | 0.996 |
| 2 | dropout-high | +0.083 | +0.073 | +0.010 | 1.072 | 1.030 |
| 2 | substitution-mid | +0.020 | +0.018 | +0.002 | 1.010 | 0.975 |
| 2 | substitution-high | +0.057 | +0.053 | +0.005 | 1.047 | 1.010 |
| 3 | dropout-mid | +0.049 | +0.043 | +0.006 | 0.970 | 0.925 |
| 3 | dropout-high | +0.093 | +0.083 | +0.010 | 1.014 | 0.965 |
| 3 | substitution-mid | +0.020 | +0.018 | +0.002 | 0.941 | 0.900 |
| 3 | substitution-high | +0.060 | +0.052 | +0.008 | 0.981 | 0.934 |

### test-other

| L | Channel | Base deg | LCA deg | robust_imp | Base mel | LCA mel |
|---:|---|---:|---:|---:|---:|---:|
| 1 | dropout-mid | +0.023 | +0.019 | +0.004 | 1.414 | 1.410 |
| 1 | dropout-high | +0.043 | +0.039 | +0.004 | 1.435 | 1.430 |
| 1 | substitution-mid | +0.014 | +0.011 | +0.002 | 1.405 | 1.402 |
| 1 | substitution-high | +0.061 | +0.043 | +0.018 | 1.452 | 1.433 |
| 2 | dropout-mid | +0.032 | +0.028 | +0.004 | 1.168 | 1.146 |
| 2 | dropout-high | +0.065 | +0.055 | +0.010 | 1.201 | 1.173 |
| 2 | substitution-mid | +0.020 | +0.017 | +0.003 | 1.156 | 1.134 |
| 2 | substitution-high | +0.065 | +0.054 | +0.011 | 1.201 | 1.172 |
| 3 | dropout-mid | +0.037 | +0.033 | +0.003 | 1.099 | 1.068 |
| 3 | dropout-high | +0.075 | +0.066 | +0.009 | 1.137 | 1.101 |
| 3 | substitution-mid | +0.025 | +0.022 | +0.004 | 1.088 | 1.057 |
| 3 | substitution-high | +0.071 | +0.058 | +0.013 | 1.133 | 1.093 |

## 3. Packet/Burst Loss Robustness

Values are mel-L1 degradation from clean. `robust_imp` is Base degradation minus LCA degradation; positive means LCA is more robust.

### test-clean

| L | Condition | actual loss | Base deg | LCA deg | robust_imp | Base mel | LCA mel |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | packet-loss-1p | 0.006 | +0.003 | +0.003 | -0.000 | 1.246 | 1.222 |
| 1 | packet-loss-3p | 0.029 | +0.012 | +0.011 | +0.001 | 1.256 | 1.231 |
| 1 | packet-loss-5p | 0.049 | +0.018 | +0.017 | +0.002 | 1.262 | 1.236 |
| 1 | burst-2f | 0.008 | +0.004 | +0.003 | +0.001 | 1.247 | 1.223 |
| 1 | burst-5f | 0.019 | +0.008 | +0.007 | +0.000 | 1.251 | 1.227 |
| 1 | burst-10f | 0.038 | +0.015 | +0.013 | +0.001 | 1.258 | 1.233 |
| 2 | packet-loss-1p | 0.010 | +0.006 | +0.006 | +0.000 | 0.996 | 0.963 |
| 2 | packet-loss-3p | 0.027 | +0.014 | +0.013 | +0.001 | 1.004 | 0.971 |
| 2 | packet-loss-5p | 0.048 | +0.025 | +0.025 | +0.000 | 1.015 | 0.983 |
| 2 | burst-2f | 0.008 | +0.004 | +0.004 | +0.001 | 0.994 | 0.961 |
| 2 | burst-5f | 0.019 | +0.009 | +0.008 | +0.000 | 0.999 | 0.966 |
| 2 | burst-10f | 0.038 | +0.017 | +0.016 | +0.000 | 1.006 | 0.974 |
| 3 | packet-loss-1p | 0.010 | +0.006 | +0.006 | +0.000 | 0.927 | 0.888 |
| 3 | packet-loss-3p | 0.029 | +0.016 | +0.015 | +0.001 | 0.937 | 0.898 |
| 3 | packet-loss-5p | 0.052 | +0.027 | +0.026 | +0.001 | 0.948 | 0.908 |
| 3 | burst-2f | 0.008 | +0.005 | +0.004 | +0.001 | 0.926 | 0.886 |
| 3 | burst-5f | 0.019 | +0.011 | +0.010 | +0.001 | 0.931 | 0.892 |
| 3 | burst-10f | 0.038 | +0.019 | +0.018 | +0.000 | 0.939 | 0.901 |

### test-other

| L | Condition | actual loss | Base deg | LCA deg | robust_imp | Base mel | LCA mel |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | packet-loss-1p | 0.013 | +0.004 | +0.004 | -0.000 | 1.396 | 1.395 |
| 1 | packet-loss-3p | 0.032 | +0.011 | +0.010 | +0.001 | 1.402 | 1.401 |
| 1 | packet-loss-5p | 0.049 | +0.015 | +0.014 | +0.001 | 1.407 | 1.405 |
| 1 | burst-2f | 0.008 | +0.003 | +0.002 | +0.001 | 1.394 | 1.393 |
| 1 | burst-5f | 0.020 | +0.006 | +0.006 | -0.000 | 1.397 | 1.397 |
| 1 | burst-10f | 0.041 | +0.011 | +0.011 | -0.000 | 1.402 | 1.402 |
| 2 | packet-loss-1p | 0.007 | +0.003 | +0.003 | -0.000 | 1.138 | 1.120 |
| 2 | packet-loss-3p | 0.032 | +0.011 | +0.010 | +0.000 | 1.147 | 1.128 |
| 2 | packet-loss-5p | 0.052 | +0.021 | +0.019 | +0.001 | 1.157 | 1.137 |
| 2 | burst-2f | 0.008 | +0.003 | +0.004 | -0.001 | 1.139 | 1.122 |
| 2 | burst-5f | 0.020 | +0.008 | +0.008 | +0.001 | 1.144 | 1.125 |
| 2 | burst-10f | 0.041 | +0.016 | +0.016 | +0.000 | 1.152 | 1.133 |
| 3 | packet-loss-1p | 0.010 | +0.003 | +0.003 | +0.000 | 1.066 | 1.038 |
| 3 | packet-loss-3p | 0.031 | +0.012 | +0.011 | +0.001 | 1.074 | 1.046 |
| 3 | packet-loss-5p | 0.046 | +0.018 | +0.018 | -0.000 | 1.080 | 1.053 |
| 3 | burst-2f | 0.008 | +0.004 | +0.003 | +0.001 | 1.066 | 1.038 |
| 3 | burst-5f | 0.020 | +0.009 | +0.008 | +0.001 | 1.071 | 1.043 |
| 3 | burst-10f | 0.041 | +0.015 | +0.014 | +0.001 | 1.077 | 1.049 |

## 4. Aggregate Readout

- `test-clean` `index_perturb`: mean mel robust_imp `+0.0066`, positive cells `12/12`.
- `test-other` `index_perturb`: mean mel robust_imp `+0.0071`, positive cells `12/12`.
- `test-clean` `packet_burst`: mean mel robust_imp `+0.0007`, positive cells `17/18`.
- `test-other` `packet_burst`: mean mel robust_imp `+0.0004`, positive cells `12/18`.

## 5. Interpretation

- On held-out test-clean/test-other clean subsets, full LCA generally lowers mel-L1 and improves STOI relative to Base, while PESQ is mixed and often slightly lower at L2/L3.
- Index-level dropout/substitution robustness should be interpreted via degradation from clean rather than absolute mel alone.
- Packet/burst conditions provide a more communication-like stress test; positive robust_imp cells indicate where full LCA degrades less than Base under packetized or bursty index loss.
- These are 100-sample subset results, not full test-clean/test-other exhaustive evaluation.
