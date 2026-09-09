# LibriSpeech test subset clean evaluation summary

- run: `exp6_librispeech_test_subset_20260606`
- splits: `test-clean` 100 samples, `test-other` 100 samples
- condition: clean only
- models: SCIT-Speech-Base vs full LCA step30000

## test-clean

| Model | L | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 100 | 1.243 | 0.798 | 1.286 | -7.74 |
| base | 2 | 100 | 0.990 | 0.862 | 1.674 | -2.42 |
| base | 3 | 100 | 0.921 | 0.884 | 1.902 | -0.99 |
| lca | 1 | 100 | 1.220 | 0.811 | 1.305 | -8.06 |
| lca | 2 | 100 | 0.958 | 0.869 | 1.655 | -2.68 |
| lca | 3 | 100 | 0.882 | 0.890 | 1.882 | -0.86 |

| L | Δmel-L1 ↓ | ΔSTOI ↑ | ΔPESQ-WB ↑ | ΔSI-SNR ↑ |
|---:|---:|---:|---:|---:|
| 1 | -0.024 | +0.013 | +0.019 | -0.33 |
| 2 | -0.032 | +0.007 | -0.019 | -0.26 |
| 3 | -0.038 | +0.007 | -0.020 | +0.12 |

## test-other

| Model | L | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 100 | 1.392 | 0.767 | 1.260 | -8.78 |
| base | 2 | 100 | 1.136 | 0.829 | 1.578 | -3.19 |
| base | 3 | 100 | 1.062 | 0.852 | 1.774 | -1.65 |
| lca | 1 | 100 | 1.391 | 0.780 | 1.279 | -9.13 |
| lca | 2 | 100 | 1.118 | 0.838 | 1.564 | -3.44 |
| lca | 3 | 100 | 1.035 | 0.860 | 1.743 | -1.51 |

| L | Δmel-L1 ↓ | ΔSTOI ↑ | ΔPESQ-WB ↑ | ΔSI-SNR ↑ |
|---:|---:|---:|---:|---:|
| 1 | -0.001 | +0.014 | +0.019 | -0.35 |
| 2 | -0.018 | +0.010 | -0.014 | -0.25 |
| 3 | -0.027 | +0.008 | -0.031 | +0.14 |

## Initial interpretation

- This clean-only subset evaluates whether the Base/LCA behavior survives on held-out LibriSpeech test splits.
- The next step is the matched perturbation subset evaluation on the same 100+100 samples to test whether full LCA retains robust_imp outside the original 8 fixed samples.
