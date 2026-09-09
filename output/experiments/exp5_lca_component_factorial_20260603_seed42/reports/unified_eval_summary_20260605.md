# Unified LCA Factorial Evaluation Summary

- evaluation: `eval_unified_20260605`
- samples: 8 fixed validation samples
- conditions: clean, dropout-mid/high, substitution-mid/high
- table values below use the `lca` rows from each variant evaluation output.

## Clean Metrics

| Variant | L | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ |
|---|---:|---:|---:|---:|---:|
| V0 Clean control | 1 | 1.143 | 0.804 | 1.374 | -6.63 |
| V0 Clean control | 2 | 0.878 | 0.867 | 1.809 | -0.92 |
| V0 Clean control | 3 | 0.814 | 0.887 | 2.039 | +0.99 |
| V1 random-L only | 1 | 1.161 | 0.798 | 1.355 | -7.70 |
| V1 random-L only | 2 | 0.931 | 0.856 | 1.695 | -1.50 |
| V1 random-L only | 3 | 0.857 | 0.882 | 1.910 | +0.18 |
| V2 ChannelSim only | 1 | 1.223 | 0.787 | 1.317 | -7.55 |
| V2 ChannelSim only | 2 | 0.955 | 0.849 | 1.667 | -1.82 |
| V2 ChannelSim only | 3 | 0.878 | 0.876 | 1.914 | +0.28 |
| V3 random-L + ChannelSim | 1 | 1.153 | 0.798 | 1.346 | -7.39 |
| V3 random-L + ChannelSim | 2 | 0.926 | 0.856 | 1.729 | -1.57 |
| V3 random-L + ChannelSim | 3 | 0.855 | 0.878 | 1.913 | +0.37 |
| V4 full LCA (+ consistency) | 1 | 1.147 | 0.797 | 1.347 | -7.77 |
| V4 full LCA (+ consistency) | 2 | 0.923 | 0.857 | 1.709 | -1.76 |
| V4 full LCA (+ consistency) | 3 | 0.859 | 0.879 | 1.891 | -0.03 |

## Mel-L1 Degradation From Clean

| Variant | L | dropout-mid | dropout-high | substitution-mid | substitution-high |
|---|---:|---:|---:|---:|---:|
| V0 | 1 | +0.026 | +0.055 | +0.020 | +0.042 |
| V0 | 2 | +0.044 | +0.085 | +0.019 | +0.054 |
| V0 | 3 | +0.042 | +0.089 | +0.019 | +0.064 |
| V1 | 1 | +0.028 | +0.056 | +0.011 | +0.060 |
| V1 | 2 | +0.036 | +0.076 | +0.023 | +0.083 |
| V1 | 3 | +0.045 | +0.079 | +0.032 | +0.069 |
| V2 | 1 | +0.028 | +0.055 | +0.008 | +0.031 |
| V2 | 2 | +0.032 | +0.065 | +0.025 | +0.064 |
| V2 | 3 | +0.033 | +0.079 | +0.022 | +0.051 |
| V3 | 1 | +0.024 | +0.041 | +0.014 | +0.043 |
| V3 | 2 | +0.030 | +0.077 | +0.014 | +0.049 |
| V3 | 3 | +0.034 | +0.085 | +0.023 | +0.063 |
| V4 | 1 | +0.015 | +0.045 | +0.009 | +0.034 |
| V4 | 2 | +0.033 | +0.066 | +0.015 | +0.044 |
| V4 | 3 | +0.035 | +0.069 | +0.019 | +0.057 |

## Mel robust_imp vs V0

Positive means the variant degrades less than V0 under the same perturbation.

| Variant | L | dropout-mid | dropout-high | substitution-mid | substitution-high |
|---|---:|---:|---:|---:|---:|
| V1 | 1 | -0.002 | -0.001 | +0.009 | -0.018 |
| V1 | 2 | +0.007 | +0.009 | -0.005 | -0.029 |
| V1 | 3 | -0.003 | +0.009 | -0.013 | -0.005 |
| V2 | 1 | -0.003 | +0.000 | +0.012 | +0.011 |
| V2 | 2 | +0.011 | +0.020 | -0.007 | -0.010 |
| V2 | 3 | +0.008 | +0.010 | -0.003 | +0.013 |
| V3 | 1 | +0.002 | +0.014 | +0.006 | -0.001 |
| V3 | 2 | +0.014 | +0.007 | +0.005 | +0.005 |
| V3 | 3 | +0.008 | +0.004 | -0.004 | +0.001 |
| V4 | 1 | +0.011 | +0.010 | +0.011 | +0.009 |
| V4 | 2 | +0.011 | +0.019 | +0.004 | +0.011 |
| V4 | 3 | +0.006 | +0.020 | +0.000 | +0.007 |

## Quick Read

- Best clean mel-L1 at L=1: `V0` (1.143).
- Best clean mel-L1 at L=2: `V0` (0.878).
- Best clean mel-L1 at L=3: `V0` (0.814).
- Mean mel robust_imp vs V0 for `V1`: -0.0033 across 12 cells.
- Mean mel robust_imp vs V0 for `V2`: +0.0054 across 12 cells.
- Mean mel robust_imp vs V0 for `V3`: +0.0052 across 12 cells.
- Mean mel robust_imp vs V0 for `V4`: +0.0099 across 12 cells.

## Interpretation

- `V1` isolates random-L under clean training; it mainly tests whether random-L alone improves low-load clean reconstruction.
- `V2` isolates ChannelSim at fixed L=3; it mainly tests whether perturbation exposure alone helps robustness.
- `V3` combines random-L and ChannelSim without consistency; it tests whether random-L stabilizes the ChannelSim objective.
- `V4` is the full LCA reference with consistency; it tests whether consistency adds robustness beyond V3.
- Treat these as 8-sample objective metrics. WER/CER is not included in this unified evaluation.
