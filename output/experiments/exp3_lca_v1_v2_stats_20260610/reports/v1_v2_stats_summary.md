# E4 LCA v1 vs v2 Statistical Post-processing

v1 input: `h:\H-CODE\speechtokenizer\output\experiments\exp3_low_load_channel_aware_adaptation_20260530_seed42\metrics\base_vs_lca_results.csv`

v2 input: `h:\H-CODE\speechtokenizer\output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42\metrics\base_vs_lca_results.csv`

## Important design note

v1 and v2 do not share the same perturbation grid: v1 has dropout-low/dropout-mid/substitution-low/substitution-mid; v2 has dropout-mid/dropout-high/substitution-mid/substitution-high. Therefore this report separates (1) descriptive summaries over each version's own 12 perturbation cells and (2) paired tests only on common mid cells.

## Descriptive summaries over each version's own perturbation grid

| metric | version | n pairs | mean rob_imp | 95% CI | positive cells / n_cells |
|---|---|---:|---:|---:|---:|
| corr | v1 | 96 | -0.000324 | [-0.001382, 0.000735] | 5/12 |
| corr | v2 | 96 | 0.000949 | [-0.003148, 0.005045] | 7/12 |
| mel_l1 | v1 | 96 | -0.000281 | [-0.001208, 0.000645] | 3/12 |
| mel_l1 | v2 | 96 | 0.005529 | [0.003563, 0.007495] | 12/12 |
| pesq_wb | v1 | 96 | -0.001889 | [-0.009877, 0.006098] | 7/12 |
| pesq_wb | v2 | 96 | 0.014562 | [0.006606, 0.022518] | 10/12 |
| si_snr_db | v1 | 96 | -0.017634 | [-0.048553, 0.013286] | 4/12 |
| si_snr_db | v2 | 96 | 0.050551 | [-0.065044, 0.166145] | 7/12 |
| stoi | v1 | 96 | -0.000073 | [-0.000616, 0.000471] | 6/12 |
| stoi | v2 | 96 | 0.002300 | [0.001152, 0.003448] | 9/12 |
| wave_l1 | v1 | 96 | -0.000001 | [-0.000034, 0.000032] | 6/12 |
| wave_l1 | v2 | 96 | 0.000066 | [-0.000011, 0.000144] | 7/12 |

## Paired v2-v1 tests on common mid perturbation cells only

Common channels: `dropout-mid, substitution-mid` (3 L × 2 channels × 8 samples = 48 pairs per metric).

| metric | n pairs | mean v2-v1 | 95% CI | t | p (normal approx) | positive cells | significant positive cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| corr | 48 | 0.000497 | [-0.003631, 0.004625] | 0.24 | 8.135e-01 | 3 | 0 |
| mel_l1 | 48 | 0.005222 | [0.003225, 0.007219] | 5.13 | 2.967e-07 | 6 | 4 |
| pesq_wb | 48 | 0.016937 | [0.004427, 0.029447] | 2.65 | 7.962e-03 | 5 | 1 |
| si_snr_db | 48 | 0.055565 | [-0.054843, 0.165973] | 0.99 | 3.239e-01 | 5 | 0 |
| stoi | 48 | 0.001357 | [0.000331, 0.002384] | 2.59 | 9.559e-03 | 6 | 0 |
| wave_l1 | 48 | 0.000048 | [-0.000053, 0.000148] | 0.93 | 3.546e-01 | 3 | 0 |

## Interpretation

- The descriptive table is the correct source for reproducing the existing v1/v2 narrative because each version was evaluated at its own perturbation grid.
- The paired table is stricter but only covers mid perturbation cells. Do not use it to claim behavior under v2 high perturbations.
- Positive `v2-v1` means v2 has larger robustness improvement than v1.
