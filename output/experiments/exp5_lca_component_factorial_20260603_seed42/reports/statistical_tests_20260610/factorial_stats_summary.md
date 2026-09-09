# E3 V0–V4 Factorial Statistical Tests (n=256)

Input root: `h:\H-CODE\speechtokenizer\output\experiments\exp5_lca_component_factorial_20260603_seed42\eval_unified_n64_20260609`

## Variant means

| Variant | n pairs | n samples | mean rob_imp | 95% CI | positive cells / 12 |
|---|---:|---:|---:|---:|---:|
| V0 | 3072 | 256 | -0.001352 | [-0.001589, -0.001116] | 2 |
| V1 | 3072 | 256 | -0.002447 | [-0.002885, -0.002009] | 6 |
| V2 | 3072 | 256 | 0.002578 | [0.002195, 0.002961] | 8 |
| V3 | 3072 | 256 | 0.004939 | [0.004599, 0.005278] | 12 |
| V4 | 3072 | 256 | 0.007423 | [0.007043, 0.007802] | 12 |

## Pairwise pooled tests

| Comparison | Description | n pairs | mean diff | 95% CI | t | p | positive cells | significant positive cells |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V3_minus_V1 | ChannelSim added on top of random-L | 3072 | 0.007386 | [0.006869, 0.007902] | 28.03 | <1e-300 | 12 | 12 |
| V4_minus_V3 | Consistency added on top of random-L + ChannelSim | 3072 | 0.002484 | [0.002025, 0.002943] | 10.61 | <1e-300 | 11 | 10 |
| V3_minus_V2 | random-L added on top of ChannelSim | 3072 | 0.002361 | [0.001881, 0.002840] | 9.64 | <1e-300 | 10 | 5 |
| V2_minus_V1 | ChannelSim-only vs random-L-only configuration contrast | 3072 | 0.005025 | [0.004484, 0.005566] | 18.20 | <1e-300 | 12 | 12 |
| V4_minus_V0 | End-to-end full LCA vs full-depth clean control | 3072 | 0.008775 | [0.008320, 0.009230] | 37.82 | <1e-300 | 12 | 12 |

## Notes

- Robustness improvement is `Base degradation - Variant degradation`; positive is better.
- Degradation is perturbed mel-L1 minus clean mel-L1 at matched `(sample_id, L)`.
- Pooled tests use 12 perturbation cells × 256 samples = 3072 paired values.
- Cell-level tests are saved to `factorial_pairwise_cell_tests.csv`.
