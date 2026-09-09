# Exp22 Paper-Grade Speaker Identity Aggregate

This report aggregates completed Exp22 speaker identity runs across seeds.

## Codes/Latent Speaker Probe

| model | feature | L | seeds | top1 | top5 | macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| base | codes_hist | 1 | 3 | 0.344 +/- 0.008 | 0.673 +/- 0.017 | 0.325 +/- 0.005 |
| base | codes_hist | 2 | 3 | 0.489 +/- 0.028 | 0.809 +/- 0.014 | 0.470 +/- 0.031 |
| base | codes_hist | 3 | 3 | 0.510 +/- 0.007 | 0.821 +/- 0.009 | 0.493 +/- 0.003 |
| base | latent_stats | 1 | 3 | 0.372 +/- 0.019 | 0.688 +/- 0.023 | 0.356 +/- 0.019 |
| base | latent_stats | 2 | 3 | 0.558 +/- 0.008 | 0.825 +/- 0.007 | 0.542 +/- 0.006 |
| base | latent_stats | 3 | 3 | 0.581 +/- 0.019 | 0.827 +/- 0.027 | 0.568 +/- 0.022 |
| lca | codes_hist | 1 | 3 | 0.325 +/- 0.011 | 0.646 +/- 0.004 | 0.304 +/- 0.016 |
| lca | codes_hist | 2 | 3 | 0.451 +/- 0.014 | 0.783 +/- 0.009 | 0.430 +/- 0.019 |
| lca | codes_hist | 3 | 3 | 0.510 +/- 0.034 | 0.807 +/- 0.017 | 0.487 +/- 0.033 |
| lca | latent_stats | 1 | 3 | 0.341 +/- 0.009 | 0.645 +/- 0.011 | 0.325 +/- 0.010 |
| lca | latent_stats | 2 | 3 | 0.527 +/- 0.023 | 0.803 +/- 0.014 | 0.513 +/- 0.025 |
| lca | latent_stats | 3 | 3 | 0.547 +/- 0.041 | 0.799 +/- 0.028 | 0.535 +/- 0.044 |

## ECAPA Speaker Preservation

| model | L | seeds | top1 | verified | EER | TAR@FAR=0.01 | margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | 3 | 0.108 +/- 0.017 | 0.964 +/- 0.010 | 0.767 +/- 0.021 | 0.004 +/- 0.006 | -0.116 +/- 0.007 |
| base | 2 | 3 | 0.412 +/- 0.036 | 0.996 +/- 0.001 | 0.544 +/- 0.021 | 0.046 +/- 0.015 | -0.022 +/- 0.009 |
| base | 3 | 3 | 0.575 +/- 0.042 | 1.000 +/- 0.000 | 0.442 +/- 0.019 | 0.099 +/- 0.018 | 0.018 +/- 0.010 |
| lca | 1 | 3 | 0.075 +/- 0.014 | 0.987 +/- 0.007 | 0.810 +/- 0.026 | 0.004 +/- 0.002 | -0.140 +/- 0.016 |
| lca | 2 | 3 | 0.233 +/- 0.040 | 0.999 +/- 0.001 | 0.666 +/- 0.039 | 0.015 +/- 0.006 | -0.080 +/- 0.017 |
| lca | 3 | 3 | 0.345 +/- 0.021 | 0.999 +/- 0.001 | 0.599 +/- 0.040 | 0.033 +/- 0.013 | -0.046 +/- 0.015 |
| original | 0 | 3 | 0.996 +/- 0.004 | 1.000 +/- 0.000 | 0.034 +/- 0.005 | 0.938 +/- 0.008 | 0.309 +/- 0.003 |

## Notes

- Probe metrics answer whether speaker identity is linearly recoverable from SCIT codes/features.
- ECAPA preservation metrics answer whether decoded waveforms remain recognizable as the same speaker.
- Table cells show mean +/- std when at least two seeds are available; single-seed cells show the raw value.
- CSV files additionally include a small-sample 95% t-interval half-width for each metric.
- MFCC preservation runs are retained as dependency-light sanity checks, not paper-grade speaker verification.
