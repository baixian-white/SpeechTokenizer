# Exp22 Speaker Probe

- train_per_speaker: 5
- test_per_speaker: 5

| model | feature | L | speakers | test n | dim | top1 | top5 | macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | codes_hist | 1 | 110 | 550 | 1024 | 0.335 | 0.664 | 0.319 |
| base | codes_hist | 2 | 110 | 550 | 2048 | 0.480 | 0.811 | 0.455 |
| base | codes_hist | 3 | 110 | 550 | 3072 | 0.511 | 0.822 | 0.495 |
| base | latent_stats | 1 | 110 | 550 | 2048 | 0.356 | 0.664 | 0.340 |
| base | latent_stats | 2 | 110 | 550 | 4096 | 0.567 | 0.820 | 0.550 |
| base | latent_stats | 3 | 110 | 550 | 6144 | 0.602 | 0.842 | 0.590 |
| lca | codes_hist | 1 | 110 | 550 | 1024 | 0.338 | 0.644 | 0.322 |
| lca | codes_hist | 2 | 110 | 550 | 2048 | 0.440 | 0.773 | 0.417 |
| lca | codes_hist | 3 | 110 | 550 | 3072 | 0.529 | 0.809 | 0.498 |
| lca | latent_stats | 1 | 110 | 550 | 2048 | 0.351 | 0.636 | 0.334 |
| lca | latent_stats | 2 | 110 | 550 | 4096 | 0.544 | 0.800 | 0.529 |
| lca | latent_stats | 3 | 110 | 550 | 6144 | 0.589 | 0.829 | 0.579 |
