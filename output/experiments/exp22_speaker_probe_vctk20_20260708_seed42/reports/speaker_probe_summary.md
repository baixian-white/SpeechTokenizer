# Exp22 Speaker Probe

- train_per_speaker: 5
- test_per_speaker: 5

| model | feature | L | speakers | test n | dim | top1 | top5 | macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | codes_hist | 1 | 20 | 100 | 1024 | 0.610 | 0.950 | 0.589 |
| base | codes_hist | 2 | 20 | 100 | 2048 | 0.690 | 0.980 | 0.674 |
| base | codes_hist | 3 | 20 | 100 | 3072 | 0.730 | 0.960 | 0.713 |
| base | latent_stats | 1 | 20 | 100 | 2048 | 0.590 | 0.920 | 0.586 |
| base | latent_stats | 2 | 20 | 100 | 4096 | 0.760 | 0.960 | 0.755 |
| base | latent_stats | 3 | 20 | 100 | 6144 | 0.790 | 0.970 | 0.785 |
| lca | codes_hist | 1 | 20 | 100 | 1024 | 0.490 | 0.890 | 0.451 |
| lca | codes_hist | 2 | 20 | 100 | 2048 | 0.700 | 0.960 | 0.690 |
| lca | codes_hist | 3 | 20 | 100 | 3072 | 0.710 | 0.970 | 0.693 |
| lca | latent_stats | 1 | 20 | 100 | 2048 | 0.540 | 0.850 | 0.537 |
| lca | latent_stats | 2 | 20 | 100 | 4096 | 0.780 | 0.940 | 0.772 |
| lca | latent_stats | 3 | 20 | 100 | 6144 | 0.810 | 0.960 | 0.809 |
