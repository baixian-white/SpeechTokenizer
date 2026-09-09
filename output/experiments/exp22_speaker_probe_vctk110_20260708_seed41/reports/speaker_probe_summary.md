# Exp22 Speaker Probe

- train_per_speaker: 5
- test_per_speaker: 5

| model | feature | L | speakers | test n | dim | top1 | top5 | macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | codes_hist | 1 | 110 | 550 | 1024 | 0.349 | 0.693 | 0.328 |
| base | codes_hist | 2 | 110 | 550 | 2048 | 0.520 | 0.822 | 0.505 |
| base | codes_hist | 3 | 110 | 550 | 3072 | 0.516 | 0.829 | 0.494 |
| base | latent_stats | 1 | 110 | 550 | 2048 | 0.365 | 0.709 | 0.351 |
| base | latent_stats | 2 | 110 | 550 | 4096 | 0.553 | 0.833 | 0.538 |
| base | latent_stats | 3 | 110 | 550 | 6144 | 0.564 | 0.796 | 0.547 |
| lca | codes_hist | 1 | 110 | 550 | 1024 | 0.316 | 0.651 | 0.295 |
| lca | codes_hist | 2 | 110 | 550 | 2048 | 0.467 | 0.791 | 0.451 |
| lca | codes_hist | 3 | 110 | 550 | 3072 | 0.531 | 0.824 | 0.513 |
| lca | latent_stats | 1 | 110 | 550 | 2048 | 0.333 | 0.658 | 0.315 |
| lca | latent_stats | 2 | 110 | 550 | 4096 | 0.536 | 0.818 | 0.526 |
| lca | latent_stats | 3 | 110 | 550 | 6144 | 0.544 | 0.793 | 0.535 |
