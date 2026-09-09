# Exp22 Speaker Probe

- train_per_speaker: 5
- test_per_speaker: 5

| model | feature | L | speakers | test n | dim | top1 | top5 | macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | codes_hist | 1 | 110 | 550 | 1024 | 0.349 | 0.664 | 0.327 |
| base | codes_hist | 2 | 110 | 550 | 2048 | 0.467 | 0.795 | 0.449 |
| base | codes_hist | 3 | 110 | 550 | 3072 | 0.502 | 0.811 | 0.489 |
| base | latent_stats | 1 | 110 | 550 | 2048 | 0.393 | 0.693 | 0.376 |
| base | latent_stats | 2 | 110 | 550 | 4096 | 0.553 | 0.824 | 0.540 |
| base | latent_stats | 3 | 110 | 550 | 6144 | 0.576 | 0.844 | 0.567 |
| lca | codes_hist | 1 | 110 | 550 | 1024 | 0.322 | 0.644 | 0.295 |
| lca | codes_hist | 2 | 110 | 550 | 2048 | 0.445 | 0.785 | 0.422 |
| lca | codes_hist | 3 | 110 | 550 | 3072 | 0.471 | 0.789 | 0.450 |
| lca | latent_stats | 1 | 110 | 550 | 2048 | 0.338 | 0.642 | 0.325 |
| lca | latent_stats | 2 | 110 | 550 | 4096 | 0.500 | 0.791 | 0.484 |
| lca | latent_stats | 3 | 110 | 550 | 6144 | 0.507 | 0.775 | 0.491 |
