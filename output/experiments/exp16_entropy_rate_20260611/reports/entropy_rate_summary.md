# Exp16: Empirical RVQ Index Entropy (SCIT-Speech-Base)

- Codebook size: 1024 (H_max = log2 = 10.000 bits/code)
- Frame rate f_q = 50 Hz (latent rate)
- Naive per-layer rate: 50 * 10 = 500 bps

## Per-layer entropy

| split | L | n_frames | n_used | dead% | top1 | H (bits) | R_entropy (bps) | R_naive (bps) | H / H_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test-clean | 1 | 132170 | 395 | 61.4% | 0.023 | 8.297 | 414.8 | 500.0 | 0.830 |
| test-clean | 2 | 132170 | 997 | 2.6% | 0.014 | 9.340 | 467.0 | 500.0 | 0.934 |
| test-clean | 3 | 132170 | 993 | 3.0% | 0.011 | 9.485 | 474.2 | 500.0 | 0.948 |
| test-other | 1 | 100812 | 400 | 60.9% | 0.041 | 7.951 | 397.6 | 500.0 | 0.795 |
| test-other | 2 | 100812 | 986 | 3.7% | 0.030 | 8.806 | 440.3 | 500.0 | 0.881 |
| test-other | 3 | 100812 | 1006 | 1.8% | 0.017 | 9.262 | 463.1 | 500.0 | 0.926 |

## Cumulative bits saved per L (entropy vs naive)

| split | L_cum | sum_H (bits) | R_entropy_cum (bps) | R_naive_cum (bps) | savings vs naive |
|---|---:|---:|---:|---:|---:|
| test-clean | 1 | 8.297 | 414.8 | 500.0 | 17.03% |
| test-clean | 2 | 17.637 | 881.8 | 1000.0 | 11.82% |
| test-clean | 3 | 27.121 | 1356.1 | 1500.0 | 9.60% |
| test-other | 1 | 7.951 | 397.6 | 500.0 | 20.49% |
| test-other | 2 | 16.757 | 837.9 | 1000.0 | 16.21% |
| test-other | 3 | 26.019 | 1300.9 | 1500.0 | 13.27% |

## Paper-ready interpretation

On the test-clean split, layer-1 indices have empirical entropy H(I_1) = 8.30 bits versus the codebook ceiling of 10 bits, giving an entropy-coded rate of 415 bps per layer instead of the naive 500 bps (a 17.0% reduction).
By layer 3 the per-layer entropy drops to H(I_3) = 9.48 bits (5.2% below the ceiling), with the largest H_max - H gap of 2.05 bits observed at split=test-other, L=1.
These measurements support the paper's caveat that the 500*L bps figure used in section 3.1 / section 7 is a CONSERVATIVE upper bound: an entropy coder operating on the empirical index distribution would require strictly fewer bits per second than the naive bit-packed limit.
