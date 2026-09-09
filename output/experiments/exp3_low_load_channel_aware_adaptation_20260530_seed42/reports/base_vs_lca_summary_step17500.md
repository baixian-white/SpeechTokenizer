# Base vs LCA evaluation summary

- Base ckpt: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output/experiments/exp3_low_load_channel_aware_adaptation_20260530_seed42/checkpoints/SCIT-Speech-LCA_step17500_full_mel_optimum.pt`
  - sha256: `c3d9bc6e11de74b2f3f1491e66a954e2df278cd401a6fd398dcdd2e55e865293`
- Sample count per (L, channel): 8
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0321 | 1.2022 | -7.183 | 0.4327 | 0.7866 | 1.3330 |
| base | 1 | dropout-low | 0.0323 | 1.2090 | -7.567 | 0.4213 | 0.7825 | 1.3221 |
| base | 1 | dropout-mid | 0.0326 | 1.2173 | -7.664 | 0.4147 | 0.7790 | 1.3124 |
| base | 1 | substitution-low | 0.0322 | 1.2062 | -7.268 | 0.4300 | 0.7841 | 1.3078 |
| base | 1 | substitution-mid | 0.0322 | 1.2081 | -7.269 | 0.4303 | 0.7834 | 1.3165 |
| base | 2 | clean | 0.0246 | 0.9549 | -1.447 | 0.6514 | 0.8442 | 1.7072 |
| base | 2 | dropout-low | 0.0249 | 0.9608 | -1.620 | 0.6451 | 0.8427 | 1.6411 |
| base | 2 | dropout-mid | 0.0254 | 0.9724 | -1.885 | 0.6350 | 0.8356 | 1.6355 |
| base | 2 | substitution-low | 0.0246 | 0.9555 | -1.458 | 0.6509 | 0.8439 | 1.7062 |
| base | 2 | substitution-mid | 0.0249 | 0.9630 | -1.812 | 0.6393 | 0.8419 | 1.6115 |
| base | 3 | clean | 0.0222 | 0.8916 | +0.354 | 0.7089 | 0.8697 | 1.9123 |
| base | 3 | dropout-low | 0.0226 | 0.9001 | -0.044 | 0.6978 | 0.8658 | 1.8725 |
| base | 3 | dropout-mid | 0.0235 | 0.9174 | -0.617 | 0.6781 | 0.8574 | 1.7025 |
| base | 3 | substitution-low | 0.0222 | 0.8938 | +0.352 | 0.7089 | 0.8697 | 1.9088 |
| base | 3 | substitution-mid | 0.0224 | 0.8982 | +0.279 | 0.7065 | 0.8677 | 1.8963 |
| lca | 1 | clean | 0.0318 | 1.1488 | -6.767 | 0.4430 | 0.8010 | 1.3531 |
| lca | 1 | dropout-low | 0.0320 | 1.1525 | -7.095 | 0.4333 | 0.7983 | 1.3418 |
| lca | 1 | dropout-mid | 0.0323 | 1.1620 | -7.283 | 0.4249 | 0.7938 | 1.3296 |
| lca | 1 | substitution-low | 0.0319 | 1.1524 | -6.839 | 0.4406 | 0.7987 | 1.3310 |
| lca | 1 | substitution-mid | 0.0319 | 1.1563 | -6.875 | 0.4394 | 0.7978 | 1.3334 |
| lca | 2 | clean | 0.0242 | 0.9175 | -1.096 | 0.6626 | 0.8617 | 1.7180 |
| lca | 2 | dropout-low | 0.0245 | 0.9236 | -1.287 | 0.6567 | 0.8590 | 1.6454 |
| lca | 2 | dropout-mid | 0.0250 | 0.9340 | -1.727 | 0.6405 | 0.8537 | 1.6546 |
| lca | 2 | substitution-low | 0.0242 | 0.9183 | -1.108 | 0.6622 | 0.8614 | 1.7156 |
| lca | 2 | substitution-mid | 0.0245 | 0.9254 | -1.462 | 0.6509 | 0.8599 | 1.6711 |
| lca | 3 | clean | 0.0216 | 0.8496 | +0.630 | 0.7196 | 0.8835 | 1.9436 |
| lca | 3 | dropout-low | 0.0219 | 0.8580 | +0.355 | 0.7113 | 0.8804 | 1.8969 |
| lca | 3 | dropout-mid | 0.0227 | 0.8750 | -0.225 | 0.6920 | 0.8694 | 1.7395 |
| lca | 3 | substitution-low | 0.0216 | 0.8510 | +0.623 | 0.7194 | 0.8833 | 1.9413 |
| lca | 3 | substitution-mid | 0.0217 | 0.8571 | +0.540 | 0.7168 | 0.8810 | 1.9225 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0003 | -0.0534 | +0.416 | +0.0144 | +0.0200 |
| 1 | dropout-low | -0.0003 | -0.0564 | +0.471 | +0.0157 | +0.0197 |
| 1 | dropout-mid | -0.0003 | -0.0553 | +0.381 | +0.0148 | +0.0172 |
| 1 | substitution-low | -0.0003 | -0.0538 | +0.429 | +0.0146 | +0.0232 |
| 1 | substitution-mid | -0.0003 | -0.0518 | +0.394 | +0.0144 | +0.0170 |
| 2 | clean | -0.0004 | -0.0374 | +0.350 | +0.0176 | +0.0107 |
| 2 | dropout-low | -0.0004 | -0.0373 | +0.333 | +0.0163 | +0.0043 |
| 2 | dropout-mid | -0.0004 | -0.0385 | +0.159 | +0.0182 | +0.0191 |
| 2 | substitution-low | -0.0004 | -0.0373 | +0.350 | +0.0175 | +0.0094 |
| 2 | substitution-mid | -0.0004 | -0.0376 | +0.351 | +0.0180 | +0.0596 |
| 3 | clean | -0.0007 | -0.0420 | +0.277 | +0.0138 | +0.0312 |
| 3 | dropout-low | -0.0007 | -0.0421 | +0.400 | +0.0146 | +0.0244 |
| 3 | dropout-mid | -0.0008 | -0.0424 | +0.392 | +0.0120 | +0.0370 |
| 3 | substitution-low | -0.0006 | -0.0428 | +0.270 | +0.0136 | +0.0325 |
| 3 | substitution-mid | -0.0006 | -0.0411 | +0.262 | +0.0133 | +0.0262 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.