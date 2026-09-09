# Base vs LCA evaluation summary

- Base ckpt: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_best.pt`
  - sha256: `af60223b2f44733f5a43d5a022871dad061dfe2421473c96f5d350c3d4b49a5a`
- Sample count per (L, channel): 8
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0321 | 1.2022 | -7.183 | 0.4327 | 0.7866 | 1.3330 |
| base | 1 | dropout-high | 0.0341 | 1.2544 | -9.770 | 0.3485 | 0.7550 | 1.2090 |
| base | 1 | dropout-mid | 0.0330 | 1.2270 | -8.044 | 0.3996 | 0.7751 | 1.2565 |
| base | 1 | substitution-high | 0.0325 | 1.2360 | -7.608 | 0.4172 | 0.7704 | 1.2930 |
| base | 1 | substitution-mid | 0.0322 | 1.2199 | -7.268 | 0.4296 | 0.7828 | 1.3206 |
| base | 2 | clean | 0.0246 | 0.9549 | -1.447 | 0.6514 | 0.8442 | 1.7072 |
| base | 2 | dropout-high | 0.0275 | 1.0220 | -3.788 | 0.5631 | 0.8120 | 1.4746 |
| base | 2 | dropout-mid | 0.0262 | 0.9966 | -2.606 | 0.6104 | 0.8286 | 1.5021 |
| base | 2 | substitution-high | 0.0254 | 1.0104 | -1.933 | 0.6332 | 0.8238 | 1.5641 |
| base | 2 | substitution-mid | 0.0249 | 0.9715 | -1.665 | 0.6436 | 0.8362 | 1.6541 |
| base | 3 | clean | 0.0222 | 0.8916 | +0.354 | 0.7089 | 0.8697 | 1.9123 |
| base | 3 | dropout-high | 0.0260 | 0.9797 | -2.261 | 0.6217 | 0.8281 | 1.4922 |
| base | 3 | dropout-mid | 0.0240 | 0.9316 | -0.859 | 0.6715 | 0.8503 | 1.7048 |
| base | 3 | substitution-high | 0.0232 | 0.9492 | -0.241 | 0.6905 | 0.8488 | 1.6926 |
| base | 3 | substitution-mid | 0.0226 | 0.9120 | +0.101 | 0.7024 | 0.8634 | 1.8574 |
| lca | 1 | clean | 0.0318 | 1.1466 | -7.772 | 0.4146 | 0.7970 | 1.3466 |
| lca | 1 | dropout-high | 0.0337 | 1.1902 | -9.950 | 0.3419 | 0.7728 | 1.2354 |
| lca | 1 | dropout-mid | 0.0325 | 1.1690 | -8.477 | 0.3864 | 0.7884 | 1.2712 |
| lca | 1 | substitution-high | 0.0322 | 1.1798 | -8.080 | 0.4019 | 0.7816 | 1.3059 |
| lca | 1 | substitution-mid | 0.0319 | 1.1588 | -7.869 | 0.4113 | 0.7927 | 1.3359 |
| lca | 2 | clean | 0.0244 | 0.9226 | -1.761 | 0.6388 | 0.8567 | 1.7089 |
| lca | 2 | dropout-high | 0.0272 | 0.9837 | -4.079 | 0.5528 | 0.8283 | 1.4529 |
| lca | 2 | dropout-mid | 0.0261 | 0.9559 | -3.040 | 0.5909 | 0.8422 | 1.5211 |
| lca | 2 | substitution-high | 0.0253 | 0.9776 | -2.419 | 0.6139 | 0.8354 | 1.5948 |
| lca | 2 | substitution-mid | 0.0247 | 0.9373 | -1.934 | 0.6322 | 0.8495 | 1.6688 |
| lca | 3 | clean | 0.0219 | 0.8587 | -0.029 | 0.7000 | 0.8790 | 1.8913 |
| lca | 3 | dropout-high | 0.0255 | 0.9307 | -2.711 | 0.6106 | 0.8449 | 1.5168 |
| lca | 3 | dropout-mid | 0.0233 | 0.8940 | -0.983 | 0.6688 | 0.8631 | 1.7298 |
| lca | 3 | substitution-high | 0.0228 | 0.9100 | -0.675 | 0.6796 | 0.8599 | 1.6958 |
| lca | 3 | substitution-mid | 0.0222 | 0.8739 | -0.264 | 0.6936 | 0.8721 | 1.8439 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0003 | -0.0556 | -0.590 | +0.0104 | +0.0135 |
| 1 | dropout-high | -0.0004 | -0.0642 | -0.181 | +0.0177 | +0.0264 |
| 1 | dropout-mid | -0.0005 | -0.0580 | -0.433 | +0.0133 | +0.0147 |
| 1 | substitution-high | -0.0003 | -0.0562 | -0.472 | +0.0112 | +0.0129 |
| 1 | substitution-mid | -0.0003 | -0.0612 | -0.601 | +0.0099 | +0.0152 |
| 2 | clean | -0.0002 | -0.0323 | -0.314 | +0.0125 | +0.0017 |
| 2 | dropout-high | -0.0003 | -0.0383 | -0.291 | +0.0163 | -0.0217 |
| 2 | dropout-mid | -0.0001 | -0.0407 | -0.434 | +0.0136 | +0.0190 |
| 2 | substitution-high | -0.0001 | -0.0328 | -0.486 | +0.0116 | +0.0307 |
| 2 | substitution-mid | -0.0002 | -0.0343 | -0.268 | +0.0133 | +0.0147 |
| 3 | clean | -0.0004 | -0.0329 | -0.383 | +0.0093 | -0.0211 |
| 3 | dropout-high | -0.0005 | -0.0490 | -0.450 | +0.0168 | +0.0246 |
| 3 | dropout-mid | -0.0007 | -0.0376 | -0.124 | +0.0129 | +0.0251 |
| 3 | substitution-high | -0.0004 | -0.0392 | -0.434 | +0.0111 | +0.0032 |
| 3 | substitution-mid | -0.0004 | -0.0381 | -0.365 | +0.0087 | -0.0135 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.