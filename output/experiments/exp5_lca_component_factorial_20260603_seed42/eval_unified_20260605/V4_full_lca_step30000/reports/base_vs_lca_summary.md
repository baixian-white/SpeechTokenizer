# Base vs LCA evaluation summary

- Base ckpt: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42\checkpoints\SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`
  - sha256: `af60223b2f44733f5a43d5a022871dad061dfe2421473c96f5d350c3d4b49a5a`
- Sample count per (L, channel): 8
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0321 | 1.2022 | -7.183 | 0.4327 | 0.7866 | 1.3330 |
| base | 1 | dropout-high | 0.0338 | 1.2533 | -9.247 | 0.3605 | 0.7584 | 1.2209 |
| base | 1 | dropout-mid | 0.0327 | 1.2253 | -7.821 | 0.4108 | 0.7734 | 1.2770 |
| base | 1 | substitution-high | 0.0324 | 1.2345 | -7.394 | 0.4252 | 0.7739 | 1.2988 |
| base | 1 | substitution-mid | 0.0322 | 1.2108 | -7.257 | 0.4299 | 0.7843 | 1.3241 |
| base | 2 | clean | 0.0246 | 0.9549 | -1.447 | 0.6514 | 0.8442 | 1.7072 |
| base | 2 | dropout-high | 0.0281 | 1.0288 | -4.066 | 0.5473 | 0.8066 | 1.4171 |
| base | 2 | dropout-mid | 0.0262 | 0.9951 | -2.683 | 0.6058 | 0.8250 | 1.5201 |
| base | 2 | substitution-high | 0.0255 | 1.0011 | -1.971 | 0.6323 | 0.8248 | 1.5340 |
| base | 2 | substitution-mid | 0.0248 | 0.9740 | -1.610 | 0.6455 | 0.8391 | 1.6266 |
| base | 3 | clean | 0.0222 | 0.8916 | +0.354 | 0.7089 | 0.8697 | 1.9123 |
| base | 3 | dropout-high | 0.0256 | 0.9714 | -2.191 | 0.6222 | 0.8271 | 1.5698 |
| base | 3 | dropout-mid | 0.0241 | 0.9356 | -1.051 | 0.6658 | 0.8475 | 1.7260 |
| base | 3 | substitution-high | 0.0236 | 0.9571 | -0.543 | 0.6814 | 0.8404 | 1.6492 |
| base | 3 | substitution-mid | 0.0225 | 0.9072 | +0.126 | 0.7025 | 0.8629 | 1.8450 |
| lca | 1 | clean | 0.0318 | 1.1466 | -7.772 | 0.4146 | 0.7970 | 1.3466 |
| lca | 1 | dropout-high | 0.0333 | 1.1918 | -9.813 | 0.3419 | 0.7741 | 1.2517 |
| lca | 1 | dropout-mid | 0.0323 | 1.1614 | -8.331 | 0.3945 | 0.7878 | 1.3175 |
| lca | 1 | substitution-high | 0.0321 | 1.1803 | -8.077 | 0.4051 | 0.7825 | 1.2860 |
| lca | 1 | substitution-mid | 0.0319 | 1.1553 | -7.821 | 0.4126 | 0.7943 | 1.3381 |
| lca | 2 | clean | 0.0244 | 0.9226 | -1.761 | 0.6388 | 0.8567 | 1.7089 |
| lca | 2 | dropout-high | 0.0274 | 0.9887 | -3.980 | 0.5517 | 0.8247 | 1.4313 |
| lca | 2 | dropout-mid | 0.0258 | 0.9551 | -2.740 | 0.6024 | 0.8392 | 1.5597 |
| lca | 2 | substitution-high | 0.0252 | 0.9665 | -2.229 | 0.6218 | 0.8381 | 1.5191 |
| lca | 2 | substitution-mid | 0.0247 | 0.9375 | -1.963 | 0.6312 | 0.8506 | 1.6427 |
| lca | 3 | clean | 0.0219 | 0.8587 | -0.029 | 0.7000 | 0.8790 | 1.8913 |
| lca | 3 | dropout-high | 0.0246 | 0.9273 | -2.183 | 0.6290 | 0.8488 | 1.6133 |
| lca | 3 | dropout-mid | 0.0236 | 0.8941 | -1.236 | 0.6600 | 0.8622 | 1.7168 |
| lca | 3 | substitution-high | 0.0230 | 0.9160 | -0.834 | 0.6742 | 0.8532 | 1.6446 |
| lca | 3 | substitution-mid | 0.0222 | 0.8774 | -0.237 | 0.6939 | 0.8728 | 1.8288 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0003 | -0.0556 | -0.590 | +0.0104 | +0.0135 |
| 1 | dropout-high | -0.0005 | -0.0615 | -0.567 | +0.0156 | +0.0308 |
| 1 | dropout-mid | -0.0004 | -0.0639 | -0.509 | +0.0144 | +0.0405 |
| 1 | substitution-high | -0.0003 | -0.0542 | -0.682 | +0.0086 | -0.0128 |
| 1 | substitution-mid | -0.0003 | -0.0555 | -0.564 | +0.0101 | +0.0141 |
| 2 | clean | -0.0002 | -0.0323 | -0.314 | +0.0125 | +0.0017 |
| 2 | dropout-high | -0.0007 | -0.0401 | +0.086 | +0.0181 | +0.0142 |
| 2 | dropout-mid | -0.0004 | -0.0400 | -0.057 | +0.0143 | +0.0396 |
| 2 | substitution-high | -0.0002 | -0.0345 | -0.258 | +0.0133 | -0.0149 |
| 2 | substitution-mid | -0.0001 | -0.0366 | -0.353 | +0.0115 | +0.0161 |
| 3 | clean | -0.0004 | -0.0329 | -0.383 | +0.0093 | -0.0211 |
| 3 | dropout-high | -0.0009 | -0.0441 | +0.008 | +0.0217 | +0.0435 |
| 3 | dropout-mid | -0.0005 | -0.0416 | -0.185 | +0.0146 | -0.0092 |
| 3 | substitution-high | -0.0005 | -0.0411 | -0.291 | +0.0128 | -0.0045 |
| 3 | substitution-mid | -0.0004 | -0.0298 | -0.364 | +0.0100 | -0.0162 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.