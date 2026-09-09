# Base vs LCA evaluation summary

- Base ckpt: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42\checkpoints\SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`
  - sha256: `af60223b2f44733f5a43d5a022871dad061dfe2421473c96f5d350c3d4b49a5a`
- Sample count per (L, channel): 100
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0311 | 1.3915 | -8.775 | 0.3877 | 0.7668 | 1.2596 |
| base | 1 | dropout-high | 0.0323 | 1.4347 | -10.137 | 0.3350 | 0.7424 | 1.2021 |
| base | 1 | dropout-mid | 0.0317 | 1.4142 | -9.463 | 0.3588 | 0.7545 | 1.2256 |
| base | 1 | substitution-high | 0.0314 | 1.4520 | -9.266 | 0.3748 | 0.7491 | 1.2176 |
| base | 1 | substitution-mid | 0.0312 | 1.4054 | -8.886 | 0.3835 | 0.7614 | 1.2453 |
| base | 2 | clean | 0.0246 | 1.1358 | -3.189 | 0.6042 | 0.8289 | 1.5775 |
| base | 2 | dropout-high | 0.0273 | 1.2012 | -5.717 | 0.5144 | 0.7945 | 1.3768 |
| base | 2 | dropout-mid | 0.0260 | 1.1682 | -4.317 | 0.5591 | 0.8118 | 1.4525 |
| base | 2 | substitution-high | 0.0254 | 1.2008 | -3.594 | 0.5810 | 0.8060 | 1.4458 |
| base | 2 | substitution-mid | 0.0248 | 1.1557 | -3.275 | 0.5984 | 0.8222 | 1.5330 |
| base | 3 | clean | 0.0227 | 1.0621 | -1.646 | 0.6614 | 0.8520 | 1.7745 |
| base | 3 | dropout-high | 0.0259 | 1.1371 | -4.126 | 0.5662 | 0.8142 | 1.4773 |
| base | 3 | dropout-mid | 0.0243 | 1.0989 | -3.122 | 0.6167 | 0.8333 | 1.5921 |
| base | 3 | substitution-high | 0.0236 | 1.1333 | -2.404 | 0.6368 | 0.8317 | 1.5808 |
| base | 3 | substitution-mid | 0.0230 | 1.0876 | -1.910 | 0.6528 | 0.8441 | 1.6960 |
| lca | 1 | clean | 0.0310 | 1.3908 | -9.126 | 0.3805 | 0.7804 | 1.2786 |
| lca | 1 | dropout-high | 0.0320 | 1.4302 | -10.510 | 0.3278 | 0.7589 | 1.2182 |
| lca | 1 | dropout-mid | 0.0315 | 1.4099 | -10.076 | 0.3518 | 0.7695 | 1.2443 |
| lca | 1 | substitution-high | 0.0312 | 1.4335 | -9.471 | 0.3665 | 0.7641 | 1.2452 |
| lca | 1 | substitution-mid | 0.0310 | 1.4022 | -9.234 | 0.3755 | 0.7756 | 1.2652 |
| lca | 2 | clean | 0.0248 | 1.1176 | -3.440 | 0.5870 | 0.8384 | 1.5636 |
| lca | 2 | dropout-high | 0.0272 | 1.1730 | -6.417 | 0.4963 | 0.8075 | 1.3813 |
| lca | 2 | dropout-mid | 0.0260 | 1.1456 | -4.533 | 0.5449 | 0.8240 | 1.4582 |
| lca | 2 | substitution-high | 0.0256 | 1.1720 | -4.159 | 0.5629 | 0.8172 | 1.4480 |
| lca | 2 | substitution-mid | 0.0250 | 1.1343 | -3.725 | 0.5805 | 0.8329 | 1.5246 |
| lca | 3 | clean | 0.0225 | 1.0348 | -1.509 | 0.6547 | 0.8602 | 1.7434 |
| lca | 3 | dropout-high | 0.0253 | 1.1010 | -4.118 | 0.5628 | 0.8277 | 1.4825 |
| lca | 3 | dropout-mid | 0.0240 | 1.0682 | -2.710 | 0.6120 | 0.8441 | 1.5875 |
| lca | 3 | substitution-high | 0.0233 | 1.0931 | -2.331 | 0.6298 | 0.8404 | 1.5888 |
| lca | 3 | substitution-mid | 0.0228 | 1.0566 | -1.734 | 0.6468 | 0.8527 | 1.6818 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0002 | -0.0008 | -0.351 | +0.0136 | +0.0190 |
| 1 | dropout-high | -0.0002 | -0.0044 | -0.372 | +0.0165 | +0.0162 |
| 1 | dropout-mid | -0.0002 | -0.0043 | -0.613 | +0.0151 | +0.0187 |
| 1 | substitution-high | -0.0002 | -0.0186 | -0.206 | +0.0149 | +0.0276 |
| 1 | substitution-mid | -0.0002 | -0.0032 | -0.348 | +0.0142 | +0.0199 |
| 2 | clean | +0.0002 | -0.0182 | -0.251 | +0.0095 | -0.0140 |
| 2 | dropout-high | -0.0001 | -0.0282 | -0.700 | +0.0130 | +0.0045 |
| 2 | dropout-mid | +0.0000 | -0.0226 | -0.216 | +0.0122 | +0.0057 |
| 2 | substitution-high | +0.0001 | -0.0288 | -0.565 | +0.0112 | +0.0022 |
| 2 | substitution-mid | +0.0002 | -0.0215 | -0.449 | +0.0106 | -0.0084 |
| 3 | clean | -0.0002 | -0.0273 | +0.138 | +0.0081 | -0.0311 |
| 3 | dropout-high | -0.0006 | -0.0361 | +0.008 | +0.0134 | +0.0052 |
| 3 | dropout-mid | -0.0004 | -0.0307 | +0.412 | +0.0108 | -0.0046 |
| 3 | substitution-high | -0.0002 | -0.0401 | +0.073 | +0.0087 | +0.0081 |
| 3 | substitution-mid | -0.0002 | -0.0310 | +0.176 | +0.0086 | -0.0142 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.