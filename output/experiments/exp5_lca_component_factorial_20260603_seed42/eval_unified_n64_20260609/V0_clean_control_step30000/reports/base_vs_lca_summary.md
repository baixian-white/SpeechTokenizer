# Base vs LCA evaluation summary

- Base ckpt: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output/experiments/exp5_lca_component_factorial_20260603_seed42/runs/V0_full_depth_clean_control/checkpoints/SpeechTokenizerTrainer_00030000`
  - sha256: `44f1a67fd123eff622e88da26eed9afe9ba54718c693b59c1af0675fe452c708`
- Sample count per (L, channel): 256
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0312 | 1.2080 | -6.181 | 0.4546 | 0.7971 | 1.3190 |
| base | 1 | dropout-high | 0.0329 | 1.2675 | -8.006 | 0.3859 | 0.7684 | 1.2342 |
| base | 1 | dropout-mid | 0.0321 | 1.2393 | -7.071 | 0.4200 | 0.7823 | 1.2519 |
| base | 1 | substitution-high | 0.0316 | 1.2546 | -6.550 | 0.4399 | 0.7799 | 1.2673 |
| base | 1 | substitution-mid | 0.0313 | 1.2239 | -6.309 | 0.4493 | 0.7911 | 1.2836 |
| base | 2 | clean | 0.0239 | 0.9491 | -0.921 | 0.6662 | 0.8596 | 1.7457 |
| base | 2 | dropout-high | 0.0271 | 1.0322 | -3.322 | 0.5709 | 0.8221 | 1.4290 |
| base | 2 | dropout-mid | 0.0255 | 0.9928 | -2.199 | 0.6164 | 0.8402 | 1.5538 |
| base | 2 | substitution-high | 0.0247 | 1.0082 | -1.514 | 0.6432 | 0.8389 | 1.5796 |
| base | 2 | substitution-mid | 0.0242 | 0.9682 | -1.137 | 0.6581 | 0.8524 | 1.6760 |
| base | 3 | clean | 0.0217 | 0.8831 | +0.654 | 0.7224 | 0.8811 | 1.9813 |
| base | 3 | dropout-high | 0.0253 | 0.9742 | -1.985 | 0.6254 | 0.8413 | 1.5685 |
| base | 3 | dropout-mid | 0.0236 | 0.9308 | -0.713 | 0.6732 | 0.8610 | 1.7092 |
| base | 3 | substitution-high | 0.0228 | 0.9458 | -0.096 | 0.6966 | 0.8594 | 1.7721 |
| base | 3 | substitution-mid | 0.0221 | 0.9049 | +0.398 | 0.7140 | 0.8736 | 1.8789 |
| lca | 1 | clean | 0.0308 | 1.1481 | -6.046 | 0.4597 | 0.8135 | 1.3509 |
| lca | 1 | dropout-high | 0.0325 | 1.2074 | -7.885 | 0.3904 | 0.7850 | 1.2562 |
| lca | 1 | dropout-mid | 0.0317 | 1.1789 | -6.953 | 0.4248 | 0.7989 | 1.2788 |
| lca | 1 | substitution-high | 0.0312 | 1.1973 | -6.419 | 0.4448 | 0.7955 | 1.2919 |
| lca | 1 | substitution-mid | 0.0310 | 1.1649 | -6.185 | 0.4540 | 0.8072 | 1.3136 |
| lca | 2 | clean | 0.0233 | 0.8768 | -0.662 | 0.6758 | 0.8767 | 1.8347 |
| lca | 2 | dropout-high | 0.0265 | 0.9612 | -3.107 | 0.5794 | 0.8397 | 1.4775 |
| lca | 2 | dropout-mid | 0.0249 | 0.9215 | -1.943 | 0.6261 | 0.8572 | 1.5986 |
| lca | 2 | substitution-high | 0.0242 | 0.9396 | -1.302 | 0.6518 | 0.8552 | 1.6251 |
| lca | 2 | substitution-mid | 0.0236 | 0.8973 | -0.876 | 0.6679 | 0.8692 | 1.7520 |
| lca | 3 | clean | 0.0208 | 0.8061 | +1.136 | 0.7387 | 0.8978 | 2.1010 |
| lca | 3 | dropout-high | 0.0244 | 0.8985 | -1.613 | 0.6388 | 0.8589 | 1.6287 |
| lca | 3 | dropout-mid | 0.0226 | 0.8548 | -0.311 | 0.6882 | 0.8776 | 1.7572 |
| lca | 3 | substitution-high | 0.0218 | 0.8711 | +0.344 | 0.7123 | 0.8757 | 1.8406 |
| lca | 3 | substitution-mid | 0.0211 | 0.8291 | +0.863 | 0.7300 | 0.8902 | 1.9739 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0004 | -0.0599 | +0.135 | +0.0164 | +0.0319 |
| 1 | dropout-high | -0.0004 | -0.0600 | +0.121 | +0.0166 | +0.0220 |
| 1 | dropout-mid | -0.0004 | -0.0604 | +0.118 | +0.0166 | +0.0269 |
| 1 | substitution-high | -0.0004 | -0.0573 | +0.131 | +0.0155 | +0.0246 |
| 1 | substitution-mid | -0.0004 | -0.0590 | +0.125 | +0.0161 | +0.0300 |
| 2 | clean | -0.0006 | -0.0723 | +0.259 | +0.0171 | +0.0891 |
| 2 | dropout-high | -0.0006 | -0.0710 | +0.215 | +0.0175 | +0.0485 |
| 2 | dropout-mid | -0.0006 | -0.0713 | +0.256 | +0.0170 | +0.0448 |
| 2 | substitution-high | -0.0005 | -0.0686 | +0.212 | +0.0163 | +0.0455 |
| 2 | substitution-mid | -0.0006 | -0.0709 | +0.261 | +0.0169 | +0.0760 |
| 3 | clean | -0.0010 | -0.0770 | +0.483 | +0.0167 | +0.1197 |
| 3 | dropout-high | -0.0009 | -0.0757 | +0.372 | +0.0176 | +0.0602 |
| 3 | dropout-mid | -0.0009 | -0.0760 | +0.403 | +0.0167 | +0.0479 |
| 3 | substitution-high | -0.0009 | -0.0747 | +0.440 | +0.0163 | +0.0685 |
| 3 | substitution-mid | -0.0009 | -0.0758 | +0.465 | +0.0166 | +0.0950 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.