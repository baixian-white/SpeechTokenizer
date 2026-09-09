# Base vs LCA evaluation summary

- Base ckpt: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output/experiments/exp3_low_load_channel_aware_adaptation_20260530_seed42/checkpoints/SCIT-Speech-LCA_best.pt`
  - sha256: `9daa924ff5aeedf2fa4aa1ddb2d7ce7f53297548e7258d0ef8d5b1af255a8609`
- Sample count per (L, channel): 8
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0321 | 1.2022 | -7.183 | 0.4327 | 0.7866 | 1.3330 |
| base | 1 | dropout-low | 0.0323 | 1.2075 | -7.347 | 0.4260 | 0.7828 | 1.3218 |
| base | 1 | dropout-mid | 0.0327 | 1.2160 | -7.703 | 0.4136 | 0.7790 | 1.2894 |
| base | 1 | substitution-low | 0.0322 | 1.2053 | -7.208 | 0.4318 | 0.7861 | 1.3258 |
| base | 1 | substitution-mid | 0.0322 | 1.2112 | -7.241 | 0.4306 | 0.7824 | 1.3176 |
| base | 2 | clean | 0.0246 | 0.9549 | -1.447 | 0.6514 | 0.8442 | 1.7072 |
| base | 2 | dropout-low | 0.0251 | 0.9640 | -1.883 | 0.6371 | 0.8382 | 1.6311 |
| base | 2 | dropout-mid | 0.0256 | 0.9791 | -2.095 | 0.6261 | 0.8321 | 1.5991 |
| base | 2 | substitution-low | 0.0246 | 0.9554 | -1.447 | 0.6514 | 0.8439 | 1.7036 |
| base | 2 | substitution-mid | 0.0249 | 0.9632 | -1.761 | 0.6407 | 0.8409 | 1.6650 |
| base | 3 | clean | 0.0222 | 0.8916 | +0.354 | 0.7089 | 0.8697 | 1.9123 |
| base | 3 | dropout-low | 0.0225 | 0.9001 | +0.203 | 0.7039 | 0.8672 | 1.8711 |
| base | 3 | dropout-mid | 0.0234 | 0.9112 | -0.524 | 0.6827 | 0.8583 | 1.8066 |
| base | 3 | substitution-low | 0.0223 | 0.8966 | +0.327 | 0.7081 | 0.8688 | 1.8613 |
| base | 3 | substitution-mid | 0.0224 | 0.9017 | +0.242 | 0.7053 | 0.8666 | 1.8790 |
| lca | 1 | clean | 0.0317 | 1.1292 | -7.182 | 0.4309 | 0.8029 | 1.3384 |
| lca | 1 | dropout-low | 0.0319 | 1.1354 | -7.360 | 0.4239 | 0.7996 | 1.3284 |
| lca | 1 | dropout-mid | 0.0321 | 1.1440 | -7.671 | 0.4133 | 0.7947 | 1.3034 |
| lca | 1 | substitution-low | 0.0317 | 1.1329 | -7.209 | 0.4298 | 0.8022 | 1.3313 |
| lca | 1 | substitution-mid | 0.0319 | 1.1401 | -7.322 | 0.4258 | 0.7974 | 1.3197 |
| lca | 2 | clean | 0.0240 | 0.9087 | -1.158 | 0.6590 | 0.8613 | 1.7151 |
| lca | 2 | dropout-low | 0.0246 | 0.9185 | -1.699 | 0.6420 | 0.8557 | 1.6132 |
| lca | 2 | dropout-mid | 0.0250 | 0.9308 | -1.808 | 0.6343 | 0.8495 | 1.6223 |
| lca | 2 | substitution-low | 0.0240 | 0.9095 | -1.169 | 0.6586 | 0.8610 | 1.7114 |
| lca | 2 | substitution-mid | 0.0242 | 0.9198 | -1.446 | 0.6498 | 0.8566 | 1.6431 |
| lca | 3 | clean | 0.0215 | 0.8348 | +0.672 | 0.7206 | 0.8843 | 1.9340 |
| lca | 3 | dropout-low | 0.0218 | 0.8408 | +0.434 | 0.7131 | 0.8818 | 1.8976 |
| lca | 3 | dropout-mid | 0.0226 | 0.8532 | -0.179 | 0.6960 | 0.8759 | 1.8451 |
| lca | 3 | substitution-low | 0.0216 | 0.8401 | +0.637 | 0.7196 | 0.8828 | 1.8945 |
| lca | 3 | substitution-mid | 0.0217 | 0.8456 | +0.573 | 0.7174 | 0.8805 | 1.8779 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0004 | -0.0730 | +0.001 | +0.0163 | +0.0053 |
| 1 | dropout-low | -0.0004 | -0.0721 | -0.013 | +0.0168 | +0.0065 |
| 1 | dropout-mid | -0.0005 | -0.0721 | +0.033 | +0.0157 | +0.0141 |
| 1 | substitution-low | -0.0004 | -0.0724 | -0.001 | +0.0161 | +0.0055 |
| 1 | substitution-mid | -0.0004 | -0.0711 | -0.082 | +0.0149 | +0.0021 |
| 2 | clean | -0.0006 | -0.0462 | +0.288 | +0.0171 | +0.0078 |
| 2 | dropout-low | -0.0005 | -0.0455 | +0.184 | +0.0175 | -0.0179 |
| 2 | dropout-mid | -0.0006 | -0.0483 | +0.287 | +0.0174 | +0.0232 |
| 2 | substitution-low | -0.0006 | -0.0459 | +0.279 | +0.0172 | +0.0078 |
| 2 | substitution-mid | -0.0007 | -0.0434 | +0.315 | +0.0157 | -0.0219 |
| 3 | clean | -0.0007 | -0.0568 | +0.318 | +0.0146 | +0.0217 |
| 3 | dropout-low | -0.0007 | -0.0593 | +0.231 | +0.0146 | +0.0265 |
| 3 | dropout-mid | -0.0008 | -0.0580 | +0.345 | +0.0176 | +0.0385 |
| 3 | substitution-low | -0.0007 | -0.0565 | +0.310 | +0.0140 | +0.0333 |
| 3 | substitution-mid | -0.0008 | -0.0561 | +0.331 | +0.0139 | -0.0011 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.