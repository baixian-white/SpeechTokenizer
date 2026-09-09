# Base vs LCA evaluation summary

- Base ckpt: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output\experiments\exp5_lca_component_factorial_20260603_seed42\runs\V1_random_l_only\checkpoints\SpeechTokenizerTrainer_00025000`
  - sha256: `4e0a28183265bb2a0a511b45746b87bcbabbde3d31636fad4f114a1602aa9df5`
- Sample count per (L, channel): 8
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0321 | 1.2022 | -7.183 | 0.4327 | 0.7866 | 1.3330 |
| base | 1 | dropout-high | 0.0336 | 1.2613 | -8.664 | 0.3770 | 0.7522 | 1.2144 |
| base | 1 | dropout-mid | 0.0327 | 1.2317 | -7.842 | 0.4109 | 0.7724 | 1.2757 |
| base | 1 | substitution-high | 0.0325 | 1.2540 | -7.535 | 0.4204 | 0.7709 | 1.2869 |
| base | 1 | substitution-mid | 0.0322 | 1.2098 | -7.359 | 0.4268 | 0.7800 | 1.3230 |
| base | 2 | clean | 0.0246 | 0.9549 | -1.447 | 0.6514 | 0.8442 | 1.7072 |
| base | 2 | dropout-high | 0.0272 | 1.0266 | -3.298 | 0.5830 | 0.8105 | 1.4487 |
| base | 2 | dropout-mid | 0.0260 | 0.9921 | -2.362 | 0.6189 | 0.8269 | 1.5316 |
| base | 2 | substitution-high | 0.0259 | 1.0217 | -2.462 | 0.6166 | 0.8139 | 1.4677 |
| base | 2 | substitution-mid | 0.0250 | 0.9797 | -1.674 | 0.6433 | 0.8384 | 1.6211 |
| base | 3 | clean | 0.0222 | 0.8916 | +0.354 | 0.7089 | 0.8697 | 1.9123 |
| base | 3 | dropout-high | 0.0260 | 0.9806 | -2.383 | 0.6167 | 0.8275 | 1.5561 |
| base | 3 | dropout-mid | 0.0244 | 0.9379 | -1.370 | 0.6544 | 0.8469 | 1.6586 |
| base | 3 | substitution-high | 0.0232 | 0.9436 | -0.523 | 0.6853 | 0.8498 | 1.7171 |
| base | 3 | substitution-mid | 0.0226 | 0.9160 | +0.116 | 0.7026 | 0.8623 | 1.8013 |
| lca | 1 | clean | 0.0318 | 1.1611 | -7.699 | 0.4171 | 0.7977 | 1.3555 |
| lca | 1 | dropout-high | 0.0336 | 1.2173 | -9.588 | 0.3495 | 0.7657 | 1.2213 |
| lca | 1 | dropout-mid | 0.0324 | 1.1893 | -8.474 | 0.3916 | 0.7863 | 1.2973 |
| lca | 1 | substitution-high | 0.0323 | 1.2214 | -8.269 | 0.3985 | 0.7789 | 1.2957 |
| lca | 1 | substitution-mid | 0.0319 | 1.1718 | -7.927 | 0.4102 | 0.7916 | 1.3436 |
| lca | 2 | clean | 0.0242 | 0.9305 | -1.500 | 0.6463 | 0.8561 | 1.6952 |
| lca | 2 | dropout-high | 0.0271 | 1.0062 | -3.646 | 0.5664 | 0.8248 | 1.4387 |
| lca | 2 | dropout-mid | 0.0255 | 0.9669 | -2.565 | 0.6109 | 0.8410 | 1.5307 |
| lca | 2 | substitution-high | 0.0255 | 1.0137 | -2.470 | 0.6133 | 0.8236 | 1.4326 |
| lca | 2 | substitution-mid | 0.0246 | 0.9539 | -1.746 | 0.6378 | 0.8488 | 1.6230 |
| lca | 3 | clean | 0.0218 | 0.8567 | +0.180 | 0.7047 | 0.8815 | 1.9101 |
| lca | 3 | dropout-high | 0.0253 | 0.9362 | -2.513 | 0.6121 | 0.8443 | 1.5920 |
| lca | 3 | dropout-mid | 0.0241 | 0.9016 | -1.651 | 0.6435 | 0.8582 | 1.6618 |
| lca | 3 | substitution-high | 0.0231 | 0.9253 | -0.766 | 0.6766 | 0.8566 | 1.6621 |
| lca | 3 | substitution-mid | 0.0223 | 0.8884 | -0.105 | 0.6967 | 0.8727 | 1.7952 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0004 | -0.0411 | -0.517 | +0.0111 | +0.0224 |
| 1 | dropout-high | -0.0001 | -0.0440 | -0.924 | +0.0135 | +0.0069 |
| 1 | dropout-mid | -0.0003 | -0.0423 | -0.631 | +0.0140 | +0.0216 |
| 1 | substitution-high | -0.0002 | -0.0325 | -0.735 | +0.0080 | +0.0087 |
| 1 | substitution-mid | -0.0004 | -0.0380 | -0.568 | +0.0116 | +0.0206 |
| 2 | clean | -0.0004 | -0.0244 | -0.054 | +0.0120 | -0.0120 |
| 2 | dropout-high | -0.0001 | -0.0203 | -0.348 | +0.0142 | -0.0101 |
| 2 | dropout-mid | -0.0005 | -0.0252 | -0.203 | +0.0141 | -0.0008 |
| 2 | substitution-high | -0.0004 | -0.0080 | -0.008 | +0.0097 | -0.0350 |
| 2 | substitution-mid | -0.0004 | -0.0258 | -0.072 | +0.0104 | +0.0019 |
| 3 | clean | -0.0004 | -0.0349 | -0.174 | +0.0118 | -0.0022 |
| 3 | dropout-high | -0.0007 | -0.0444 | -0.130 | +0.0168 | +0.0360 |
| 3 | dropout-mid | -0.0004 | -0.0363 | -0.281 | +0.0113 | +0.0033 |
| 3 | substitution-high | -0.0002 | -0.0183 | -0.243 | +0.0068 | -0.0550 |
| 3 | substitution-mid | -0.0003 | -0.0276 | -0.221 | +0.0104 | -0.0061 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.