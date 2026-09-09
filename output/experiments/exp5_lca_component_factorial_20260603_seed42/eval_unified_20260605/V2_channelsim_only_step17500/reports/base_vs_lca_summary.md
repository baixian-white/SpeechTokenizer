# Base vs LCA evaluation summary

- Base ckpt: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output\experiments\exp5_lca_component_factorial_20260603_seed42\runs\V2_channelsim_only\checkpoints\SpeechTokenizerTrainer_00017500`
  - sha256: `b59e3282a46b132ed0432e4fc01663439714664f8f154c4b7c1835905739a03f`
- Sample count per (L, channel): 8
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0321 | 1.2022 | -7.183 | 0.4327 | 0.7866 | 1.3330 |
| base | 1 | dropout-high | 0.0335 | 1.2589 | -8.716 | 0.3788 | 0.7585 | 1.2185 |
| base | 1 | dropout-mid | 0.0330 | 1.2313 | -8.087 | 0.4002 | 0.7684 | 1.2397 |
| base | 1 | substitution-high | 0.0324 | 1.2382 | -7.400 | 0.4244 | 0.7736 | 1.2997 |
| base | 1 | substitution-mid | 0.0322 | 1.2152 | -7.253 | 0.4303 | 0.7833 | 1.3201 |
| base | 2 | clean | 0.0246 | 0.9549 | -1.447 | 0.6514 | 0.8442 | 1.7072 |
| base | 2 | dropout-high | 0.0274 | 1.0312 | -3.342 | 0.5808 | 0.8064 | 1.4544 |
| base | 2 | dropout-mid | 0.0262 | 0.9905 | -2.615 | 0.6065 | 0.8274 | 1.5495 |
| base | 2 | substitution-high | 0.0255 | 1.0121 | -2.227 | 0.6248 | 0.8234 | 1.4885 |
| base | 2 | substitution-mid | 0.0249 | 0.9779 | -1.673 | 0.6428 | 0.8350 | 1.6473 |
| base | 3 | clean | 0.0222 | 0.8916 | +0.354 | 0.7089 | 0.8697 | 1.9123 |
| base | 3 | dropout-high | 0.0262 | 0.9832 | -2.566 | 0.6129 | 0.8249 | 1.4813 |
| base | 3 | dropout-mid | 0.0239 | 0.9342 | -0.625 | 0.6779 | 0.8469 | 1.6792 |
| base | 3 | substitution-high | 0.0232 | 0.9408 | -0.337 | 0.6877 | 0.8491 | 1.7452 |
| base | 3 | substitution-mid | 0.0226 | 0.9095 | +0.180 | 0.7038 | 0.8633 | 1.8657 |
| lca | 1 | clean | 0.0321 | 1.2226 | -7.554 | 0.4173 | 0.7872 | 1.3174 |
| lca | 1 | dropout-high | 0.0336 | 1.2773 | -9.347 | 0.3570 | 0.7602 | 1.2243 |
| lca | 1 | dropout-mid | 0.0330 | 1.2511 | -8.432 | 0.3860 | 0.7689 | 1.2324 |
| lca | 1 | substitution-high | 0.0325 | 1.2539 | -7.835 | 0.4069 | 0.7714 | 1.2898 |
| lca | 1 | substitution-mid | 0.0322 | 1.2305 | -7.566 | 0.4161 | 0.7835 | 1.3096 |
| lca | 2 | clean | 0.0246 | 0.9550 | -1.815 | 0.6377 | 0.8492 | 1.6666 |
| lca | 2 | dropout-high | 0.0272 | 1.0195 | -3.726 | 0.5690 | 0.8164 | 1.4703 |
| lca | 2 | dropout-mid | 0.0263 | 0.9874 | -3.054 | 0.5911 | 0.8350 | 1.5272 |
| lca | 2 | substitution-high | 0.0257 | 1.0191 | -2.497 | 0.6116 | 0.8242 | 1.4829 |
| lca | 2 | substitution-mid | 0.0251 | 0.9804 | -2.171 | 0.6240 | 0.8388 | 1.6175 |
| lca | 3 | clean | 0.0218 | 0.8778 | +0.281 | 0.7090 | 0.8761 | 1.9142 |
| lca | 3 | dropout-high | 0.0255 | 0.9566 | -2.438 | 0.6145 | 0.8384 | 1.5121 |
| lca | 3 | dropout-mid | 0.0233 | 0.9113 | -0.722 | 0.6774 | 0.8584 | 1.7357 |
| lca | 3 | substitution-high | 0.0229 | 0.9286 | -0.375 | 0.6881 | 0.8519 | 1.7423 |
| lca | 3 | substitution-mid | 0.0223 | 0.9000 | -0.083 | 0.6980 | 0.8680 | 1.8464 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | +0.0000 | +0.0204 | -0.371 | +0.0006 | -0.0157 |
| 1 | dropout-high | +0.0001 | +0.0184 | -0.631 | +0.0017 | +0.0058 |
| 1 | dropout-mid | -0.0000 | +0.0197 | -0.344 | +0.0005 | -0.0073 |
| 1 | substitution-high | +0.0001 | +0.0156 | -0.436 | -0.0022 | -0.0098 |
| 1 | substitution-mid | +0.0000 | +0.0152 | -0.313 | +0.0003 | -0.0105 |
| 2 | clean | +0.0000 | +0.0001 | -0.368 | +0.0050 | -0.0407 |
| 2 | dropout-high | -0.0002 | -0.0117 | -0.384 | +0.0100 | +0.0159 |
| 2 | dropout-mid | +0.0001 | -0.0031 | -0.438 | +0.0075 | -0.0224 |
| 2 | substitution-high | +0.0001 | +0.0070 | -0.270 | +0.0008 | -0.0057 |
| 2 | substitution-mid | +0.0001 | +0.0025 | -0.498 | +0.0038 | -0.0298 |
| 3 | clean | -0.0004 | -0.0138 | -0.072 | +0.0064 | +0.0019 |
| 3 | dropout-high | -0.0008 | -0.0266 | +0.128 | +0.0135 | +0.0308 |
| 3 | dropout-mid | -0.0006 | -0.0230 | -0.096 | +0.0115 | +0.0565 |
| 3 | substitution-high | -0.0003 | -0.0121 | -0.038 | +0.0028 | -0.0029 |
| 3 | substitution-mid | -0.0003 | -0.0094 | -0.263 | +0.0046 | -0.0193 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.