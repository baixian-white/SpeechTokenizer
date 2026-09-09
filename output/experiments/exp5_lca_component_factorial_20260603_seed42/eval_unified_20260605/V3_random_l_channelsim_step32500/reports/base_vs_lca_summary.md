# Base vs LCA evaluation summary

- Base ckpt: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output\experiments\exp5_lca_component_factorial_20260603_seed42\runs\V3_random_l_channelsim\checkpoints\SpeechTokenizerTrainer_00032500`
  - sha256: `237bb5472f6120b1a80004c9e6eb8bf7c83d043e4381cf7a72bec859c52a0d77`
- Sample count per (L, channel): 8
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0321 | 1.2022 | -7.183 | 0.4327 | 0.7866 | 1.3330 |
| base | 1 | dropout-high | 0.0331 | 1.2490 | -8.174 | 0.3996 | 0.7642 | 1.2607 |
| base | 1 | dropout-mid | 0.0330 | 1.2325 | -8.056 | 0.4039 | 0.7712 | 1.2680 |
| base | 1 | substitution-high | 0.0325 | 1.2436 | -7.626 | 0.4156 | 0.7671 | 1.2491 |
| base | 1 | substitution-mid | 0.0322 | 1.2184 | -7.266 | 0.4293 | 0.7796 | 1.3204 |
| base | 2 | clean | 0.0246 | 0.9549 | -1.447 | 0.6514 | 0.8442 | 1.7072 |
| base | 2 | dropout-high | 0.0283 | 1.0336 | -4.254 | 0.5425 | 0.8060 | 1.4179 |
| base | 2 | dropout-mid | 0.0259 | 0.9871 | -2.412 | 0.6209 | 0.8283 | 1.5899 |
| base | 2 | substitution-high | 0.0254 | 1.0024 | -2.144 | 0.6286 | 0.8270 | 1.5810 |
| base | 2 | substitution-mid | 0.0247 | 0.9671 | -1.532 | 0.6481 | 0.8403 | 1.6641 |
| base | 3 | clean | 0.0222 | 0.8916 | +0.354 | 0.7089 | 0.8697 | 1.9123 |
| base | 3 | dropout-high | 0.0259 | 0.9841 | -2.380 | 0.6171 | 0.8253 | 1.4636 |
| base | 3 | dropout-mid | 0.0241 | 0.9304 | -1.246 | 0.6568 | 0.8501 | 1.7417 |
| base | 3 | substitution-high | 0.0230 | 0.9561 | -0.089 | 0.6940 | 0.8514 | 1.7308 |
| base | 3 | substitution-mid | 0.0226 | 0.9139 | +0.147 | 0.7028 | 0.8603 | 1.7771 |
| lca | 1 | clean | 0.0319 | 1.1528 | -7.388 | 0.4216 | 0.7975 | 1.3459 |
| lca | 1 | dropout-high | 0.0328 | 1.1936 | -8.416 | 0.3882 | 0.7800 | 1.2875 |
| lca | 1 | dropout-mid | 0.0325 | 1.1764 | -8.174 | 0.3967 | 0.7878 | 1.2929 |
| lca | 1 | substitution-high | 0.0324 | 1.1961 | -7.875 | 0.4035 | 0.7744 | 1.2710 |
| lca | 1 | substitution-mid | 0.0320 | 1.1667 | -7.569 | 0.4152 | 0.7900 | 1.3317 |
| lca | 2 | clean | 0.0244 | 0.9255 | -1.566 | 0.6446 | 0.8563 | 1.7294 |
| lca | 2 | dropout-high | 0.0281 | 1.0030 | -4.510 | 0.5291 | 0.8225 | 1.4372 |
| lca | 2 | dropout-mid | 0.0256 | 0.9555 | -2.463 | 0.6148 | 0.8427 | 1.5735 |
| lca | 2 | substitution-high | 0.0253 | 0.9749 | -2.276 | 0.6179 | 0.8355 | 1.6015 |
| lca | 2 | substitution-mid | 0.0246 | 0.9391 | -1.701 | 0.6396 | 0.8514 | 1.6993 |
| lca | 3 | clean | 0.0217 | 0.8548 | +0.371 | 0.7096 | 0.8784 | 1.9134 |
| lca | 3 | dropout-high | 0.0253 | 0.9394 | -2.493 | 0.6143 | 0.8406 | 1.5361 |
| lca | 3 | dropout-mid | 0.0235 | 0.8888 | -1.258 | 0.6559 | 0.8626 | 1.7355 |
| lca | 3 | substitution-high | 0.0227 | 0.9175 | -0.339 | 0.6872 | 0.8605 | 1.7593 |
| lca | 3 | substitution-mid | 0.0222 | 0.8775 | +0.114 | 0.7015 | 0.8696 | 1.7856 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0002 | -0.0494 | -0.205 | +0.0109 | +0.0129 |
| 1 | dropout-high | -0.0003 | -0.0553 | -0.243 | +0.0158 | +0.0268 |
| 1 | dropout-mid | -0.0004 | -0.0561 | -0.117 | +0.0166 | +0.0249 |
| 1 | substitution-high | -0.0001 | -0.0475 | -0.249 | +0.0073 | +0.0219 |
| 1 | substitution-mid | -0.0002 | -0.0516 | -0.303 | +0.0104 | +0.0113 |
| 2 | clean | -0.0002 | -0.0294 | -0.120 | +0.0121 | +0.0222 |
| 2 | dropout-high | -0.0002 | -0.0306 | -0.256 | +0.0165 | +0.0193 |
| 2 | dropout-mid | -0.0003 | -0.0316 | -0.051 | +0.0144 | -0.0163 |
| 2 | substitution-high | -0.0000 | -0.0275 | -0.133 | +0.0085 | +0.0205 |
| 2 | substitution-mid | -0.0001 | -0.0280 | -0.169 | +0.0111 | +0.0353 |
| 3 | clean | -0.0005 | -0.0368 | +0.017 | +0.0087 | +0.0010 |
| 3 | dropout-high | -0.0005 | -0.0447 | -0.114 | +0.0153 | +0.0725 |
| 3 | dropout-mid | -0.0006 | -0.0416 | -0.012 | +0.0125 | -0.0063 |
| 3 | substitution-high | -0.0003 | -0.0386 | -0.251 | +0.0090 | +0.0285 |
| 3 | substitution-mid | -0.0004 | -0.0364 | -0.033 | +0.0093 | +0.0085 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.