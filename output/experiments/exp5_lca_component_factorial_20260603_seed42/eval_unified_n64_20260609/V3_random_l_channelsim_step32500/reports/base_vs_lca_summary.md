# Base vs LCA evaluation summary

- Base ckpt: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output\experiments\exp5_lca_component_factorial_20260603_seed42\runs\V3_random_l_channelsim\checkpoints\SpeechTokenizerTrainer_00032500`
  - sha256: `237bb5472f6120b1a80004c9e6eb8bf7c83d043e4381cf7a72bec859c52a0d77`
- Sample count per (L, channel): 256
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0312 | 1.2080 | -6.181 | 0.4546 | 0.7971 | nan |
| base | 1 | dropout-high | 0.0329 | 1.2671 | -7.962 | 0.3861 | 0.7686 | nan |
| base | 1 | dropout-mid | 0.0321 | 1.2380 | -7.065 | 0.4205 | 0.7826 | nan |
| base | 1 | substitution-high | 0.0316 | 1.2581 | -6.584 | 0.4386 | 0.7789 | nan |
| base | 1 | substitution-mid | 0.0313 | 1.2237 | -6.298 | 0.4498 | 0.7913 | nan |
| base | 2 | clean | 0.0239 | 0.9491 | -0.921 | 0.6662 | 0.8596 | nan |
| base | 2 | dropout-high | 0.0271 | 1.0321 | -3.308 | 0.5711 | 0.8223 | nan |
| base | 2 | dropout-mid | 0.0255 | 0.9920 | -2.163 | 0.6181 | 0.8405 | nan |
| base | 2 | substitution-high | 0.0248 | 1.0062 | -1.510 | 0.6435 | 0.8392 | nan |
| base | 2 | substitution-mid | 0.0242 | 0.9687 | -1.151 | 0.6577 | 0.8525 | nan |
| base | 3 | clean | 0.0217 | 0.8831 | +0.654 | 0.7224 | 0.8811 | nan |
| base | 3 | dropout-high | 0.0253 | 0.9729 | -2.047 | 0.6222 | 0.8421 | nan |
| base | 3 | dropout-mid | 0.0236 | 0.9305 | -0.746 | 0.6722 | 0.8607 | nan |
| base | 3 | substitution-high | 0.0227 | 0.9444 | -0.044 | 0.6983 | 0.8595 | nan |
| base | 3 | substitution-mid | 0.0221 | 0.9028 | +0.418 | 0.7144 | 0.8741 | nan |
| lca | 1 | clean | 0.0314 | 1.1818 | -6.943 | 0.4275 | 0.8077 | nan |
| lca | 1 | dropout-high | 0.0328 | 1.2310 | -8.682 | 0.3640 | 0.7838 | nan |
| lca | 1 | dropout-mid | 0.0321 | 1.2072 | -7.774 | 0.3959 | 0.7957 | nan |
| lca | 1 | substitution-high | 0.0318 | 1.2254 | -7.389 | 0.4106 | 0.7888 | nan |
| lca | 1 | substitution-mid | 0.0315 | 1.1957 | -7.088 | 0.4222 | 0.8019 | nan |
| lca | 2 | clean | 0.0241 | 0.9220 | -1.354 | 0.6496 | 0.8683 | nan |
| lca | 2 | dropout-high | 0.0269 | 0.9969 | -3.643 | 0.5581 | 0.8353 | nan |
| lca | 2 | dropout-mid | 0.0255 | 0.9605 | -2.488 | 0.6051 | 0.8518 | nan |
| lca | 2 | substitution-high | 0.0250 | 0.9743 | -2.015 | 0.6239 | 0.8474 | nan |
| lca | 2 | substitution-mid | 0.0244 | 0.9400 | -1.595 | 0.6405 | 0.8610 | nan |
| lca | 3 | clean | 0.0214 | 0.8447 | +0.574 | 0.7198 | 0.8898 | nan |
| lca | 3 | dropout-high | 0.0248 | 0.9264 | -2.061 | 0.6211 | 0.8557 | nan |
| lca | 3 | dropout-mid | 0.0232 | 0.8879 | -0.784 | 0.6707 | 0.8722 | nan |
| lca | 3 | substitution-high | 0.0225 | 0.9016 | -0.193 | 0.6929 | 0.8677 | nan |
| lca | 3 | substitution-mid | 0.0218 | 0.8632 | +0.312 | 0.7107 | 0.8827 | nan |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | +0.0001 | -0.0263 | -0.762 | +0.0106 | +nan |
| 1 | dropout-high | -0.0001 | -0.0361 | -0.720 | +0.0152 | +nan |
| 1 | dropout-mid | +0.0000 | -0.0308 | -0.708 | +0.0131 | +nan |
| 1 | substitution-high | +0.0001 | -0.0327 | -0.805 | +0.0099 | +nan |
| 1 | substitution-mid | +0.0001 | -0.0281 | -0.790 | +0.0106 | +nan |
| 2 | clean | +0.0002 | -0.0271 | -0.432 | +0.0087 | +nan |
| 2 | dropout-high | -0.0001 | -0.0352 | -0.335 | +0.0130 | +nan |
| 2 | dropout-mid | -0.0000 | -0.0315 | -0.325 | +0.0113 | +nan |
| 2 | substitution-high | +0.0002 | -0.0318 | -0.506 | +0.0082 | +nan |
| 2 | substitution-mid | +0.0002 | -0.0287 | -0.444 | +0.0084 | +nan |
| 3 | clean | -0.0003 | -0.0384 | -0.080 | +0.0088 | +nan |
| 3 | dropout-high | -0.0006 | -0.0464 | -0.013 | +0.0136 | +nan |
| 3 | dropout-mid | -0.0004 | -0.0426 | -0.038 | +0.0115 | +nan |
| 3 | substitution-high | -0.0002 | -0.0427 | -0.149 | +0.0082 | +nan |
| 3 | substitution-mid | -0.0003 | -0.0396 | -0.106 | +0.0086 | +nan |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.