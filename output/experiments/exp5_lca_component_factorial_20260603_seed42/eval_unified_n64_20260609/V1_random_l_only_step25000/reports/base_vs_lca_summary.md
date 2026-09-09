# Base vs LCA evaluation summary

- Base ckpt: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output\experiments\exp5_lca_component_factorial_20260603_seed42\runs\V1_random_l_only\checkpoints\SpeechTokenizerTrainer_00025000`
  - sha256: `4e0a28183265bb2a0a511b45746b87bcbabbde3d31636fad4f114a1602aa9df5`
- Sample count per (L, channel): 256
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0312 | 1.2080 | -6.181 | 0.4546 | 0.7971 | nan |
| base | 1 | dropout-high | 0.0328 | 1.2675 | -7.922 | 0.3878 | 0.7690 | nan |
| base | 1 | dropout-mid | 0.0320 | 1.2384 | -7.005 | 0.4225 | 0.7830 | nan |
| base | 1 | substitution-high | 0.0316 | 1.2541 | -6.560 | 0.4399 | 0.7800 | nan |
| base | 1 | substitution-mid | 0.0313 | 1.2233 | -6.300 | 0.4495 | 0.7913 | nan |
| base | 2 | clean | 0.0239 | 0.9491 | -0.921 | 0.6662 | 0.8596 | nan |
| base | 2 | dropout-high | 0.0270 | 1.0305 | -3.263 | 0.5739 | 0.8230 | nan |
| base | 2 | dropout-mid | 0.0255 | 0.9919 | -2.169 | 0.6176 | 0.8403 | nan |
| base | 2 | substitution-high | 0.0248 | 1.0072 | -1.543 | 0.6427 | 0.8383 | nan |
| base | 2 | substitution-mid | 0.0242 | 0.9692 | -1.129 | 0.6583 | 0.8526 | nan |
| base | 3 | clean | 0.0217 | 0.8831 | +0.654 | 0.7224 | 0.8811 | nan |
| base | 3 | dropout-high | 0.0253 | 0.9742 | -1.967 | 0.6253 | 0.8423 | nan |
| base | 3 | dropout-mid | 0.0235 | 0.9288 | -0.709 | 0.6732 | 0.8617 | nan |
| base | 3 | substitution-high | 0.0228 | 0.9444 | -0.076 | 0.6968 | 0.8594 | nan |
| base | 3 | substitution-mid | 0.0220 | 0.9033 | +0.423 | 0.7148 | 0.8742 | nan |
| lca | 1 | clean | 0.0314 | 1.1732 | -6.820 | 0.4313 | 0.8105 | nan |
| lca | 1 | dropout-high | 0.0328 | 1.2292 | -8.522 | 0.3687 | 0.7849 | nan |
| lca | 1 | dropout-mid | 0.0321 | 1.2031 | -7.647 | 0.4002 | 0.7977 | nan |
| lca | 1 | substitution-high | 0.0318 | 1.2272 | -7.306 | 0.4133 | 0.7898 | nan |
| lca | 1 | substitution-mid | 0.0315 | 1.1908 | -6.976 | 0.4255 | 0.8037 | nan |
| lca | 2 | clean | 0.0239 | 0.9223 | -1.160 | 0.6567 | 0.8691 | nan |
| lca | 2 | dropout-high | 0.0269 | 1.0015 | -3.471 | 0.5647 | 0.8359 | nan |
| lca | 2 | dropout-mid | 0.0254 | 0.9636 | -2.384 | 0.6088 | 0.8514 | nan |
| lca | 2 | substitution-high | 0.0249 | 0.9907 | -1.873 | 0.6295 | 0.8440 | nan |
| lca | 2 | substitution-mid | 0.0243 | 0.9455 | -1.395 | 0.6479 | 0.8611 | nan |
| lca | 3 | clean | 0.0214 | 0.8459 | +0.689 | 0.7238 | 0.8907 | nan |
| lca | 3 | dropout-high | 0.0248 | 0.9338 | -1.901 | 0.6271 | 0.8547 | nan |
| lca | 3 | dropout-mid | 0.0231 | 0.8912 | -0.652 | 0.6755 | 0.8728 | nan |
| lca | 3 | substitution-high | 0.0225 | 0.9200 | -0.128 | 0.6951 | 0.8660 | nan |
| lca | 3 | substitution-mid | 0.0217 | 0.8700 | +0.430 | 0.7150 | 0.8826 | nan |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | +0.0001 | -0.0348 | -0.639 | +0.0134 | +nan |
| 1 | dropout-high | -0.0000 | -0.0383 | -0.600 | +0.0159 | +nan |
| 1 | dropout-mid | +0.0001 | -0.0354 | -0.641 | +0.0147 | +nan |
| 1 | substitution-high | +0.0003 | -0.0269 | -0.746 | +0.0098 | +nan |
| 1 | substitution-mid | +0.0002 | -0.0325 | -0.676 | +0.0124 | +nan |
| 2 | clean | +0.0000 | -0.0268 | -0.239 | +0.0096 | +nan |
| 2 | dropout-high | -0.0002 | -0.0290 | -0.207 | +0.0129 | +nan |
| 2 | dropout-mid | -0.0001 | -0.0283 | -0.215 | +0.0111 | +nan |
| 2 | substitution-high | +0.0002 | -0.0166 | -0.330 | +0.0057 | +nan |
| 2 | substitution-mid | +0.0001 | -0.0236 | -0.265 | +0.0084 | +nan |
| 3 | clean | -0.0004 | -0.0372 | +0.035 | +0.0096 | +nan |
| 3 | dropout-high | -0.0005 | -0.0404 | +0.066 | +0.0125 | +nan |
| 3 | dropout-mid | -0.0004 | -0.0376 | +0.057 | +0.0111 | +nan |
| 3 | substitution-high | -0.0002 | -0.0244 | -0.053 | +0.0065 | +nan |
| 3 | substitution-mid | -0.0003 | -0.0333 | +0.007 | +0.0084 | +nan |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.