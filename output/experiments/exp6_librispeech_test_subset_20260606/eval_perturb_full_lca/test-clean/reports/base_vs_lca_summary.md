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
| base | 1 | clean | 0.0300 | 1.2434 | -7.738 | 0.4103 | 0.7982 | 1.2865 |
| base | 1 | dropout-high | 0.0314 | 1.3028 | -9.432 | 0.3508 | 0.7690 | 1.2056 |
| base | 1 | dropout-mid | 0.0306 | 1.2734 | -8.523 | 0.3827 | 0.7827 | 1.2396 |
| base | 1 | substitution-high | 0.0303 | 1.2951 | -8.154 | 0.3972 | 0.7810 | 1.2445 |
| base | 1 | substitution-mid | 0.0301 | 1.2586 | -7.895 | 0.4060 | 0.7921 | 1.2718 |
| base | 2 | clean | 0.0236 | 0.9898 | -2.419 | 0.6105 | 0.8619 | 1.6741 |
| base | 2 | dropout-high | 0.0266 | 1.0723 | -4.855 | 0.5168 | 0.8251 | 1.3908 |
| base | 2 | dropout-mid | 0.0251 | 1.0341 | -3.619 | 0.5642 | 0.8424 | 1.4948 |
| base | 2 | substitution-high | 0.0244 | 1.0469 | -3.114 | 0.5845 | 0.8380 | 1.4962 |
| base | 2 | substitution-mid | 0.0239 | 1.0097 | -2.595 | 0.6037 | 0.8546 | 1.6062 |
| base | 3 | clean | 0.0218 | 0.9209 | -0.987 | 0.6627 | 0.8835 | 1.9024 |
| base | 3 | dropout-high | 0.0251 | 1.0139 | -3.540 | 0.5665 | 0.8434 | 1.4935 |
| base | 3 | dropout-mid | 0.0236 | 0.9695 | -2.355 | 0.6126 | 0.8625 | 1.6468 |
| base | 3 | substitution-high | 0.0228 | 0.9809 | -1.734 | 0.6367 | 0.8621 | 1.6580 |
| base | 3 | substitution-mid | 0.0221 | 0.9409 | -1.165 | 0.6554 | 0.8765 | 1.8160 |
| lca | 1 | clean | 0.0298 | 1.2195 | -8.064 | 0.3957 | 0.8110 | 1.3051 |
| lca | 1 | dropout-high | 0.0310 | 1.2698 | -9.731 | 0.3426 | 0.7870 | 1.2288 |
| lca | 1 | dropout-mid | 0.0303 | 1.2443 | -8.824 | 0.3701 | 0.7980 | 1.2591 |
| lca | 1 | substitution-high | 0.0301 | 1.2596 | -8.471 | 0.3802 | 0.7946 | 1.2677 |
| lca | 1 | substitution-mid | 0.0299 | 1.2308 | -8.193 | 0.3910 | 0.8052 | 1.2913 |
| lca | 2 | clean | 0.0237 | 0.9576 | -2.677 | 0.5993 | 0.8687 | 1.6551 |
| lca | 2 | dropout-high | 0.0262 | 1.0305 | -4.910 | 0.5132 | 0.8373 | 1.4058 |
| lca | 2 | dropout-mid | 0.0250 | 0.9959 | -3.732 | 0.5583 | 0.8517 | 1.5004 |
| lca | 2 | substitution-high | 0.0245 | 1.0102 | -3.373 | 0.5731 | 0.8461 | 1.4900 |
| lca | 2 | substitution-mid | 0.0240 | 0.9755 | -2.865 | 0.5918 | 0.8620 | 1.5962 |
| lca | 3 | clean | 0.0215 | 0.8824 | -0.864 | 0.6663 | 0.8903 | 1.8819 |
| lca | 3 | dropout-high | 0.0244 | 0.9653 | -3.400 | 0.5705 | 0.8551 | 1.5020 |
| lca | 3 | dropout-mid | 0.0230 | 0.9252 | -2.106 | 0.6197 | 0.8725 | 1.6517 |
| lca | 3 | substitution-high | 0.0223 | 0.9345 | -1.625 | 0.6391 | 0.8712 | 1.6718 |
| lca | 3 | substitution-mid | 0.0217 | 0.9001 | -1.082 | 0.6580 | 0.8836 | 1.8019 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0002 | -0.0239 | -0.326 | +0.0128 | +0.0186 |
| 1 | dropout-high | -0.0004 | -0.0331 | -0.299 | +0.0180 | +0.0233 |
| 1 | dropout-mid | -0.0003 | -0.0290 | -0.301 | +0.0153 | +0.0195 |
| 1 | substitution-high | -0.0002 | -0.0355 | -0.317 | +0.0136 | +0.0232 |
| 1 | substitution-mid | -0.0002 | -0.0278 | -0.297 | +0.0131 | +0.0195 |
| 2 | clean | +0.0001 | -0.0322 | -0.258 | +0.0068 | -0.0190 |
| 2 | dropout-high | -0.0003 | -0.0418 | -0.055 | +0.0122 | +0.0149 |
| 2 | dropout-mid | -0.0001 | -0.0382 | -0.113 | +0.0093 | +0.0056 |
| 2 | substitution-high | +0.0001 | -0.0367 | -0.259 | +0.0081 | -0.0061 |
| 2 | substitution-mid | +0.0001 | -0.0342 | -0.270 | +0.0074 | -0.0099 |
| 3 | clean | -0.0003 | -0.0384 | +0.123 | +0.0067 | -0.0204 |
| 3 | dropout-high | -0.0007 | -0.0486 | +0.141 | +0.0117 | +0.0085 |
| 3 | dropout-mid | -0.0006 | -0.0443 | +0.249 | +0.0100 | +0.0049 |
| 3 | substitution-high | -0.0004 | -0.0465 | +0.109 | +0.0091 | +0.0139 |
| 3 | substitution-mid | -0.0004 | -0.0408 | +0.083 | +0.0071 | -0.0141 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.