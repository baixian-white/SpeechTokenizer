# Base vs LCA evaluation summary

- Base ckpt: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42\checkpoints\SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`
  - sha256: `af60223b2f44733f5a43d5a022871dad061dfe2421473c96f5d350c3d4b49a5a`
- Sample count per (L, channel): 256
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0312 | 1.2080 | -6.181 | 0.4546 | 0.7971 | nan |
| base | 1 | dropout-high | 0.0329 | 1.2674 | -7.949 | 0.3872 | 0.7679 | nan |
| base | 1 | dropout-mid | 0.0321 | 1.2396 | -7.128 | 0.4180 | 0.7824 | nan |
| base | 1 | substitution-high | 0.0316 | 1.2536 | -6.547 | 0.4401 | 0.7793 | nan |
| base | 1 | substitution-mid | 0.0313 | 1.2248 | -6.306 | 0.4497 | 0.7914 | nan |
| base | 2 | clean | 0.0239 | 0.9491 | -0.921 | 0.6662 | 0.8596 | nan |
| base | 2 | dropout-high | 0.0270 | 1.0305 | -3.275 | 0.5731 | 0.8235 | nan |
| base | 2 | dropout-mid | 0.0255 | 0.9912 | -2.132 | 0.6188 | 0.8407 | nan |
| base | 2 | substitution-high | 0.0248 | 1.0091 | -1.522 | 0.6431 | 0.8389 | nan |
| base | 2 | substitution-mid | 0.0242 | 0.9684 | -1.138 | 0.6581 | 0.8529 | nan |
| base | 3 | clean | 0.0217 | 0.8831 | +0.654 | 0.7224 | 0.8811 | nan |
| base | 3 | dropout-high | 0.0254 | 0.9754 | -2.017 | 0.6226 | 0.8418 | nan |
| base | 3 | dropout-mid | 0.0236 | 0.9301 | -0.782 | 0.6704 | 0.8610 | nan |
| base | 3 | substitution-high | 0.0227 | 0.9467 | -0.044 | 0.6983 | 0.8592 | nan |
| base | 3 | substitution-mid | 0.0221 | 0.9026 | +0.424 | 0.7146 | 0.8741 | nan |
| lca | 1 | clean | 0.0312 | 1.1656 | -6.687 | 0.4358 | 0.8097 | nan |
| lca | 1 | dropout-high | 0.0326 | 1.2158 | -8.350 | 0.3735 | 0.7855 | nan |
| lca | 1 | dropout-mid | 0.0319 | 1.1922 | -7.556 | 0.4029 | 0.7973 | nan |
| lca | 1 | substitution-high | 0.0315 | 1.1999 | -7.101 | 0.4197 | 0.7928 | nan |
| lca | 1 | substitution-mid | 0.0313 | 1.1784 | -6.834 | 0.4302 | 0.8043 | nan |
| lca | 2 | clean | 0.0241 | 0.9217 | -1.332 | 0.6503 | 0.8674 | nan |
| lca | 2 | dropout-high | 0.0269 | 0.9920 | -3.546 | 0.5617 | 0.8367 | nan |
| lca | 2 | dropout-mid | 0.0255 | 0.9576 | -2.477 | 0.6048 | 0.8515 | nan |
| lca | 2 | substitution-high | 0.0249 | 0.9717 | -1.987 | 0.6247 | 0.8474 | nan |
| lca | 2 | substitution-mid | 0.0244 | 0.9379 | -1.549 | 0.6418 | 0.8610 | nan |
| lca | 3 | clean | 0.0215 | 0.8470 | +0.554 | 0.7194 | 0.8890 | nan |
| lca | 3 | dropout-high | 0.0248 | 0.9271 | -1.994 | 0.6234 | 0.8554 | nan |
| lca | 3 | dropout-mid | 0.0231 | 0.8877 | -0.749 | 0.6719 | 0.8720 | nan |
| lca | 3 | substitution-high | 0.0225 | 0.9023 | -0.198 | 0.6932 | 0.8687 | nan |
| lca | 3 | substitution-mid | 0.0218 | 0.8640 | +0.291 | 0.7103 | 0.8825 | nan |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0001 | -0.0424 | -0.507 | +0.0126 | +nan |
| 1 | dropout-high | -0.0003 | -0.0516 | -0.400 | +0.0176 | +nan |
| 1 | dropout-mid | -0.0002 | -0.0474 | -0.429 | +0.0149 | +nan |
| 1 | substitution-high | -0.0001 | -0.0537 | -0.554 | +0.0135 | +nan |
| 1 | substitution-mid | -0.0001 | -0.0464 | -0.528 | +0.0129 | +nan |
| 2 | clean | +0.0002 | -0.0274 | -0.410 | +0.0078 | +nan |
| 2 | dropout-high | -0.0002 | -0.0385 | -0.271 | +0.0132 | +nan |
| 2 | dropout-mid | +0.0000 | -0.0336 | -0.345 | +0.0108 | +nan |
| 2 | substitution-high | +0.0002 | -0.0374 | -0.465 | +0.0086 | +nan |
| 2 | substitution-mid | +0.0002 | -0.0305 | -0.410 | +0.0082 | +nan |
| 3 | clean | -0.0002 | -0.0361 | -0.100 | +0.0079 | +nan |
| 3 | dropout-high | -0.0006 | -0.0482 | +0.023 | +0.0136 | +nan |
| 3 | dropout-mid | -0.0005 | -0.0424 | +0.034 | +0.0110 | +nan |
| 3 | substitution-high | -0.0002 | -0.0444 | -0.154 | +0.0095 | +nan |
| 3 | substitution-mid | -0.0002 | -0.0386 | -0.133 | +0.0084 | +nan |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.