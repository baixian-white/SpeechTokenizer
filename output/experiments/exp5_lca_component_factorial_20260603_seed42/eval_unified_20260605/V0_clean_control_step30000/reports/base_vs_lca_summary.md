# Base vs LCA evaluation summary

- Base ckpt: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output\experiments\exp5_lca_component_factorial_20260603_seed42\runs\V0_full_depth_clean_control\checkpoints\SpeechTokenizerTrainer_00030000`
  - sha256: `44f1a67fd123eff622e88da26eed9afe9ba54718c693b59c1af0675fe452c708`
- Sample count per (L, channel): 8
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0321 | 1.2022 | -7.183 | 0.4327 | 0.7866 | 1.3330 |
| base | 1 | dropout-high | 0.0339 | 1.2559 | -9.221 | 0.3590 | 0.7589 | 1.2160 |
| base | 1 | dropout-mid | 0.0330 | 1.2301 | -8.101 | 0.3990 | 0.7725 | 1.2525 |
| base | 1 | substitution-high | 0.0323 | 1.2403 | -7.336 | 0.4266 | 0.7689 | 1.2661 |
| base | 1 | substitution-mid | 0.0323 | 1.2227 | -7.359 | 0.4268 | 0.7722 | 1.3004 |
| base | 2 | clean | 0.0246 | 0.9549 | -1.447 | 0.6514 | 0.8442 | 1.7072 |
| base | 2 | dropout-high | 0.0277 | 1.0349 | -3.749 | 0.5681 | 0.8035 | 1.4089 |
| base | 2 | dropout-mid | 0.0267 | 0.9994 | -3.258 | 0.5847 | 0.8211 | 1.4824 |
| base | 2 | substitution-high | 0.0257 | 1.0107 | -2.156 | 0.6268 | 0.8229 | 1.5046 |
| base | 2 | substitution-mid | 0.0248 | 0.9725 | -1.504 | 0.6492 | 0.8366 | 1.6541 |
| base | 3 | clean | 0.0222 | 0.8916 | +0.354 | 0.7089 | 0.8697 | 1.9123 |
| base | 3 | dropout-high | 0.0263 | 0.9822 | -2.455 | 0.6138 | 0.8253 | 1.5082 |
| base | 3 | dropout-mid | 0.0236 | 0.9304 | -0.723 | 0.6762 | 0.8528 | 1.7229 |
| base | 3 | substitution-high | 0.0232 | 0.9508 | -0.235 | 0.6906 | 0.8496 | 1.6774 |
| base | 3 | substitution-mid | 0.0227 | 0.9118 | -0.012 | 0.6986 | 0.8605 | 1.8322 |
| lca | 1 | clean | 0.0313 | 1.1429 | -6.634 | 0.4432 | 0.8038 | 1.3739 |
| lca | 1 | dropout-high | 0.0332 | 1.1979 | -8.587 | 0.3721 | 0.7743 | 1.2407 |
| lca | 1 | dropout-mid | 0.0321 | 1.1688 | -7.335 | 0.4168 | 0.7914 | 1.2802 |
| lca | 1 | substitution-high | 0.0316 | 1.1852 | -6.883 | 0.4337 | 0.7859 | 1.2867 |
| lca | 1 | substitution-mid | 0.0316 | 1.1630 | -6.823 | 0.4365 | 0.7882 | 1.3444 |
| lca | 2 | clean | 0.0237 | 0.8781 | -0.917 | 0.6656 | 0.8667 | 1.8092 |
| lca | 2 | dropout-high | 0.0271 | 0.9630 | -3.252 | 0.5829 | 0.8282 | 1.4573 |
| lca | 2 | dropout-mid | 0.0255 | 0.9219 | -2.487 | 0.6117 | 0.8433 | 1.5689 |
| lca | 2 | substitution-high | 0.0249 | 0.9325 | -1.621 | 0.6410 | 0.8450 | 1.5865 |
| lca | 2 | substitution-mid | 0.0240 | 0.8966 | -1.022 | 0.6618 | 0.8595 | 1.7510 |
| lca | 3 | clean | 0.0211 | 0.8138 | +0.988 | 0.7280 | 0.8870 | 2.0392 |
| lca | 3 | dropout-high | 0.0253 | 0.9026 | -2.040 | 0.6295 | 0.8478 | 1.5911 |
| lca | 3 | dropout-mid | 0.0225 | 0.8554 | +0.070 | 0.6988 | 0.8704 | 1.8017 |
| lca | 3 | substitution-high | 0.0222 | 0.8776 | +0.361 | 0.7094 | 0.8661 | 1.7643 |
| lca | 3 | substitution-mid | 0.0215 | 0.8329 | +0.684 | 0.7199 | 0.8781 | 1.9293 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0008 | -0.0593 | +0.549 | +0.0171 | +0.0408 |
| 1 | dropout-high | -0.0007 | -0.0581 | +0.634 | +0.0154 | +0.0247 |
| 1 | dropout-mid | -0.0009 | -0.0613 | +0.766 | +0.0188 | +0.0277 |
| 1 | substitution-high | -0.0007 | -0.0550 | +0.453 | +0.0170 | +0.0206 |
| 1 | substitution-mid | -0.0007 | -0.0596 | +0.536 | +0.0159 | +0.0441 |
| 2 | clean | -0.0009 | -0.0768 | +0.529 | +0.0226 | +0.1019 |
| 2 | dropout-high | -0.0007 | -0.0719 | +0.496 | +0.0248 | +0.0485 |
| 2 | dropout-mid | -0.0011 | -0.0775 | +0.772 | +0.0222 | +0.0865 |
| 2 | substitution-high | -0.0008 | -0.0782 | +0.535 | +0.0221 | +0.0819 |
| 2 | substitution-mid | -0.0008 | -0.0759 | +0.482 | +0.0229 | +0.0968 |
| 3 | clean | -0.0011 | -0.0778 | +0.634 | +0.0173 | +0.1269 |
| 3 | dropout-high | -0.0010 | -0.0796 | +0.415 | +0.0225 | +0.0829 |
| 3 | dropout-mid | -0.0011 | -0.0750 | +0.793 | +0.0175 | +0.0788 |
| 3 | substitution-high | -0.0010 | -0.0732 | +0.596 | +0.0164 | +0.0869 |
| 3 | substitution-mid | -0.0012 | -0.0790 | +0.696 | +0.0176 | +0.0971 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.