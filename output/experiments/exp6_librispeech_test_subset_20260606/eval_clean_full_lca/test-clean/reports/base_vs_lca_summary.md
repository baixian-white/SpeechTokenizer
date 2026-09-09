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
| base | 2 | clean | 0.0236 | 0.9898 | -2.419 | 0.6105 | 0.8619 | 1.6741 |
| base | 3 | clean | 0.0218 | 0.9209 | -0.987 | 0.6627 | 0.8835 | 1.9024 |
| lca | 1 | clean | 0.0298 | 1.2195 | -8.064 | 0.3957 | 0.8110 | 1.3051 |
| lca | 2 | clean | 0.0237 | 0.9576 | -2.677 | 0.5993 | 0.8687 | 1.6551 |
| lca | 3 | clean | 0.0215 | 0.8824 | -0.864 | 0.6663 | 0.8903 | 1.8819 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0002 | -0.0239 | -0.326 | +0.0128 | +0.0186 |
| 2 | clean | +0.0001 | -0.0322 | -0.258 | +0.0068 | -0.0190 |
| 3 | clean | -0.0003 | -0.0384 | +0.123 | +0.0067 | -0.0204 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.