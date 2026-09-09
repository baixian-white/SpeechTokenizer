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
| base | 1 | clean | 0.0311 | 1.3915 | -8.775 | 0.3877 | 0.7668 | 1.2596 |
| base | 2 | clean | 0.0246 | 1.1358 | -3.189 | 0.6042 | 0.8289 | 1.5775 |
| base | 3 | clean | 0.0227 | 1.0621 | -1.646 | 0.6614 | 0.8520 | 1.7745 |
| lca | 1 | clean | 0.0310 | 1.3908 | -9.126 | 0.3805 | 0.7804 | 1.2786 |
| lca | 2 | clean | 0.0248 | 1.1176 | -3.440 | 0.5870 | 0.8384 | 1.5636 |
| lca | 3 | clean | 0.0225 | 1.0348 | -1.509 | 0.6547 | 0.8602 | 1.7434 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0002 | -0.0008 | -0.351 | +0.0136 | +0.0190 |
| 2 | clean | +0.0002 | -0.0182 | -0.251 | +0.0095 | -0.0140 |
| 3 | clean | -0.0002 | -0.0273 | +0.138 | +0.0081 | -0.0311 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.