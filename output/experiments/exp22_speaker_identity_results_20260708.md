# Exp22 Speaker Identity Results - 2026-07-08

## Scope

This run evaluates whether SCIT-Speech preserves or exposes speaker identity information.

- Dataset: VCTK `data/VCTK/wav48_silence_trimmed`
- Main probe split: 110 speakers, 5 train + 5 test utterances per speaker
- Preservation split: 110 speakers, 3 enrollment + 5 test utterances per speaker
- Models: SCIT-Speech Base and SCIT-Speech LCA v2
- Layers: `L=1/2/3`
- Speaker backend for decoded-waveform preservation and demo: lightweight MFCC statistics + cosine profiles

## Main Result: Codes/Latent Speaker Probe

Run directory:
`output/experiments/exp22_speaker_probe_vctk110_20260708_seed42`

Chance top-1 for 110 speakers is about `0.009`.

| model | feature | L | speakers | test n | top1 | top5 | macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| base | codes_hist | 1 | 110 | 550 | 0.335 | 0.664 | 0.319 |
| base | codes_hist | 2 | 110 | 550 | 0.480 | 0.811 | 0.455 |
| base | codes_hist | 3 | 110 | 550 | 0.511 | 0.822 | 0.495 |
| base | latent_stats | 1 | 110 | 550 | 0.356 | 0.664 | 0.340 |
| base | latent_stats | 2 | 110 | 550 | 0.567 | 0.820 | 0.550 |
| base | latent_stats | 3 | 110 | 550 | 0.602 | 0.842 | 0.590 |
| lca | codes_hist | 1 | 110 | 550 | 0.338 | 0.644 | 0.322 |
| lca | codes_hist | 2 | 110 | 550 | 0.440 | 0.773 | 0.417 |
| lca | codes_hist | 3 | 110 | 550 | 0.529 | 0.809 | 0.498 |
| lca | latent_stats | 1 | 110 | 550 | 0.351 | 0.636 | 0.334 |
| lca | latent_stats | 2 | 110 | 550 | 0.544 | 0.800 | 0.529 |
| lca | latent_stats | 3 | 110 | 550 | 0.589 | 0.829 | 0.579 |

Interpretation: speaker identity is strongly recoverable from SCIT-Speech codes/features, especially as `L` increases. The strongest full-VCTK result is Base `latent_stats, L=3` with top1 `0.602`; LCA `latent_stats, L=3` is close at `0.589`.

## Decoded-Waveform Speaker Preservation

Run directory:
`output/experiments/exp22_speaker_identity_vctk110_20260708_seed42`

| model | L | n | top1 | verified | EER | TAR@FAR=0.01 |
|---|---:|---:|---:|---:|---:|---:|
| original | 0 | 550 | 0.278 | 1.000 | 0.669 | 0.038 |
| base | 1 | 550 | 0.053 | 1.000 | 0.833 | 0.000 |
| base | 2 | 550 | 0.124 | 1.000 | 0.771 | 0.000 |
| base | 3 | 550 | 0.162 | 1.000 | 0.736 | 0.005 |
| lca | 1 | 550 | 0.060 | 1.000 | 0.798 | 0.000 |
| lca | 2 | 550 | 0.118 | 1.000 | 0.744 | 0.002 |
| lca | 3 | 550 | 0.153 | 1.000 | 0.736 | 0.000 |

Interpretation: this MFCC backend is weak on 110-speaker VCTK even for original audio, so the decoded-waveform result should be treated as a conservative sanity check rather than biometric-strength evidence. It still shows the expected `L` trend: higher `L` preserves more speaker-identifying information.

## Three-User Demo Speaker-ID Smoke

Run directory:
`output/experiments/exp22_three_user_speaker_demo_smoke_20260708_seed42`

Setup:

- Router: local TCP room `exp22demo`, port `12450`
- Users: `p225`, `p226`, `p227`
- Each user sent a held-out VCTK utterance through the three-user SCIT-Speech route
- Speaker profiles: 3 enrollment utterances per user
- Runtime: about 20 seconds per client
- Result files: `demo_summary_clean.csv`, `logs/client_*.log`

| client | sent | recv | decoded | drops | speaker eval | correct | verified | speaker acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p225 | 35 | 70 | 70 | 0 | 32 | 14 | 32 | 0.438 |
| p226 | 35 | 70 | 70 | 0 | 32 | 17 | 32 | 0.531 |
| p227 | 35 | 70 | 70 | 0 | 32 | 19 | 32 | 0.594 |

Interpretation: the three-user demo speaker-ID path is live and produces `spk`, `score`, `margin`, `verified`, and summary CSV fields during actual routed communication. Accuracy is limited by the lightweight MFCC backend and tiny enrollment set, but the integration itself is validated.

## Caveat

For paper-grade speaker identity claims, rerun Exp22 with a frozen speaker-recognition model such as ECAPA-TDNN or x-vector. The current MFCC backend is dependency-light and useful for reproducible smoke/pilot checks, but not a state-of-the-art speaker verifier.
