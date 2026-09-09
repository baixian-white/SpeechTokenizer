# Exp22 Speaker Identity Preservation

- speakers: 110
- enroll_per_speaker: 3
- test_per_speaker: 5
- backend: SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`)

| model | L | n | top1 | verified | correct cos | impostor cos | margin | EER | TAR@FAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | 550 | 0.091 | 0.976 | 0.224 | 0.341 | -0.117 | 0.775 | 0.000 |
| base | 2 | 550 | 0.407 | 0.996 | 0.361 | 0.385 | -0.024 | 0.540 | 0.036 |
| base | 3 | 550 | 0.551 | 1.000 | 0.414 | 0.400 | 0.014 | 0.453 | 0.089 |
| lca | 1 | 550 | 0.089 | 0.982 | 0.215 | 0.342 | -0.128 | 0.784 | 0.005 |
| lca | 2 | 550 | 0.255 | 1.000 | 0.317 | 0.384 | -0.067 | 0.642 | 0.011 |
| lca | 3 | 550 | 0.364 | 1.000 | 0.361 | 0.394 | -0.033 | 0.569 | 0.022 |
| original | 0 | 550 | 0.993 | 1.000 | 0.708 | 0.403 | 0.305 | 0.038 | 0.929 |

Note: ECAPA is the preferred paper-grade backend for speaker-preservation claims. MFCC runs are retained as dependency-light sanity checks.
