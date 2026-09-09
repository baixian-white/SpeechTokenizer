# Exp22 Speaker Identity Preservation

- speakers: 110
- enroll_per_speaker: 3
- test_per_speaker: 5
- backend: SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`)

| model | L | n | top1 | verified | correct cos | impostor cos | margin | EER | TAR@FAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | 550 | 0.107 | 0.958 | 0.221 | 0.343 | -0.122 | 0.784 | 0.002 |
| base | 2 | 550 | 0.378 | 0.995 | 0.359 | 0.389 | -0.030 | 0.567 | 0.064 |
| base | 3 | 550 | 0.551 | 1.000 | 0.410 | 0.400 | 0.010 | 0.453 | 0.087 |
| lca | 1 | 550 | 0.062 | 0.995 | 0.213 | 0.371 | -0.158 | 0.836 | 0.002 |
| lca | 2 | 550 | 0.187 | 0.998 | 0.313 | 0.412 | -0.099 | 0.711 | 0.022 |
| lca | 3 | 550 | 0.322 | 0.998 | 0.359 | 0.422 | -0.063 | 0.644 | 0.029 |
| original | 0 | 550 | 0.995 | 1.000 | 0.712 | 0.400 | 0.312 | 0.035 | 0.940 |

Note: ECAPA is the preferred paper-grade backend for speaker-preservation claims. MFCC runs are retained as dependency-light sanity checks.
