# Exp22 Speaker Identity Preservation

- speakers: 110
- enroll_per_speaker: 3
- test_per_speaker: 5
- backend: SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`)

| model | L | n | top1 | verified | correct cos | impostor cos | margin | EER | TAR@FAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | 550 | 0.125 | 0.958 | 0.227 | 0.334 | -0.108 | 0.744 | 0.011 |
| base | 2 | 550 | 0.449 | 0.996 | 0.363 | 0.376 | -0.012 | 0.525 | 0.038 |
| base | 3 | 550 | 0.624 | 1.000 | 0.418 | 0.389 | 0.029 | 0.420 | 0.120 |
| lca | 1 | 550 | 0.075 | 0.985 | 0.211 | 0.346 | -0.135 | 0.809 | 0.004 |
| lca | 2 | 550 | 0.258 | 1.000 | 0.319 | 0.393 | -0.074 | 0.645 | 0.011 |
| lca | 3 | 550 | 0.349 | 1.000 | 0.364 | 0.407 | -0.043 | 0.584 | 0.047 |
| original | 0 | 550 | 1.000 | 1.000 | 0.713 | 0.402 | 0.311 | 0.029 | 0.945 |

Note: ECAPA is the preferred paper-grade backend for speaker-preservation claims. MFCC runs are retained as dependency-light sanity checks.
