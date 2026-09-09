# Exp22 Speaker Identity Preservation

- speakers: 2
- enroll_per_speaker: 2
- test_per_speaker: 1
- backend: SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`)

| model | L | n | top1 | verified | correct cos | impostor cos | margin | EER | TAR@FAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | 2 | 1.000 | 0.000 | 0.210 | 0.125 | 0.085 | 0.000 | 1.000 |
| original | 0 | 2 | 1.000 | 1.000 | 0.729 | 0.046 | 0.683 | 0.000 | 1.000 |

Note: ECAPA is the preferred paper-grade backend for speaker-preservation claims. MFCC runs are retained as dependency-light sanity checks.
