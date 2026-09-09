# Exp22 Speaker Identity Preservation

- speakers: 2
- enroll_per_speaker: 1
- test_per_speaker: 1
- backend: MFCC statistics (`n_mfcc=40`)

| model | L | n | top1 | verified | correct cos | impostor cos | margin | EER | TAR@FAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | 2 | 1.000 | 1.000 | 0.982 | 0.976 | 0.006 | 0.000 | 1.000 |
| original | 0 | 2 | 1.000 | 1.000 | 0.990 | 0.981 | 0.009 | 0.000 | 1.000 |

Note: the default backend is a lightweight MFCC speaker-feature baseline. Use a frozen speaker-recognition model before making biometric-strength claims.
