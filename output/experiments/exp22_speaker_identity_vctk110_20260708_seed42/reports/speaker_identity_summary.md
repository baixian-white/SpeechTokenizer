# Exp22 Speaker Identity Preservation

- speakers: 110
- enroll_per_speaker: 3
- test_per_speaker: 5
- backend: MFCC statistics (`n_mfcc=40`)

| model | L | n | top1 | verified | correct cos | impostor cos | margin | EER | TAR@FAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | 550 | 0.053 | 1.000 | 0.985 | 0.993 | -0.008 | 0.833 | 0.000 |
| base | 2 | 550 | 0.124 | 1.000 | 0.992 | 0.996 | -0.005 | 0.771 | 0.000 |
| base | 3 | 550 | 0.162 | 1.000 | 0.992 | 0.996 | -0.004 | 0.736 | 0.005 |
| lca | 1 | 550 | 0.060 | 1.000 | 0.986 | 0.994 | -0.008 | 0.798 | 0.000 |
| lca | 2 | 550 | 0.118 | 1.000 | 0.991 | 0.996 | -0.005 | 0.744 | 0.002 |
| lca | 3 | 550 | 0.153 | 1.000 | 0.991 | 0.996 | -0.005 | 0.736 | 0.000 |
| original | 0 | 550 | 0.278 | 1.000 | 0.993 | 0.996 | -0.003 | 0.669 | 0.038 |

Note: the default backend is a lightweight MFCC speaker-feature baseline. Use a frozen speaker-recognition model before making biometric-strength claims.
