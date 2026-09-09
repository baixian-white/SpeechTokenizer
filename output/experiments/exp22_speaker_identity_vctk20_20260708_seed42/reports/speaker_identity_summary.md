# Exp22 Speaker Identity Preservation

- speakers: 20
- enroll_per_speaker: 3
- test_per_speaker: 5
- backend: MFCC statistics (`n_mfcc=40`)

| model | L | n | top1 | verified | correct cos | impostor cos | margin | EER | TAR@FAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | 100 | 0.120 | 1.000 | 0.987 | 0.991 | -0.004 | 0.700 | 0.010 |
| base | 2 | 100 | 0.350 | 1.000 | 0.992 | 0.995 | -0.002 | 0.580 | 0.010 |
| base | 3 | 100 | 0.380 | 1.000 | 0.993 | 0.995 | -0.002 | 0.610 | 0.030 |
| lca | 1 | 100 | 0.110 | 1.000 | 0.988 | 0.992 | -0.004 | 0.680 | 0.020 |
| lca | 2 | 100 | 0.300 | 1.000 | 0.992 | 0.994 | -0.002 | 0.610 | 0.020 |
| lca | 3 | 100 | 0.370 | 1.000 | 0.993 | 0.994 | -0.002 | 0.570 | 0.020 |
| original | 0 | 100 | 0.480 | 1.000 | 0.994 | 0.995 | -0.001 | 0.550 | 0.050 |

Note: the default backend is a lightweight MFCC speaker-feature baseline. Use a frozen speaker-recognition model before making biometric-strength claims.
