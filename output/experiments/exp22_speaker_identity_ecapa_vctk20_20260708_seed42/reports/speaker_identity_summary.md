# Exp22 Speaker Identity Preservation

- speakers: 20
- enroll_per_speaker: 3
- test_per_speaker: 5
- backend: SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`)

| model | L | n | top1 | verified | correct cos | impostor cos | margin | EER | TAR@FAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | 100 | 0.250 | 0.660 | 0.205 | 0.259 | -0.054 | 0.630 | 0.000 |
| base | 2 | 100 | 0.650 | 0.890 | 0.346 | 0.302 | 0.045 | 0.390 | 0.080 |
| base | 3 | 100 | 0.770 | 0.960 | 0.403 | 0.320 | 0.082 | 0.310 | 0.070 |
| lca | 1 | 100 | 0.230 | 0.820 | 0.210 | 0.296 | -0.086 | 0.690 | 0.000 |
| lca | 2 | 100 | 0.430 | 0.920 | 0.305 | 0.324 | -0.019 | 0.580 | 0.090 |
| lca | 3 | 100 | 0.580 | 0.900 | 0.348 | 0.330 | 0.018 | 0.470 | 0.110 |
| original | 0 | 100 | 1.000 | 1.000 | 0.708 | 0.310 | 0.398 | 0.050 | 0.940 |

Note: ECAPA is the preferred paper-grade backend for speaker-preservation claims. MFCC runs are retained as dependency-light sanity checks.
