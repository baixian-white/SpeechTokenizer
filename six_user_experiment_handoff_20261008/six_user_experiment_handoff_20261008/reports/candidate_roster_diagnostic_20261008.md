# Known-Roster Recognition Diagnostic

Date: 2026-10-08

This is a development diagnostic, not a replacement for the paper's frozen
110-class result.

## Setup

- Six VCTK test speakers: `p225`, `p226`, `p227`, `p228`, `p229`, `p230`
- 90 independent three-second test utterances
- Same frozen LCA checkpoint and seed42 Exp23 bundle
- CUDA LCA encode/decode, CPU TorchScript classifier
- Demo-style independent chunk encode/decode followed by `codes_waveform`
  classification

## Results

| Decision/input condition | Correct | Total | Top-1 |
|---|---:|---:|---:|
| Original 110-class, 0.5 s chunks | 76 | 90 | 84.44% |
| Known six-roster candidates, 0.5 s chunks | 89 | 90 | 98.89% |
| Known six-roster candidates, 1.5 s chunks | 90 | 90 | 100.00% |
| Known six-roster candidates, 3.0 s chunks | 90 | 90 | 100.00% |

The one remaining 0.5-second candidate error was `p226_283`, predicted as
`p227`. The corresponding full-utterance source/audio path predicts `p226`;
the error is therefore sensitive to the short independent chunk codec path.

## Interpretation

Candidate masking is a deployment-time known-roster decision rule. It does
not change classifier weights, the 110-class bundle, or the paper baseline.
The 1.5-second setting is a recognition-focused demo profile and should be
reported separately from the original 0.5-second streaming baseline.

