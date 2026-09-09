# Full Utterance Audio Quality

Objective proxy comparison between original full utterances and generated L1/L2/L3 reconstructions.
These metrics do not replace human listening, WER, PESQ, STOI, or channel evaluation.

| Layer | n | wave L1 mean | mel L1 mean | SI-SNR mean | corr mean | RMS ratio mean | clip ratio mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 4 | 0.036629 | 1.450744 | -14.578 | 0.2669 | -4.682 dB | 0.000000 |
| L2 | 4 | 0.029362 | 1.143400 | -7.507 | 0.5025 | -2.243 dB | 0.000000 |
| L3 | 4 | 0.026936 | 1.075713 | -4.922 | 0.5645 | -1.426 dB | 0.000000 |

- CSV: `output\experiments\exp2_scit_speech_training_20260527_232327_seed42\metrics\full_utterance_audio_quality.csv`
- JSON: `output\experiments\exp2_scit_speech_training_20260527_232327_seed42\metrics\full_utterance_audio_quality.json`
