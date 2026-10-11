# Exp23 Speaker Classifier Bundle

Exp23 bundle mode is an optional replacement for the existing profile matcher. It does not alter routing, buffering, encryption, decoding, or playback.

```powershell
python group_client.py --speaker_id_enable --speaker_classifier_bundle path/to/exp23_bundle --speaker_classifier_device cpu --speaker_classifier_threshold 0.65
```

Do not pass `--speaker_profile_dir` together with `--speaker_classifier_bundle`.

The interactive `group_client.py` entry point defaults to the
recognition-focused profile: a 1.5-second codec chunk and the fixed six-speaker
roster (`p225,p226,p227,p228,p229,p230`). The candidate mask keeps the same
110-class model and only masks non-roster labels at decision time:

```powershell
python group_client.py --speaker_id_enable --speaker_classifier_bundle path/to/exp23_bundle
```

To restore the open-set/110-class interactive condition, pass the explicit
sentinel and the original 0.5-second chunk size:

```powershell
python group_client.py --speaker_id_enable --speaker_classifier_bundle path/to/exp23_bundle --speaker_classifier_candidates all --chunk_seconds 0.5
```

An empty candidate value also disables the mask. The recognition-focused
defaults apply to this interactive entry point. The frozen Stage 1 replay
continues to use its fixed 0.5-second packet protocol, while
`run_metrics_experiment.py` keeps the original 110-class baseline; pass `all`
whenever a scripted caller must explicitly restore the open-set output.

The bundle must contain:

- `classifier.pt`
- `speaker_labels.json`
- `preprocessing.json`
- `calibration.json`
- `model_config.json`
- `manifest.json`

The formal Exp23 classifier uses `input_mode=codes_waveform`. The scripted model receives the buffered L3 RVQ indices and the matching 16 kHz decoded waveform. The receiver already has both values before playback resampling, so the frozen SpeechTokenizer is not duplicated inside the bundle.

`user_id` is the routing name and may be `A`, `B`, `C`, and so on. In the
six-client Stage 1 replay, the A-F to VCTK label mapping lives in the local
trial configuration. The prediction is compared with this local ground truth;
the speaker label is not sent in the RVQ packet header.

Default deployment preprocessing is a 3-second window (`48000` samples), three RVQ layers, and a nominal `320`-sample token-frame hop. The streaming helper buffers audio and RVQ frames together and passes post-channel-perturbation codes that match the decoded waveform.

`verified` is only a display threshold over calibrated closed-set confidence. It is not unknown-speaker rejection or biometric authentication.

Exp23 is a fixed-110 closed-set classifier. It recognizes only identities present during training. It does not support arbitrary-user enrollment, unknown-speaker rejection, biometric authentication, or spoof detection.
