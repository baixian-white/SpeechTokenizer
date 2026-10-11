# Exp23 Speaker Classifier Bundle

Exp23 bundle mode is an optional replacement for the existing profile matcher. It does not alter routing, buffering, encryption, decoding, or playback.

```powershell
python group_client.py --speaker_id_enable --speaker_classifier_bundle path/to/exp23_bundle --speaker_classifier_device cpu --speaker_classifier_threshold 0.65
```

Do not pass `--speaker_profile_dir` together with `--speaker_classifier_bundle`.

The bundle must contain:

- `classifier.pt`
- `speaker_labels.json`
- `preprocessing.json`
- `calibration.json`
- `model_config.json`
- `manifest.json`

The formal Exp23 classifier uses `input_mode=codes_waveform`. The scripted model receives the buffered L3 RVQ indices and the matching 16 kHz decoded waveform. The receiver already has both values before playback resampling, so the frozen SpeechTokenizer is not duplicated inside the bundle.

Default deployment preprocessing is a 3-second window (`48000` samples), three RVQ layers, and a nominal `320`-sample token-frame hop. The streaming helper buffers audio and RVQ frames together and passes post-channel-perturbation codes that match the decoded waveform.

`verified` is only a display threshold over calibrated closed-set confidence. It is not unknown-speaker rejection or biometric authentication.

Exp23 is a fixed-110 closed-set classifier. It recognizes only identities present during training. It does not support arbitrary-user enrollment, unknown-speaker rejection, biometric authentication, or spoof detection.
