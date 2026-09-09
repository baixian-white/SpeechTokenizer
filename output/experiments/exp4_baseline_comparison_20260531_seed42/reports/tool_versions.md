# Tool & library versions for exp4

- **ffmpeg**: `ffmpeg version 6.1.2 Copyright (c) 2000-2024 the FFmpeg developers`
- **python**: `3.10.19`
- **torch**: `2.9.1+cu128`
- **torchaudio**: `2.9.1+cu128`
- **soundfile**: `0.13.1`
- **whisper**: `20250625`
- **dac**: `1.0.0`
- **encodec**: `0.1.x (Meta facebook/encodec)`
- **jiwer**: `4.0.0`
- **platform**: `Windows-10-10.0.22631-SP0`
- **cuda_available**: `True`
- **cuda_device**: `NVIDIA GeForce RTX 5070 Ti`

## Codec capability check
- ffmpeg `libopus` encoder: AVAILABLE
- ffmpeg `amrwb` encoder: NOT AVAILABLE (only decoder; conda-forge build lacks `--enable-libvo-amrwbenc`)
- ffmpeg `amrnb` encoder: NOT AVAILABLE
- Whisper model used: `base.en` (English-only, ~74M params)
- DAC model: 16 kHz weights (n_codebooks=12, codebook_size=1024)
- EnCodec model: 24 kHz (target bandwidths: 1.5, 3.0, 6.0, 12.0, 24.0 kbps)