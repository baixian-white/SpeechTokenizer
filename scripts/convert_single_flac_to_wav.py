#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import soundfile as sf
except ModuleNotFoundError:
    sf = None
    import torchaudio


def convert_flac_to_wav(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if sf is not None:
        audio, sample_rate = sf.read(str(input_path))
        sf.write(str(output_path), audio, sample_rate, format="WAV")
        return

    waveform, sample_rate = torchaudio.load(str(input_path))
    torchaudio.save(str(output_path), waveform, sample_rate)


def main() -> None:
    default_input = Path(
        "SpeechTokenizer/data/SpeechPretrain/LibriSpeech/train-clean-100/19/198/19-198-0003.flac"
    )
    parser = argparse.ArgumentParser(description="Convert one FLAC file to WAV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input FLAC path (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output WAV path (default: same path/name with .wav suffix)",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output if args.output else input_path.with_suffix(".wav")

    convert_flac_to_wav(input_path, output_path)
    print(f"Converted: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
