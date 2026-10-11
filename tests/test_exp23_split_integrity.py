import ast
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from speechtokenizer.speaker_identity import data as speaker_data
from speechtokenizer.speaker_identity.data import (
    SpeakerSample,
    build_canonical_split,
    discover_audio_groups,
    normalize_utterance_group,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "data" / "VCTK" / "wav48_silence_trimmed"
DATA_MODULE_PATH = PROJECT_ROOT / "speechtokenizer" / "speaker_identity" / "data.py"
REAL_VCTK_AUDIO_AVAILABLE = (
    AUDIO_ROOT.is_dir()
    and (AUDIO_ROOT / "s5" / "s5_001_mic1.flac").is_file()
)


def write_silent_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(b"\x00\x00" * 160)


def formal_samples(audio_root, speaker_count=110, include_s5=True, create_audio=False):
    speaker_ids = [f"p{index:03d}" for index in range(speaker_count)]
    if include_s5:
        speaker_ids[-1] = "s5"
    samples = []
    for speaker in speaker_ids:
        audio_path = Path(audio_root) / f"{speaker}_000.wav"
        if create_audio:
            write_silent_wav(audio_path)
        samples.append(SpeakerSample(speaker, f"{speaker}_000", audio_path, None, None))
    return samples


class Exp23CanonicalSplitTest(unittest.TestCase):
    def test_data_module_uses_python38_compatible_annotations(self):
        source = DATA_MODULE_PATH.read_text(encoding="utf-8")
        ast.parse(source, filename=str(DATA_MODULE_PATH), feature_version=(3, 8))
        self.assertTrue(source.startswith("from __future__ import annotations"))
        self.assertNotIn(" | None", source)

    def test_normalize_utterance_group_merges_microphone_copies(self):
        self.assertEqual(normalize_utterance_group(Path("p225_001_mic1.flac")), "p225_001")
        self.assertEqual(normalize_utterance_group(Path("p225_001_mic2.flac")), "p225_001")

    def test_discovery_includes_non_p_speaker_and_selects_microphone(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_root = Path(temp_dir)
            for speaker_id in ("p225", "s5"):
                write_silent_wav(audio_root / speaker_id / f"{speaker_id}_001_mic1.wav")
                write_silent_wav(audio_root / speaker_id / f"{speaker_id}_001_mic2.wav")
            samples = discover_audio_groups(audio_root, microphone="mic1")
        self.assertEqual([sample.speaker_id for sample in samples], ["p225", "s5"])
        self.assertTrue(all(sample.audio_path.name.endswith("_mic1.wav") for sample in samples))

    def test_discovery_without_microphone_deduplicates_copies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_root = Path(temp_dir)
            write_silent_wav(audio_root / "p225" / "p225_001_mic1.wav")
            write_silent_wav(audio_root / "p225" / "p225_001_mic2.wav")
            samples = discover_audio_groups(audio_root, microphone=None)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].utterance_group, "p225_001")

    def test_canonical_split_is_deterministic_balanced_and_serializable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_root = Path(temp_dir)
            for speaker_id in ("p225", "s5"):
                for index in range(6):
                    for microphone in ("mic1", "mic2"):
                        write_silent_wav(audio_root / speaker_id / f"{speaker_id}_{index:03d}_{microphone}.wav")
            samples = discover_audio_groups(audio_root, microphone=None)
            first = speaker_data._build_canonical_split_for_samples(
                samples, 2, 1, 1, seed=42, expected_speaker_count=2,
                required_speakers={"s5"},
            )
            second = speaker_data._build_canonical_split_for_samples(
                list(reversed(samples)), 2, 1, 1, seed=42,
                expected_speaker_count=2, required_speakers={"s5"},
            )
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], 42)
        self.assertEqual(first["speakers"], ["p225", "s5"])
        self.assertEqual(first["counts"]["per_speaker"], {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(first["counts"]["total"], {"train": 4, "validation": 2, "test": 2})
        self.assertEqual(first["protocol"]["audio_validation"], "container_signature")
        json.dumps(first)
        groups = {name: {item["utterance_group"] for item in items} for name, items in first["splits"].items()}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        paths = [item["audio_path"] for items in first["splits"].values() for item in items]
        self.assertEqual(len(paths), len(set(paths)))

    def test_canonical_split_fails_on_shortage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "p225" / "p225_001_mic1.wav"
            write_silent_wav(audio_path)
            samples = [SpeakerSample("p225", "p225_001", audio_path, None, None)]
            with self.assertRaisesRegex(ValueError, "p225.*shortage"):
                speaker_data._build_canonical_split_for_samples(
                    samples, 1, 1, 1,
                    seed=42,
                    expected_speaker_count=1, required_speakers=set(),
                )

    def test_canonical_split_fails_on_corrupt_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "p225" / "p225_001_mic1.flac"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"not audio")
            samples = [SpeakerSample("p225", "p225_001", audio_path, None, None)]
            with self.assertRaisesRegex(ValueError, "invalid FLAC signature"):
                speaker_data._build_canonical_split_for_samples(
                    samples, 1, 0, 0,
                    seed=42,
                    expected_speaker_count=1, required_speakers=set(),
                )

    @patch("speechtokenizer.speaker_identity.data.discover_audio_groups")
    def test_canonical_root_fails_on_corrupt_audio(self, mock_discover):
        with tempfile.TemporaryDirectory() as temp_dir:
            samples = formal_samples(temp_dir, create_audio=True)
            samples[0].audio_path.write_bytes(b"not audio")
            mock_discover.return_value = samples
            with self.assertRaisesRegex(ValueError, "invalid WAV signature"):
                build_canonical_split(Path(temp_dir), 1, 0, 0)

    @patch("speechtokenizer.speaker_identity.data.discover_audio_groups")
    def test_canonical_root_fails_if_discovered_path_goes_missing(self, mock_discover):
        with tempfile.TemporaryDirectory() as temp_dir:
            samples = formal_samples(temp_dir, create_audio=True)
            samples[0].audio_path.unlink()
            mock_discover.return_value = samples
            with self.assertRaisesRegex(ValueError, "missing audio"):
                build_canonical_split(Path(temp_dir), 1, 0, 0)

    def test_public_canonical_requires_root_path(self):
        with self.assertRaises(TypeError):
            build_canonical_split([])

    def test_public_canonical_rejects_speaker_count_bypass_parameter(self):
        with self.assertRaises(TypeError):
            build_canonical_split([], expected_speaker_count=1)

    def test_public_canonical_rejects_required_speaker_bypass_parameter(self):
        with self.assertRaises(TypeError):
            build_canonical_split([], required_speakers=set())

    def test_formal_canonical_rejects_109_speakers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            samples = formal_samples(temp_dir, speaker_count=109)
            with self.assertRaisesRegex(ValueError, "110 speakers"):
                with patch(
                    "speechtokenizer.speaker_identity.data.discover_audio_groups",
                    return_value=samples,
                ):
                    build_canonical_split(Path(temp_dir), 1, 0, 0)

    def test_formal_canonical_requires_s5(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            samples = formal_samples(temp_dir, include_s5=False)
            with self.assertRaisesRegex(ValueError, "s5"):
                with patch(
                    "speechtokenizer.speaker_identity.data.discover_audio_groups",
                    return_value=samples,
                ):
                    build_canonical_split(Path(temp_dir), 1, 0, 0)

    def test_canonical_counts_reject_invalid_values(self):
        invalid_counts = [(-1, 0, 1), (True, 0, 1), (1.5, 0, 1), (0, 0, 0)]
        for counts in invalid_counts:
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, "count"):
                speaker_data._build_canonical_split_for_samples(
                    [], *counts, seed=42,
                    expected_speaker_count=0, required_speakers=set(),
                )

    def test_canonical_rejects_duplicate_normalized_audio_paths(self):
        shared_path = Path("duplicate.wav")
        samples = [
            SpeakerSample("p225", "p225_001", shared_path, None, None),
            SpeakerSample("s5", "s5_001", shared_path, None, None),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate audio path"):
            speaker_data._build_canonical_split_for_samples(
                samples, 1, 0, 0, seed=42,
                expected_speaker_count=2, required_speakers={"s5"},
            )

    def test_audio_errors_distinguish_extension_and_permission(self):
        unsupported = SpeakerSample("p225", "p225_001", Path("audio.mp3"), None, None)
        with self.assertRaisesRegex(ValueError, "unsupported audio extension"):
            speaker_data._build_canonical_split_for_samples(
                [unsupported], 1, 0, 0, seed=42,
                expected_speaker_count=1, required_speakers=set(),
            )
        unreadable = SpeakerSample("p225", "p225_001", Path("audio.wav"), None, None)
        with patch("pathlib.Path.open", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(ValueError, "audio read/permission error"):
                speaker_data._build_canonical_split_for_samples(
                    [unreadable], 1, 0, 0, seed=42,
                    expected_speaker_count=1, required_speakers=set(),
                )

    @unittest.skipUnless(REAL_VCTK_AUDIO_AVAILABLE, "real VCTK audio is unavailable")
    def test_real_canonical_dry_run_has_frozen_counts(self):
        result = build_canonical_split(AUDIO_ROOT)
        self.assertEqual(result["protocol"]["speaker_count"], 110)
        self.assertEqual(len(result["speakers"]), 110)
        self.assertIn("s5", result["speakers"])
        self.assertEqual(result["counts"]["total"], {"train": 8800, "validation": 1650, "test": 1650})
        self.assertGreaterEqual(result["counts"]["minimum_available_groups"], 123)


if __name__ == "__main__":
    unittest.main()
