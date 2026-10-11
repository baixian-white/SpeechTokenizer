import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import speechtokenizer.speaker_identity as speaker_identity
from speechtokenizer.speaker_identity import data as speaker_data
from speechtokenizer.speaker_identity.data import (
    SpeakerSample,
    build_text_controlled_split,
    normalize_transcript,
    transcript_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "data" / "VCTK" / "wav48_silence_trimmed"
TRANSCRIPT_ROOT = PROJECT_ROOT / "data" / "VCTK" / "txt"
REAL_VCTK_TEXT_AVAILABLE = (
    AUDIO_ROOT.is_dir()
    and TRANSCRIPT_ROOT.is_dir()
    and (AUDIO_ROOT / "s5" / "s5_001_mic1.flac").is_file()
    and (TRANSCRIPT_ROOT / "p225" / "p225_001.txt").is_file()
)


def sample(speaker_id, index, transcript):
    normalized = None if transcript is None else normalize_transcript(transcript)
    return SpeakerSample(
        speaker_id=speaker_id,
        utterance_group=f"{speaker_id}_{index:03d}",
        audio_path=Path(f"/{speaker_id}_{index:03d}_mic1.flac"),
        transcript=normalized,
        transcript_hash=None if normalized is None else transcript_sha256(normalized),
    )


class Exp23TextControlledSplitTest(unittest.TestCase):
    def test_transcript_normalization_and_hash_are_frozen(self):
        normalized = normalize_transcript("  HELLO\n\tWorld  ")
        self.assertEqual(normalized, "hello world")
        self.assertEqual(transcript_sha256(normalized), hashlib.sha256(b"hello world").hexdigest())

    def test_exact_transcript_resolution_never_infers_across_speakers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            transcript_root = Path(temp_dir)
            other_speaker = transcript_root / "p226"
            other_speaker.mkdir(parents=True)
            (other_speaker / "p226_001.txt").write_text("shared sentence", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "p225_001"):
                speaker_data._load_exact_transcripts([sample("p225", 1, None)], transcript_root)

    def test_transcript_components_reject_path_traversal(self):
        bad_components = [".", "..", "../escape", "part/name", "part\\name", "C:\\escape"]
        with tempfile.TemporaryDirectory() as temp_dir:
            for bad_component in bad_components:
                with self.subTest(speaker_id=bad_component), self.assertRaisesRegex(ValueError, "unsafe"):
                    unsafe = SpeakerSample(bad_component, "safe_group", Path("audio.wav"), None, None)
                    speaker_data._load_exact_transcripts([unsafe], Path(temp_dir))
                with self.subTest(utterance_group=bad_component), self.assertRaisesRegex(ValueError, "unsafe"):
                    unsafe = SpeakerSample("p225", bad_component, Path("audio.wav"), None, None)
                    speaker_data._load_exact_transcripts([unsafe], Path(temp_dir))

    def test_missing_transcript_for_nonexcluded_speaker_fails(self):
        with self.assertRaisesRegex(ValueError, "p316"):
            speaker_data._build_text_controlled_split_for_samples(
                [sample("p316", 1, None)], 1, 0, 0,
                seed=1, expected_speaker_count=1,
            )

    def test_private_text_builder_reports_frozen_exclusion_metadata(self):
        samples = [sample("p225", index, f"text {index}") for index in range(10)]
        result = speaker_data._build_text_controlled_split_for_samples(
            samples, 2, 1, 1, seed=1, expected_speaker_count=1,
        )
        self.assertEqual(result["excluded_speakers"], ["p315"])
        self.assertEqual(result["speakers"], ["p225"])
        self.assertEqual(result["protocol"]["speaker_count"], 109)
        self.assertEqual(result["protocol"]["canonical_speaker_count"], 110)

    def test_text_split_is_deterministic_balanced_and_hash_disjoint(self):
        samples = []
        for speaker_id in ("p225", "s5"):
            samples.extend(sample(speaker_id, index, f"GLOBAL TEXT {index}") for index in range(10))
        first = speaker_data._build_text_controlled_split_for_samples(
            samples, 2, 1, 1, seed=1, expected_speaker_count=2,
        )
        second = speaker_data._build_text_controlled_split_for_samples(
            list(reversed(samples)), 2, 1, 1, seed=1,
            expected_speaker_count=2,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], 1)
        self.assertEqual(first["counts"]["total"], {"train": 4, "validation": 2, "test": 2})
        self.assertEqual(first["protocol"]["audio_validation"], "container_signature")
        hashes = {name: set(values) for name, values in first["transcript_hashes"].items()}
        self.assertFalse(hashes["train"] & hashes["validation"])
        self.assertFalse(hashes["train"] & hashes["test"])
        self.assertFalse(hashes["validation"] & hashes["test"])
        json.dumps(first)

    def test_excluding_any_speaker_other_than_p315_is_rejected(self):
        with self.assertRaises(TypeError):
            build_text_controlled_split([], excluded_speakers={"p314"})

    def test_public_text_split_rejects_speaker_count_bypass_parameter(self):
        with self.assertRaises(TypeError):
            build_text_controlled_split([], expected_speaker_count=1)

    def test_public_text_split_requires_root_path(self):
        with self.assertRaises(TypeError):
            build_text_controlled_split([])

    def test_package_exports_only_stable_high_level_api(self):
        self.assertNotIn("discover_audio_groups", speaker_identity.__all__)
        self.assertNotIn("load_exact_transcripts", speaker_identity.__all__)

    def test_text_root_fails_on_corrupt_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "audio" / "p225" / "p225_001_mic1.flac"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"not audio")
            transcript_path = root / "txt" / "p225" / "p225_001.txt"
            transcript_path.parent.mkdir(parents=True)
            transcript_path.write_text("hello", encoding="utf-8")
            corrupt_samples = [SpeakerSample("p315", "p315_001", root / "p315.wav", None, None)]
            for index in range(109):
                path = root / f"p{index:03d}.wav"
                write_path = path
                write_path.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(write_path), "wb") as audio_file:
                    audio_file.setnchannels(1)
                    audio_file.setsampwidth(2)
                    audio_file.setframerate(16000)
                    audio_file.writeframes(b"\x00\x00" * 10)
                corrupt_samples.append(SpeakerSample(f"p{index:03d}", f"p{index:03d}_001", path, None, None))
            corrupt_samples[1].audio_path.write_bytes(b"not audio")
            with self.assertRaisesRegex(ValueError, "invalid WAV signature"):
                with patch(
                    "speechtokenizer.speaker_identity.data.discover_audio_groups",
                    return_value=corrupt_samples,
                ):
                    build_text_controlled_split(
                        root / "audio", 0, 0, 0, transcript_root=root / "txt",
                    )

    @patch("speechtokenizer.speaker_identity.data.discover_audio_groups")
    def test_text_root_fails_if_discovered_path_goes_missing(self, mock_discover):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mock_discover.return_value = [sample("p315", 1, None)]
            for index in range(109):
                path = root / f"p{index:03d}.wav"
                if index != 0:
                    with wave.open(str(path), "wb") as audio_file:
                        audio_file.setnchannels(1)
                        audio_file.setsampwidth(2)
                        audio_file.setframerate(16000)
                        audio_file.writeframes(b"\x00\x00" * 10)
                mock_discover.return_value.append(
                    SpeakerSample(f"p{index:03d}", f"p{index:03d}_001", path, None, None)
                )
            transcript_path = root / "txt" / "p225" / "p225_001.txt"
            transcript_path.parent.mkdir(parents=True)
            transcript_path.write_text("hello", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing audio"):
                build_text_controlled_split(
                    root / "audio", 0, 0, 0, transcript_root=root / "txt",
                )

    def test_formal_text_split_rejects_108_or_110_speakers(self):
        for output_speaker_count in (108, 110):
            with self.subTest(output_speaker_count=output_speaker_count):
                samples = [sample("p315", 1, None)]
                samples.extend(
                    sample(f"p{index:03d}", 1, f"text {index}")
                    for index in range(output_speaker_count)
                )
                with self.assertRaisesRegex(ValueError, "109 speakers"):
                    with patch(
                        "speechtokenizer.speaker_identity.data.discover_audio_groups",
                        return_value=samples,
                    ):
                        build_text_controlled_split(Path("audio_root"), 1, 0, 0, transcript_root=Path("txt"))

    def test_formal_text_split_requires_p315_before_filtering(self):
        samples = [sample(f"p{index:03d}", 1, f"text {index}") for index in range(109)]
        with self.assertRaisesRegex(ValueError, "p315"):
            with patch(
                "speechtokenizer.speaker_identity.data.discover_audio_groups",
                return_value=samples,
            ):
                build_text_controlled_split(Path("audio_root"), 1, 0, 0, transcript_root=Path("txt"))

    def test_text_counts_reject_invalid_values(self):
        invalid_counts = [(-1, 0, 1), (True, 0, 1), (1.5, 0, 1), (0, 0, 0)]
        for counts in invalid_counts:
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, "count"):
                speaker_data._build_text_controlled_split_for_samples(
                    [], *counts, seed=1, expected_speaker_count=0,
                )

    def test_text_builder_rejects_duplicate_normalized_audio_paths(self):
        shared_path = Path("duplicate.flac")
        samples = [
            SpeakerSample("p225", "p225_001", shared_path, "one", transcript_sha256("one")),
            SpeakerSample("p226", "p226_001", shared_path, "two", transcript_sha256("two")),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate audio path"):
            speaker_data._build_text_controlled_split_for_samples(
                samples, 1, 0, 0, seed=1, expected_speaker_count=2,
            )

    @unittest.skipUnless(REAL_VCTK_TEXT_AVAILABLE, "real VCTK audio/text is unavailable")
    def test_real_text_controlled_dry_run_has_frozen_counts(self):
        result = build_text_controlled_split(AUDIO_ROOT, transcript_root=TRANSCRIPT_ROOT)
        self.assertEqual(result["seed"], 1)
        self.assertEqual(result["protocol"]["speaker_count"], 109)
        self.assertEqual(len(result["speakers"]), 109)
        self.assertNotIn("p315", result["speakers"])
        self.assertEqual(result["excluded_speakers"], ["p315"])
        self.assertEqual(result["counts"]["total"], {"train": 8720, "validation": 1090, "test": 1090})
        minimum = result["counts"]["minimum_available_by_split"]
        self.assertGreaterEqual(minimum["train"], 96)
        self.assertGreaterEqual(minimum["validation"], 14)
        self.assertGreaterEqual(minimum["test"], 13)


if __name__ == "__main__":
    unittest.main()
