import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch


def make_record(root, **overrides):
    from speechtokenizer.speaker_identity.cache import CacheRecord

    root = Path(root)
    source = root / "source.flac"
    reconstruction = root / "reconstruction.flac"
    codes = root / "codes.npz"
    audio = np.linspace(-0.1, 0.1, 1600, dtype=np.float32)
    sf.write(source, audio, 16000)
    sf.write(reconstruction, audio, 16000)
    np.savez_compressed(codes, codes=np.arange(15, dtype=np.int64).reshape(3, 5))
    values = {
        "speaker_id": "p225",
        "label_index": 0,
        "utterance_group": "p225_001",
        "split": "train",
        "source_audio_path": source,
        "model_name": "lca",
        "rvq_layers": 3,
        "codes_path": codes,
        "reconstruction_path": reconstruction,
        "original_sample_count": 1600,
        "code_frame_count": 5,
        "valid_sample_count": 1600,
        "transcript_hash": "transcript",
        "checkpoint_sha256": "checkpoint",
        "config_sha256": "config",
    }
    values.update(overrides)
    return CacheRecord(**values)


class CacheRecordTest(unittest.TestCase):
    def test_json_roundtrip_preserves_frozen_record_and_string_paths(self):
        from speechtokenizer.speaker_identity.cache import CacheRecord

        with tempfile.TemporaryDirectory() as temp_dir:
            record = make_record(temp_dir)
            payload = record.to_json_dict()
            self.assertIsInstance(payload["source_audio_path"], str)
            self.assertEqual(CacheRecord.from_json_dict(payload), record)
            with self.assertRaises(Exception):
                record.speaker_id = "changed"

    def test_state_hash_is_key_order_independent_and_detects_change(self):
        from speechtokenizer.speaker_identity.cache import state_dict_sha256

        first = {"b": torch.tensor([2], dtype=torch.int16), "a": torch.tensor([1.0])}
        reordered = {"a": first["a"].clone(), "b": first["b"].clone()}
        changed = {"a": torch.tensor([2.0]), "b": first["b"].clone()}
        self.assertEqual(state_dict_sha256(first), state_dict_sha256(reordered))
        self.assertNotEqual(state_dict_sha256(first), state_dict_sha256(changed))

    def test_assert_state_dict_unchanged_hard_fails_on_difference(self):
        from speechtokenizer.speaker_identity.cache import assert_state_dict_unchanged

        with self.assertRaisesRegex(RuntimeError, "changed"):
            assert_state_dict_unchanged({"x": torch.zeros(1)}, {"x": torch.ones(1)})


class CacheValidationTest(unittest.TestCase):
    def test_valid_record_loads_full_item(self):
        from speechtokenizer.speaker_identity.cache import load_cache_item, validate_cache_record

        with tempfile.TemporaryDirectory() as temp_dir:
            record = make_record(temp_dir)
            audit = validate_cache_record(record)
            item = load_cache_item(record)
            self.assertEqual(audit["code_shape"], [3, 5])
            self.assertEqual(item["codes"].shape, (3, 5))
            self.assertEqual(item["audio"].shape, (1600,))

    def test_missing_path_is_actionable(self):
        from speechtokenizer.speaker_identity.cache import validate_cache_record

        with tempfile.TemporaryDirectory() as temp_dir:
            record = make_record(temp_dir, codes_path=Path(temp_dir) / "missing.npz")
            with self.assertRaisesRegex(ValueError, "codes_path.*does not exist"):
                validate_cache_record(record)

    def test_codes_require_integer_l_by_t_shape_and_range(self):
        from speechtokenizer.speaker_identity.cache import validate_cache_record

        cases = [
            (np.zeros((3, 5), dtype=np.float32), "integer"),
            (np.zeros((3, 2, 5), dtype=np.int64), "two-dimensional"),
            (np.zeros((2, 5), dtype=np.int64), "rvq_layers"),
            (np.full((3, 5), 1024, dtype=np.int64), r"\[0, 1024\)"),
            (np.full((3, 5), -1, dtype=np.int64), r"\[0, 1024\)"),
        ]
        for codes, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp_dir:
                record = make_record(temp_dir)
                np.savez_compressed(record.codes_path, codes=codes)
                with self.assertRaisesRegex(ValueError, message):
                    validate_cache_record(record)

    def test_audio_requires_finite_mono_16k(self):
        from speechtokenizer.speaker_identity.cache import validate_cache_record

        cases = [
            (np.zeros(100, dtype=np.float32), 8000, "16000"),
            (np.zeros((100, 2), dtype=np.float32), 16000, "mono"),
            (np.array([0.0, np.nan], dtype=np.float32), 16000, "finite"),
        ]
        for audio, sample_rate, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp_dir:
                record = make_record(temp_dir, valid_sample_count=len(audio))
                sf.write(record.reconstruction_path, audio, sample_rate, format="WAV", subtype="FLOAT")
                with self.assertRaisesRegex(ValueError, message):
                    validate_cache_record(record)

    def test_lengths_frames_and_hashes_must_be_valid(self):
        from speechtokenizer.speaker_identity.cache import validate_cache_record

        overrides = [
            ({"original_sample_count": 0}, "original_sample_count"),
            ({"code_frame_count": 0}, "code_frame_count"),
            ({"valid_sample_count": 0}, "valid_sample_count"),
            ({"valid_sample_count": 1601}, "valid_sample_count"),
            ({"checkpoint_sha256": ""}, "checkpoint_sha256"),
            ({"config_sha256": ""}, "config_sha256"),
        ]
        for values, message in overrides:
            with self.subTest(values=values), tempfile.TemporaryDirectory() as temp_dir:
                with self.assertRaisesRegex(ValueError, message):
                    validate_cache_record(make_record(temp_dir, **values))


class StorageAndWriteTest(unittest.TestCase):
    def test_estimate_and_preflight_return_audit(self):
        from speechtokenizer.speaker_identity.cache import estimate_cache_storage, preflight_cache_storage

        estimate = estimate_cache_storage(2, 32000, 100, 3)
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "speechtokenizer.speaker_identity.cache.shutil.disk_usage",
            return_value=(1000000, 100, estimate["estimated_bytes"] * 2),
        ):
            audit = preflight_cache_storage(Path(temp_dir) / "cache", estimate["estimated_bytes"])
        self.assertEqual(audit["estimated_bytes"], estimate["estimated_bytes"])
        self.assertGreaterEqual(audit["free_bytes"], audit["required_free_bytes"])

    def test_preflight_rejects_budget_and_free_space_failures(self):
        from speechtokenizer.speaker_identity.cache import preflight_cache_storage

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "budget"):
                preflight_cache_storage(Path(temp_dir) / "cache", 11, max_bytes=10)
            with patch(
                "speechtokenizer.speaker_identity.cache.shutil.disk_usage",
                return_value=(1000, 900, 100),
            ), self.assertRaisesRegex(ValueError, "free space"):
                preflight_cache_storage(Path(temp_dir) / "cache", 100, free_space_multiplier=1.2)

    def test_storage_inputs_reject_bool_negative_and_invalid_multiplier(self):
        from speechtokenizer.speaker_identity.cache import estimate_cache_storage, preflight_cache_storage

        with self.assertRaises((TypeError, ValueError)):
            estimate_cache_storage(True, 1, 1, 1)
        with self.assertRaises((TypeError, ValueError)):
            estimate_cache_storage(1, -1, 1, 1)
        with tempfile.TemporaryDirectory() as temp_dir, self.assertRaises((TypeError, ValueError)):
            preflight_cache_storage(Path(temp_dir) / "cache", 1, free_space_multiplier=1.0)

    def test_atomic_npz_and_flac_replace_completed_temp_files(self):
        from speechtokenizer.speaker_identity.cache import atomic_write_flac, atomic_write_npz

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            observed = []
            real_replace = os.replace

            def inspect_replace(source, destination):
                observed.append((Path(source), Path(destination), Path(source).exists()))
                real_replace(source, destination)

            with patch("speechtokenizer.speaker_identity.cache.os.replace", side_effect=inspect_replace):
                atomic_write_npz(root / "codes.npz", np.zeros((1, 2), dtype=np.int64))
                atomic_write_flac(root / "audio.flac", np.zeros(320, dtype=np.float32), 16000)
            self.assertTrue(all(existed for _, _, existed in observed))
            self.assertTrue((root / "codes.npz").is_file())
            self.assertTrue((root / "audio.flac").is_file())
            self.assertFalse(any(path.name.startswith(".tmp-") for path in root.iterdir()))


class ResumeAndCropTest(unittest.TestCase):
    def test_resume_requires_valid_files_and_exact_manifest_metadata(self):
        from speechtokenizer.speaker_identity.cache import CacheRecord, is_resumable_cache_item

        with tempfile.TemporaryDirectory() as temp_dir:
            record = make_record(temp_dir)
            self.assertTrue(is_resumable_cache_item(record, record.to_json_dict()))
            mismatched = dict(record.to_json_dict(), checkpoint_sha256="other")
            self.assertFalse(is_resumable_cache_item(record, mismatched))
            Path(record.codes_path).unlink()
            self.assertFalse(is_resumable_cache_item(record, record.to_json_dict()))
            self.assertEqual(CacheRecord.from_json_dict(record.to_json_dict()), record)

    def test_crop_slices_audio_and_codes_with_same_bounds(self):
        from speechtokenizer.speaker_identity.cache import CropBounds, crop_cache_item

        with tempfile.TemporaryDirectory() as temp_dir:
            record = make_record(temp_dir)
            bounds = CropBounds(audio_start=320, audio_end=960, token_start=1, token_end=3)
            cropped = crop_cache_item(record, bounds)
            self.assertEqual(cropped["audio"].shape, (640,))
            self.assertEqual(cropped["codes"].shape, (3, 2))
            self.assertEqual(cropped["bounds"], bounds)


if __name__ == "__main__":
    unittest.main()
