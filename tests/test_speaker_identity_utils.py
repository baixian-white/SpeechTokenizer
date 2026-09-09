import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SpeakerIdentityUtilsTests(unittest.TestCase):
    def test_parse_speaker_id_from_supported_corpora(self):
        from scripts.speaker_identity_utils import parse_speaker_id

        self.assertEqual(
            parse_speaker_id(r"H:\H-CODE\speechtokenizer\data\VCTK\wav48_silence_trimmed\p225\p225_056_mic1.flac"),
            "p225",
        )
        self.assertEqual(
            parse_speaker_id(r"H:\H-CODE\speechtokenizer\data\AISHELL\data_aishell\wav_extracted\test\S0764\BAC009S0764W0472.wav"),
            "S0764",
        )
        self.assertEqual(
            parse_speaker_id(r"H:\H-CODE\speechtokenizer\data\SpeechPretrain\LibriSpeech\test-clean\1089\134686\1089-134686-0000.flac"),
            "1089",
        )

    def test_cosine_predictor_identifies_nearest_profile(self):
        from scripts.speaker_identity_utils import SpeakerProfile, predict_speaker

        profiles = {
            "A": SpeakerProfile("A", np.array([1.0, 0.0], dtype=np.float32), 2),
            "B": SpeakerProfile("B", np.array([0.0, 1.0], dtype=np.float32), 2),
        }

        result = predict_speaker(np.array([0.9, 0.1], dtype=np.float32), profiles, threshold=0.5)

        self.assertEqual(result.predicted_speaker, "A")
        self.assertGreater(result.score, 0.9)
        self.assertGreater(result.margin, 0.5)
        self.assertTrue(result.verified)

    def test_eer_separates_clean_positive_and_negative_scores(self):
        from scripts.speaker_identity_utils import compute_eer

        eer, threshold = compute_eer(
            labels=[1, 1, 1, 0, 0, 0],
            scores=[0.91, 0.82, 0.73, 0.33, 0.22, 0.11],
        )

        self.assertEqual(eer, 0.0)
        self.assertGreater(threshold, 0.33)
        self.assertLessEqual(threshold, 0.73)

    def test_tar_at_far_uses_highest_allowed_false_alarm_threshold(self):
        from scripts.speaker_identity_utils import tar_at_far

        tar, threshold = tar_at_far(
            labels=[1, 1, 1, 0, 0, 0],
            scores=[0.91, 0.82, 0.73, 0.60, 0.30, 0.10],
            target_far=0.34,
        )

        self.assertTrue(math.isclose(tar, 1.0))
        self.assertGreaterEqual(threshold, 0.60)

    def test_per_speaker_split_is_deterministic_and_keeps_enroll_and_test(self):
        from scripts.speaker_identity_utils import split_enrollment_and_test

        groups = {
            "A": [Path(f"A_{idx}.wav") for idx in range(6)],
            "B": [Path(f"B_{idx}.wav") for idx in range(6)],
        }

        enroll1, test1 = split_enrollment_and_test(groups, enroll_per_speaker=2, test_per_speaker=3, seed=42)
        enroll2, test2 = split_enrollment_and_test(groups, enroll_per_speaker=2, test_per_speaker=3, seed=42)

        self.assertEqual(enroll1, enroll2)
        self.assertEqual(test1, test2)
        self.assertEqual({k: len(v) for k, v in enroll1.items()}, {"A": 2, "B": 2})
        self.assertEqual({k: len(v) for k, v in test1.items()}, {"A": 3, "B": 3})
        self.assertTrue(set(enroll1["A"]).isdisjoint(set(test1["A"])))

    def test_enroll_profiles_accepts_custom_embedding_extractor(self):
        from scripts.speaker_identity_utils import enroll_profiles

        class DummyExtractor:
            def from_path(self, path):
                if str(path).startswith("A"):
                    return np.array([1.0, 0.0], dtype=np.float32)
                return np.array([0.0, 1.0], dtype=np.float32)

        profiles = enroll_profiles(
            {
                "A": [Path("A_1.wav"), Path("A_2.wav")],
                "B": [Path("B_1.wav"), Path("B_2.wav")],
            },
            extractor=DummyExtractor(),
        )

        self.assertEqual(set(profiles), {"A", "B"})
        self.assertTrue(np.allclose(profiles["A"].centroid, [1.0, 0.0]))
        self.assertTrue(np.allclose(profiles["B"].centroid, [0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
