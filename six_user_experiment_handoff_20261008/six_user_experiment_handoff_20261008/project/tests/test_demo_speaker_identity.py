import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "3用户demo" / "speechtokenizer_now" / "speechtokenizer" / "三用户中心路由通信demo"
DEMO_MODULE = DEMO_DIR / "speaker_identity.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_demo_module():
    spec = importlib.util.spec_from_file_location("demo_speaker_identity", DEMO_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DemoSpeakerIdentityTests(unittest.TestCase):
    def test_streaming_identifier_accepts_ecapa_backend_via_extractor_interface(self):
        module = load_demo_module()

        calls = []

        class FakeExtractor:
            def __init__(
                self,
                backend="mfcc",
                sample_rate=16000,
                n_mfcc=40,
                device="cpu",
                ecapa_source="",
                ecapa_savedir="",
            ):
                calls.append(
                    {
                        "backend": backend,
                        "sample_rate": sample_rate,
                        "n_mfcc": n_mfcc,
                        "device": device,
                        "ecapa_source": ecapa_source,
                        "ecapa_savedir": str(ecapa_savedir),
                    }
                )

            def from_path(self, path):
                return np.array([1.0, 0.0], dtype=np.float32) if "A" in str(path) else np.array([0.0, 1.0], dtype=np.float32)

            def from_waveform(self, waveform):
                return np.array([1.0, 0.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for speaker in ("A", "B"):
                speaker_dir = root / speaker
                speaker_dir.mkdir()
                sf.write(speaker_dir / f"{speaker}_profile.wav", np.ones(1600, dtype=np.float32), 16000)

            original_extractor = module.SpeakerEmbeddingExtractor
            module.SpeakerEmbeddingExtractor = FakeExtractor
            try:
                identifier = module.StreamingSpeakerIdentifier(
                    profile_dir=str(root),
                    sample_rate=16000,
                    window_sec=0.1,
                    hop_sec=0.1,
                    threshold=0.5,
                    backend="ecapa",
                    speaker_device="cuda",
                    ecapa_source="speechbrain/test-ecapa",
                    ecapa_savedir="output/models/test-ecapa",
                )
            finally:
                module.SpeakerEmbeddingExtractor = original_extractor

        prediction = identifier.update("A", np.ones(1600, dtype=np.float32))

        self.assertEqual(calls[0]["backend"], "ecapa")
        self.assertEqual(calls[0]["device"], "cuda")
        self.assertEqual(calls[0]["ecapa_source"], "speechbrain/test-ecapa")
        self.assertEqual(calls[0]["ecapa_savedir"].replace("\\", "/"), "output/models/test-ecapa")
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.predicted_speaker, "A")

    def test_streaming_identifier_uses_window_and_hop_per_sender(self):
        module = load_demo_module()
        from scripts.speaker_identity_utils import SpeakerProfile

        profiles = {
            "A": SpeakerProfile("A", np.array([1.0, 0.0], dtype=np.float32), 1),
            "B": SpeakerProfile("B", np.array([0.0, 1.0], dtype=np.float32), 1),
        }
        calls = []

        def embedding_fn(waveform, sample_rate, n_mfcc):
            calls.append((len(waveform), sample_rate, n_mfcc))
            return np.array([1.0, 0.0], dtype=np.float32)

        identifier = module.StreamingSpeakerIdentifier(
            profile_dir="",
            sample_rate=4,
            window_sec=1.0,
            hop_sec=0.5,
            threshold=0.5,
            profiles=profiles,
            embedding_fn=embedding_fn,
        )

        self.assertIsNone(identifier.update("senderA", np.ones(2, dtype=np.float32)))
        first = identifier.update("senderA", np.ones(2, dtype=np.float32))
        self.assertIsNotNone(first)
        self.assertEqual(first.predicted_speaker, "A")
        self.assertTrue(first.verified)
        self.assertEqual(calls[-1], (4, 4, 40))

        self.assertIsNone(identifier.update("senderA", np.ones(1, dtype=np.float32)))
        second = identifier.update("senderA", np.ones(1, dtype=np.float32))
        self.assertIsNotNone(second)
        self.assertEqual(second.predicted_speaker, "A")

        other_sender = identifier.update("senderB", np.ones(2, dtype=np.float32))
        self.assertIsNone(other_sender)

    def test_streaming_identifier_loads_profiles_from_speaker_subdirs(self):
        module = load_demo_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr = 16000
            t = np.arange(sr // 4, dtype=np.float32) / sr
            for speaker, freq in {"A": 220.0, "B": 440.0}.items():
                speaker_dir = root / speaker
                speaker_dir.mkdir()
                audio = 0.1 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
                sf.write(speaker_dir / f"{speaker}_profile.wav", audio, sr)

            identifier = module.StreamingSpeakerIdentifier(
                profile_dir=str(root),
                sample_rate=sr,
                window_sec=0.25,
                hop_sec=0.25,
                threshold=0.0,
            )

        self.assertEqual(set(identifier.profiles), {"A", "B"})
        self.assertEqual(identifier.profiles["A"].utterance_count, 1)
        self.assertEqual(identifier.profiles["B"].utterance_count, 1)


if __name__ == "__main__":
    unittest.main()
