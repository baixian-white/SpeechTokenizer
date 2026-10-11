import importlib.util
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE = PROJECT_ROOT / "speechtokenizer" / "speaker_identity" / "inference.py"


def load_inference_module():
    module_name = "speaker_identity_inference_candidate_test"
    try:
        torch_available = importlib.util.find_spec("torch") is not None
    except ValueError:
        torch_available = "torch" in sys.modules
    if not torch_available:
        fake_torch = types.ModuleType("torch")
        fake_torch.nn = types.SimpleNamespace(Module=object)
        sys.modules.setdefault("torch", fake_torch)
    spec = importlib.util.spec_from_file_location(module_name, INFERENCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SpeakerClassifierCandidateTests(unittest.TestCase):
    def test_candidate_labels_are_normalized_and_validated(self):
        module = load_inference_module()
        bundle = object.__new__(module.SpeakerClassifierBundle)
        bundle.labels = ["p225", "p226", "p227"]

        self.assertEqual(
            bundle._normalize_candidate_labels("p225, p226,p225"),
            ("p225", "p226"),
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            bundle._normalize_candidate_labels(" , ")
        with self.assertRaisesRegex(ValueError, "not present"):
            bundle._normalize_candidate_labels(["p225", "p999"])

    def test_output_space_reports_known_roster_size(self):
        module = load_inference_module()
        bundle = object.__new__(module.SpeakerClassifierBundle)
        bundle.speaker_count = 110
        bundle.candidate_labels = ("p225", "p226", "p227", "p228", "p229", "p230")
        self.assertEqual(bundle.output_speaker_count, 6)


if __name__ == "__main__":
    unittest.main()
