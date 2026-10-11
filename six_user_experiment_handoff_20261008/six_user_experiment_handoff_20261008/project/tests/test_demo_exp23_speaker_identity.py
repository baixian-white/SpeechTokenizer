import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_MODULE = PROJECT_ROOT / '3用户demo' / 'speechtokenizer_now' / 'speechtokenizer' / '三用户中心路由通信demo' / 'speaker_identity.py'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GROUP_CLIENT = DEMO_MODULE.with_name('group_client.py')


def load_module():
    spec = importlib.util.spec_from_file_location('demo_exp23_identity', DEMO_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_group_client():
    module_name = 'demo_group_client_defaults'
    if str(GROUP_CLIENT.parent) not in sys.path:
        sys.path.insert(0, str(GROUP_CLIENT.parent))
    spec = importlib.util.spec_from_file_location(module_name, GROUP_CLIENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakePrediction:
    predicted_speaker = 'p225'
    score = 0.9
    margin = 0.8
    verified = True
    scores = {'p225': 0.9}


class FakeBundle:
    speaker_count = 110
    def __init__(self):
        self.codes = None

    def predict(self, waveform, codes=None):
        self.codes = codes
        return FakePrediction()


class DemoExp23Test(unittest.TestCase):
    def test_injected_bundle_classifier_has_priority(self):
        module = load_module()
        bundle = FakeBundle()
        identifier = module.StreamingSpeakerIdentifier(profile_dir='', sample_rate=16000, window_sec=0.1, hop_sec=0.1, bundle_classifier=bundle)
        codes = np.ones((3, 5), dtype=np.int64)
        prediction = identifier.update('sender', np.ones(1600, dtype=np.float32), codes=codes)
        self.assertEqual(prediction.predicted_speaker, 'p225')
        self.assertEqual(identifier.speaker_count, 110)
        np.testing.assert_array_equal(bundle.codes, codes)

    def test_group_client_passes_received_codes_to_identifier(self):
        source = GROUP_CLIENT.read_text(encoding='utf-8')
        self.assertIn('self.update_speaker_identity(stream, speaker_pcm, codes)', source)
        self.assertIn('self.speaker_identifier.update(stream.sender_id, pcm, codes=codes)', source)

    def test_interactive_demo_defaults_to_recognition_focused_profile(self):
        source = GROUP_CLIENT.read_text(encoding='utf-8')
        self.assertIn('DEFAULT_RECOGNITION_CANDIDATES = "p225,p226,p227,p228,p229,p230"', source)
        self.assertIn('DEFAULT_RECOGNITION_CHUNK_SECONDS = 1.5', source)
        self.assertIn("default=DEFAULT_RECOGNITION_CANDIDATES", source)
        self.assertIn("default=DEFAULT_RECOGNITION_CHUNK_SECONDS", source)
        self.assertIn('OPEN_SET_CANDIDATE_SENTINEL = "all"', source)
        self.assertIn('Pass "all"', source)

    def test_all_sentinel_restores_full_classifier_output(self):
        module = load_group_client()
        self.assertIsNone(module.parse_speaker_labels("all"))
        self.assertIsNone(module.parse_speaker_labels("ALL"))
        self.assertEqual(module.parse_speaker_labels("p225,p226,p225"), ("p225", "p226"))


if __name__ == '__main__':
    unittest.main()
