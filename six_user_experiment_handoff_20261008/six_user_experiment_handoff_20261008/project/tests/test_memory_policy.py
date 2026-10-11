import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = next(ROOT.glob("3*demo/speechtokenizer_now/speechtokenizer/*/run_metrics_experiment.py")).parent
sys.path.insert(0, str(DEMO))
from run_metrics_experiment import RAM_REQUIREMENTS_GIB


class MemoryPolicyTest(unittest.TestCase):
    def test_strict_policy_keeps_original_six_client_guard(self):
        self.assertEqual(RAM_REQUIREMENTS_GIB["strict"]["D"], 8.0)
        self.assertEqual(RAM_REQUIREMENTS_GIB["strict"]["C"], 8.0)

    def test_balanced_policy_is_explicitly_lower(self):
        self.assertEqual(RAM_REQUIREMENTS_GIB["balanced"]["D"], 6.0)
        self.assertLess(RAM_REQUIREMENTS_GIB["balanced"]["D"], RAM_REQUIREMENTS_GIB["strict"]["D"])
