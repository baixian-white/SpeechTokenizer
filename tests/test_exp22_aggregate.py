import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Exp22AggregateTests(unittest.TestCase):
    def test_aggregate_rows_computes_mean_sample_std_and_ci(self):
        from scripts.aggregate_exp22_speaker_identity import aggregate_rows

        rows = [
            {"model": "base", "feature_kind": "latent_stats", "L": "3", "seed": 41, "top1_accuracy": "0.5"},
            {"model": "base", "feature_kind": "latent_stats", "L": "3", "seed": 42, "top1_accuracy": "0.7"},
        ]

        [agg] = aggregate_rows(rows, ("model", "feature_kind", "L"), ("top1_accuracy",))

        self.assertEqual(agg["n_seeds"], 2)
        self.assertEqual(agg["seeds"], "41 42")
        self.assertAlmostEqual(agg["top1_accuracy_mean"], 0.6)
        self.assertAlmostEqual(agg["top1_accuracy_std"], math.sqrt(0.02))
        self.assertAlmostEqual(agg["top1_accuracy_ci95"], 1.2706, places=4)


if __name__ == "__main__":
    unittest.main()
