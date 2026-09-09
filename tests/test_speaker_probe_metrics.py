import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SpeakerProbeMetricsTests(unittest.TestCase):
    def test_fit_and_score_handles_binary_topk_without_sklearn_binary_shape_error(self):
        from scripts.evaluate_speaker_probe import fit_and_score

        X_train = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        y_train = ["A", "A", "B", "B"]
        X_test = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
        y_test = ["A", "B"]

        metrics = fit_and_score(X_train, y_train, X_test, y_test, seed=42)

        self.assertEqual(metrics["speaker_count"], 2)
        self.assertEqual(metrics["test_count"], 2)
        self.assertEqual(metrics["top5_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
