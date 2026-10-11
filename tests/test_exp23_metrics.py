import unittest

import numpy as np


class MetricsTest(unittest.TestCase):
    def test_temperature_and_metrics_are_finite(self):
        from speechtokenizer.speaker_identity.metrics import classification_metrics, fit_temperature
        logits = np.array([[3.0, 0.0], [0.0, 3.0], [2.0, 1.0]])
        labels = np.array([0, 1, 0])
        temperature = fit_temperature(logits, labels)
        metrics = classification_metrics(logits / temperature, labels)
        self.assertGreater(temperature, 0.0)
        self.assertEqual(metrics['top1'], 1.0)
        self.assertIn('macro_f1', metrics)

    def test_bootstrap_contains_point_estimate(self):
        from speechtokenizer.speaker_identity.metrics import hierarchical_bootstrap_accuracy
        rows = [{'speaker_id': 'a', 'utterance_group': '1', 'correct': 1}, {'speaker_id': 'b', 'utterance_group': '2', 'correct': 0}]
        result = hierarchical_bootstrap_accuracy(rows, replicates=100, seed=42)
        self.assertLessEqual(result['ci_low'], result['point_estimate'])
        self.assertGreaterEqual(result['ci_high'], result['point_estimate'])

    def test_test_evaluation_requires_explicit_flag(self):
        from speechtokenizer.speaker_identity.metrics import assert_evaluation_allowed
        with self.assertRaisesRegex(PermissionError, 'allow-test-evaluation'):
            assert_evaluation_allowed('test', False)
        assert_evaluation_allowed('test', True)


if __name__ == '__main__':
    unittest.main()
