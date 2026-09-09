import unittest


class AggregateTest(unittest.TestCase):
    def test_product_and_mechanism_claims_are_separate(self):
        from speechtokenizer.speaker_identity.aggregate import summarize_claims
        rows = [
            {'seed': 41, 'top1': 0.90, 'top5': 0.98, 'macro_f1': 0.90},
            {'seed': 42, 'top1': 0.91, 'top5': 0.99, 'macro_f1': 0.905},
            {'seed': 43, 'top1': 0.92, 'top5': 1.00, 'macro_f1': 0.91},
        ]
        result = summarize_claims(rows, bootstrap_ci_low=0.89, fusion_gain=0.005, relative_error_reduction=0.05)
        self.assertTrue(result['product_success'])
        self.assertFalse(result['mechanism_success'])
        self.assertAlmostEqual(result['std_top1'], 0.01)
        self.assertAlmostEqual(result['mean_top5'], 0.99)
        self.assertAlmostEqual(result['std_top5'], 0.01)
        self.assertAlmostEqual(result['std_macro_f1'], 0.005)
        self.assertEqual(result['bootstrap_ci_low'], 0.89)


if __name__ == '__main__':
    unittest.main()
