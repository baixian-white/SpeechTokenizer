import unittest

from experiments_semcom.run_public_nq8_layer_sweep import recommend_layer


class RecommendLayerTests(unittest.TestCase):
    def test_selects_first_elbow_that_meets_realtime(self):
        rows = [
            {
                "rvq_layers": 1,
                "stoi_mean": 0.70,
                "pesq_mean": 1.10,
                "si_snr_mean": -12.0,
                "mel_error_mean": 1.40,
                "total_online_bps_mean": 3300.0,
                "rtf_mean": 0.20,
            },
            {
                "rvq_layers": 2,
                "stoi_mean": 0.82,
                "pesq_mean": 1.45,
                "si_snr_mean": -2.0,
                "mel_error_mean": 1.00,
                "total_online_bps_mean": 3900.0,
                "rtf_mean": 0.25,
            },
            {
                "rvq_layers": 3,
                "stoi_mean": 0.86,
                "pesq_mean": 1.62,
                "si_snr_mean": -0.5,
                "mel_error_mean": 0.92,
                "total_online_bps_mean": 4500.0,
                "rtf_mean": 0.30,
            },
            {
                "rvq_layers": 4,
                "stoi_mean": 0.865,
                "pesq_mean": 1.65,
                "si_snr_mean": -0.4,
                "mel_error_mean": 0.91,
                "total_online_bps_mean": 5100.0,
                "rtf_mean": 0.35,
            },
        ]

        decision = recommend_layer(rows, min_stoi=0.85, max_rtf=1.0, elbow_gain_threshold=0.02)

        self.assertEqual(decision["recommended_layers"], 3)
        self.assertEqual(decision["reason"], "quality_elbow")


if __name__ == "__main__":
    unittest.main()
