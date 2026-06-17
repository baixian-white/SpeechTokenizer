import unittest
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def candidate():
    return {
        "candidate_id": "nas_seed42_000896",
        "search_mode": "random",
        "seed": 42,
        "sample_index": 896,
        "encoder_strides": [5, 4, 4, 4],
        "decoder_strides": [4, 4, 4, 5],
        "decoder_condition": "geometry_matched_decoder",
        "decoder_ops": "not_searched",
        "seanet_ratios_arg": [4, 4, 4, 5],
        "dimension": 1024,
        "sample_rate": 16000,
        "encoder_downsample_rate": 320,
        "latent_rate": 50,
        "n_q": 3,
        "codebook_size": 1024,
        "n_filters": 24,
        "compress": 2,
        "lstm": 2,
        "activation": "ELU",
        "layer_ops_list": ["sep_k3", "std_k5", "dil_k3", "std_k3"],
        "layer_se_list": [False, True, False, True],
    }


class EncoderOnlyHandoffConfigTests(unittest.TestCase):
    def test_best_config_exports_encoder_only_decoder_condition_from_selected_row(self):
        from nas.encoder_handoff import build_encoder_only_handoff_config

        handoff = build_encoder_only_handoff_config(
            candidate(),
            selected_row={
                "candidate_id": "nas_seed42_000896",
                "decoder_strides": [8, 5, 4, 2],
                "decoder_condition": "frozen_teacher_decoder",
            },
            base_config={"strides": [8, 5, 4, 2]},
        )

        self.assertTrue(handoff["encoder_only"])
        self.assertEqual(handoff["candidate_id"], "nas_seed42_000896")
        self.assertEqual(handoff["encoder_strides"], [5, 4, 4, 4])
        self.assertEqual(handoff["seanet_ratios_arg"], [4, 4, 4, 5])
        self.assertEqual(handoff["decoder_condition"], "frozen_teacher_decoder")
        self.assertEqual(handoff["decoder_strides"], [8, 5, 4, 2])
        self.assertEqual(handoff["source_candidate_decoder_condition"], "geometry_matched_decoder")
        self.assertEqual(handoff["source_candidate_decoder_strides"], [4, 4, 4, 5])

    def test_encoder_only_loader_normalization_preserves_handoff_decoder_fields(self):
        from nas.encoder_handoff import normalize_encoder_only_nas_config

        handoff = dict(candidate())
        handoff.update(
            {
                "encoder_only": True,
                "decoder_condition": "frozen_teacher_decoder",
                "decoder_strides": [8, 5, 4, 2],
                "source_candidate_decoder_condition": "geometry_matched_decoder",
                "source_candidate_decoder_strides": [4, 4, 4, 5],
            }
        )

        normalized = normalize_encoder_only_nas_config(handoff, {"strides": [8, 5, 4, 2], "seed": 42})

        self.assertTrue(normalized["encoder_only"])
        self.assertEqual(normalized["decoder_condition"], "frozen_teacher_decoder")
        self.assertEqual(normalized["decoder_strides"], [8, 5, 4, 2])
        self.assertEqual(normalized["source_candidate_decoder_condition"], "geometry_matched_decoder")
        self.assertEqual(normalized["source_candidate_decoder_strides"], [4, 4, 4, 5])

    def test_encoder_only_loader_defaults_raw_candidate_to_base_decoder(self):
        from nas.encoder_handoff import normalize_encoder_only_nas_config

        normalized = normalize_encoder_only_nas_config(candidate(), {"strides": [8, 5, 4, 2], "seed": 42})

        self.assertTrue(normalized["encoder_only"])
        self.assertEqual(normalized["decoder_condition"], "base_config_decoder")
        self.assertEqual(normalized["decoder_strides"], [8, 5, 4, 2])
        self.assertEqual(normalized["source_candidate_decoder_condition"], "geometry_matched_decoder")
        self.assertEqual(normalized["source_candidate_decoder_strides"], [4, 4, 4, 5])


if __name__ == "__main__":
    unittest.main()
