import hashlib
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class Exp2ScriptChainTests(unittest.TestCase):
    def setUp(self):
        temp_root = PROJECT_ROOT / "tests" / "_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.root = temp_root / f"exp2_script_chain_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.base_config = self.root / "base_config.json"
        self.best_config = self.root / "best_seanet_config.json"
        self.train_manifest = self.root / "data" / "train_files.txt"
        self.valid_manifest = self.root / "data" / "valid_files.txt"

        audio = self.root / "data" / "audio.flac"
        feat = self.root / "data" / "audio.hubert.npy"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"fake-audio")
        feat.write_bytes(b"fake-feature")
        self.train_manifest.write_text(f"{audio}\t{feat}\n", encoding="utf-8")
        self.valid_manifest.write_text(f"{audio}\t{feat}\n", encoding="utf-8")

        write_json(
            self.base_config,
            {
                "n_filters": 64,
                "strides": [8, 5, 4, 2],
                "dimension": 1024,
                "semantic_dimension": 768,
                "codebook_size": 1024,
                "n_q": 3,
                "train_files": str(self.train_manifest),
                "valid_files": str(self.valid_manifest),
                "results_folder": "Log/spt_base",
                "sample_rate": 16000,
                "batch_size": 8,
                "epochs": 60,
                "learning_rate": 0.0001,
                "intial_learning_rate": 0.0001,
                "distill_loss_lambda": 120,
                "segment_size": 16000,
                "num_workers": 8,
            },
        )
        write_json(
            self.best_config,
            {
                "candidate_id": "nas_seed42_000896",
                "encoder_strides": [5, 4, 4, 4],
                "n_filters": 24,
                "compress": 2,
                "lstm": 1,
                "activation": "Snake",
                "layer_ops_list": ["skip", "std_k7", "sep_k9", "dil_k5"],
                "layer_se_list": [False, False, False, False],
                "handoff_schema": "encoder_only_nas_v1",
                "decoder_condition": "frozen_teacher_decoder",
                "decoder_strides": [8, 5, 4, 2],
                "dimension": 1024,
                "sample_rate": 16000,
                "encoder_downsample_rate": 320,
                "latent_rate": 50,
                "n_q": 3,
                "codebook_size": 1024,
            },
        )

    def tearDown(self):
        if self.root.exists() and self.root.parent.name == "_tmp":
            shutil.rmtree(self.root)

    def _prepare(self, run_id, *, debug=False, tracer=False):
        from scripts.prepare_exp2_config import prepare_exp2_config

        return prepare_exp2_config(
            base_config=self.base_config,
            exp1_best_config=self.best_config,
            run_id=run_id,
            seed=42,
            experiments_root=self.root / "experiments",
            debug=debug,
            tracer=tracer,
        )

    def test_prepare_results_folder_points_to_run_checkpoints(self):
        result = self._prepare("exp2_formal_seed42")
        cfg = json.loads(result.config_path.read_text(encoding="utf-8"))

        self.assertEqual(Path(cfg["results_folder"]).resolve(), (result.run_dir / "checkpoints").resolve())

    def test_prepare_nas_encoder_config_uses_best_seanet_not_raw_candidate(self):
        result = self._prepare("exp2_nas_handoff_seed42")
        cfg = json.loads(result.config_path.read_text(encoding="utf-8"))

        self.assertTrue(cfg["nas_encoder_config"].endswith("best_seanet_config.json"))
        self.assertNotIn("best_candidate_raw", cfg["nas_encoder_config"])
        copied = json.loads(Path(cfg["nas_encoder_config"]).read_text(encoding="utf-8"))
        self.assertEqual(copied["candidate_id"], "nas_seed42_000896")
        self.assertEqual(copied["handoff_schema"], "encoder_only_nas_v1")

    def test_debug_tracer_and_formal_training_limits_are_different(self):
        formal = json.loads(self._prepare("exp2_formal_limits_seed42").config_path.read_text(encoding="utf-8"))
        debug = json.loads(
            self._prepare("exp2_debug_limits_seed42", debug=True).config_path.read_text(encoding="utf-8")
        )
        tracer = json.loads(
            self._prepare("exp2_tracer_limits_seed42", tracer=True).config_path.read_text(encoding="utf-8")
        )

        self.assertNotIn("max_train_steps", formal)
        self.assertEqual(formal["epochs"], 60)
        self.assertEqual(debug["max_train_steps"], 2)
        self.assertEqual(debug["epochs"], 1)
        self.assertEqual(tracer["max_train_steps"], 2)
        self.assertEqual(tracer["epochs"], 1)

    def test_formal_distill30_override_records_retrain_metadata(self):
        from scripts.prepare_exp2_config import prepare_exp2_config

        result = prepare_exp2_config(
            base_config=self.base_config,
            exp1_best_config=self.best_config,
            run_id="exp2_distill30_retrain_seed42",
            seed=42,
            experiments_root=self.root / "experiments",
            distill_loss_lambda=30,
            experiment_tag="distill30_retrain",
            experiment_note="Formal retrain with lower semantic distillation weight.",
        )
        cfg = json.loads(result.config_path.read_text(encoding="utf-8"))

        self.assertEqual(cfg["distill_loss_lambda"], 30)
        self.assertEqual(cfg["exp2_experiment_tag"], "distill30_retrain")
        self.assertEqual(cfg["exp2_distill_loss_lambda_source"], "override")
        self.assertIn("Formal retrain", cfg["exp2_experiment_note"])
        self.assertNotIn("pretrained_generator_checkpoint", cfg)
        self.assertNotIn("finetune_scope", cfg)
        command = result.command_path.read_text(encoding="utf-8")
        self.assertIn("scripts/train_example.py", command)
        self.assertIn(str(result.config_path), command)
        report = (result.run_dir / "reports" / "prepare_config.md").read_text(encoding="utf-8")
        self.assertIn("distill_loss_lambda: 30", report)
        self.assertIn("distill30_retrain", report)

    def test_package_checkpoint_manifest_contains_sha256(self):
        from scripts.package_checkpoint import package_checkpoint

        run_dir = self.root / "experiments" / "exp2_ckpt_seed42"
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir(parents=True)
        config_path = run_dir / "configs" / "scit_speech_base_config.json"
        write_json(config_path, {"results_folder": str(checkpoints)})
        source = checkpoints / "SpeechTokenizer_best_dev.pt"
        source.write_bytes(b"checkpoint-bytes")

        manifest = package_checkpoint(run_dir=run_dir, config=config_path)

        expected_hash = hashlib.sha256(b"checkpoint-bytes").hexdigest()
        self.assertEqual(manifest["sha256"], expected_hash)
        self.assertTrue((checkpoints / "SCIT-Speech-Base_best.pt").exists())
        self.assertEqual(
            json.loads((checkpoints / "checkpoint_manifest.json").read_text(encoding="utf-8"))["sha256"],
            expected_hash,
        )

    def test_summarize_missing_checkpoint_is_not_completed(self):
        from scripts.summarize_exp2 import summarize_exp2
        from scripts.experiment_utils import ensure_run_layout

        run_dir = ensure_run_layout(self.root / "experiments" / "exp2_partial_seed42")
        config_path = run_dir / "configs" / "scit_speech_base_config.json"
        write_json(config_path, {"run_id": run_dir.name, "results_folder": str(run_dir / "checkpoints")})

        status = summarize_exp2(run_dir=run_dir, config=config_path)

        self.assertIn(status["status"], {"partial", "failure"})
        self.assertNotEqual(status["status"], "completed")
        saved = json.loads((run_dir / "reports" / "status.json").read_text(encoding="utf-8"))
        self.assertNotEqual(saved["status"], "completed")


if __name__ == "__main__":
    unittest.main()
