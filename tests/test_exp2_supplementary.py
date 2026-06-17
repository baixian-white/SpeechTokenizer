import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class DummyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.quantizer = torch.nn.Linear(2, 2)
        self.transform = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Linear(2, 2)


class Exp2SupplementaryTests(unittest.TestCase):
    def setUp(self):
        temp_root = PROJECT_ROOT / "tests" / "_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.root = temp_root / f"exp2_supplementary_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.base_run_dir = self.root / "base_run"
        self.base_run_dir.mkdir(parents=True)
        self.base_config = self.base_run_dir / "configs" / "scit_speech_base_config.json"
        self.nas_config = self.base_run_dir / "configs" / "best_seanet_config.json"
        self.checkpoint = self.base_run_dir / "checkpoints" / "SCIT-Speech-Base_best.pt"
        self.train_manifest = self.base_run_dir / "artifacts" / "train_files.txt"
        self.valid_manifest = self.base_run_dir / "artifacts" / "valid_files.txt"
        self.train_manifest.parent.mkdir(parents=True)
        self.train_manifest.write_text("audio.wav\tfeature.npy\n", encoding="utf-8")
        self.valid_manifest.write_text("audio.wav\tfeature.npy\n", encoding="utf-8")
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"fake-checkpoint")
        write_json(self.nas_config, {"candidate_id": "nas_seed42_000896"})
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
                "results_folder": str(self.base_run_dir / "checkpoints"),
                "sample_rate": 16000,
                "batch_size": 8,
                "epochs": 60,
                "learning_rate": 0.0001,
                "intial_learning_rate": 0.0001,
                "distill_loss_lambda": 120,
                "segment_size": 16000,
                "num_workers": 8,
                "nas_encoder_config": str(self.nas_config),
            },
        )

    def tearDown(self):
        if self.root.exists() and self.root.parent.name == "_tmp":
            shutil.rmtree(self.root)

    def test_decoder_only_freeze_leaves_only_decoder_trainable(self):
        from scripts.exp2_supplementary import apply_generator_train_scope

        model = DummyGenerator()
        report = apply_generator_train_scope(model, "decoder_only_acoustic")

        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.quantizer.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.transform.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.decoder.parameters()))
        self.assertEqual(report["scope"], "decoder_only_acoustic")
        self.assertEqual(report["trainable_modules"], ["decoder"])
        self.assertGreater(report["frozen_param_count"], 0)
        self.assertGreater(report["trainable_param_count"], 0)

    def test_prepare_decoder_only_config_records_checkpoint_and_command(self):
        from scripts.exp2_supplementary import prepare_decoder_only_acoustic_run

        run_dir = self.root / "supplementary" / "S1_decoder_only_seed42"
        result = prepare_decoder_only_acoustic_run(
            base_config_path=self.base_config,
            base_run_dir=self.base_run_dir,
            run_dir=run_dir,
            checkpoint_path=self.checkpoint,
            learning_rate=3e-6,
            distill_loss_lambda=0,
            epochs=3,
            max_train_steps=1200,
            seed=42,
        )

        cfg = json.loads(result.config_path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["finetune_scope"], "decoder_only_acoustic")
        self.assertEqual(cfg["pretrained_generator_checkpoint"], str(self.checkpoint))
        self.assertEqual(cfg["learning_rate"], 3e-6)
        self.assertEqual(cfg["intial_learning_rate"], 3e-6)
        self.assertEqual(cfg["distill_loss_lambda"], 0)
        self.assertEqual(cfg["epochs"], 3)
        self.assertEqual(cfg["max_train_steps"], 1200)
        self.assertLessEqual(cfg["save_model_steps"], cfg["max_train_steps"])
        self.assertEqual(Path(cfg["results_folder"]).resolve(), (run_dir / "checkpoints").resolve())
        self.assertTrue((run_dir / "commands" / "run_decoder_only_acoustic_finetune.txt").exists())
        command = (run_dir / "commands" / "run_decoder_only_acoustic_finetune.txt").read_text(encoding="utf-8")
        self.assertIn("scripts/train_decoder_only_finetune.py", command)
        self.assertIn(str(result.config_path), command)
        self.assertTrue((run_dir / "reports" / "launch_guide.md").exists())

    def test_prepare_static_distill_ablation_config_records_variant_and_command(self):
        from scripts.exp2_supplementary import prepare_distill_weight_ablation_run

        run_dir = self.root / "supplementary" / "S2_B_distill60_seed42"
        result = prepare_distill_weight_ablation_run(
            base_config_path=self.base_config,
            base_run_dir=self.base_run_dir,
            run_dir=run_dir,
            checkpoint_path=self.checkpoint,
            variant_id="S2-B",
            distill_loss_lambda=60,
            learning_rate=1e-5,
            epochs=3,
            max_train_steps=3000,
            save_model_steps=500,
            seed=42,
        )

        cfg = json.loads(result.config_path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["finetune_scope"], "distill_weight_ablation")
        self.assertEqual(cfg["exp2_supplementary_variant"], "S2-B")
        self.assertEqual(cfg["pretrained_generator_checkpoint"], str(self.checkpoint))
        self.assertEqual(cfg["distill_loss_lambda"], 60)
        self.assertNotIn("distill_loss_schedule", cfg)
        self.assertEqual(cfg["learning_rate"], 1e-5)
        self.assertEqual(cfg["intial_learning_rate"], 1e-5)
        self.assertEqual(cfg["max_train_steps"], 3000)
        self.assertEqual(cfg["save_model_steps"], 500)
        self.assertEqual(cfg["frozen_modules"], [])
        self.assertEqual(cfg["trainable_modules"], ["encoder", "quantizer", "transform", "decoder"])
        command = result.command_path.read_text(encoding="utf-8")
        self.assertIn("scripts/train_distill_weight_ablation.py", command)
        self.assertIn(str(result.config_path), command)
        self.assertTrue(result.launch_guide_path.exists())

    def test_prepare_scheduled_distill_ablation_records_linear_decay(self):
        from scripts.exp2_supplementary import prepare_distill_weight_ablation_run

        run_dir = self.root / "supplementary" / "S2_D_distill60to30_seed42"
        result = prepare_distill_weight_ablation_run(
            base_config_path=self.base_config,
            base_run_dir=self.base_run_dir,
            run_dir=run_dir,
            checkpoint_path=self.checkpoint,
            variant_id="S2-D",
            distill_loss_lambda=60,
            distill_loss_schedule={"type": "linear_decay", "start_value": 60, "end_value": 30},
            max_train_steps=3000,
            save_model_steps=500,
            seed=42,
        )

        cfg = json.loads(result.config_path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["distill_loss_lambda"], 60)
        self.assertEqual(
            cfg["distill_loss_schedule"],
            {"type": "linear_decay", "start_value": 60.0, "end_value": 30.0, "decay_steps": 3000},
        )
        self.assertEqual(cfg["exp2_supplementary_variant"], "S2-D")

    def test_linear_distill_loss_schedule_resolves_by_step(self):
        from speechtokenizer.trainer.trainer import resolve_distill_loss_lambda

        cfg = {
            "distill_loss_lambda": 60,
            "distill_loss_schedule": {
                "type": "linear_decay",
                "start_value": 60,
                "end_value": 30,
                "decay_steps": 3000,
            },
        }

        self.assertEqual(resolve_distill_loss_lambda(cfg, current_step=0), 60)
        self.assertEqual(resolve_distill_loss_lambda(cfg, current_step=1500), 45)
        self.assertEqual(resolve_distill_loss_lambda(cfg, current_step=3000), 30)
        self.assertEqual(resolve_distill_loss_lambda(cfg, current_step=5000), 30)


if __name__ == "__main__":
    unittest.main()
