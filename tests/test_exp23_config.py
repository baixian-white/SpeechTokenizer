import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from scripts.experiment_utils import (
    collect_environment_metadata,
    ensure_run_layout,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "exp23_speaker_classifier.json"


class Exp23ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            cls.config = json.load(config_file)

    def test_primary_task_and_condition_are_frozen(self):
        self.assertEqual(
            self.config["task"]["name"],
            "closed_set_speaker_classification",
        )
        self.assertEqual(self.config["task"]["speaker_count"], 110)
        self.assertEqual(
            self.config["primary_condition"],
            {"model": "lca", "rvq_layers": 3},
        )

    def test_dataset_splits_are_frozen(self):
        canonical = self.config["split"]
        self.assertEqual(canonical["seed"], 42)
        self.assertEqual(
            canonical["counts"],
            {"train": 80, "validation": 15, "test": 15},
        )

        text_controlled = self.config["text_controlled_split"]
        self.assertEqual(text_controlled["seed"], 1)
        self.assertEqual(
            text_controlled["counts"],
            {"train": 80, "validation": 10, "test": 10},
        )
        self.assertEqual(text_controlled["excluded_speakers"], ["p315"])

    def test_model_and_training_defaults_are_frozen(self):
        token_branch = self.config["model"]["token_branch"]
        self.assertEqual(token_branch["pad_index"], 1024)
        self.assertEqual(token_branch["token_embedding_dim"], 128)
        self.assertEqual(token_branch["model_dim"], 256)
        self.assertEqual(token_branch["speaker_embedding_dim"], 256)
        self.assertEqual(token_branch["block_count"], 4)

        aam = self.config["model"]["classification_head"]["aam_softmax"]
        self.assertEqual(aam, {"scale": 30, "margin": 0.2})

        training = self.config["training"]
        self.assertEqual(training["seeds"], [41, 42, 43])
        self.assertEqual(training["crop_seconds"], 3)
        self.assertEqual(training["effective_batch_size"], 128)
        self.assertEqual(training["early_stopping_patience"], 6)

    def test_cache_and_stage_gates_are_frozen(self):
        self.assertEqual(
            self.config["cache"],
            {"budget_gb": 30, "free_space_multiplier": 1.2},
        )

        stage_gates = self.config["stage_gates"]
        self.assertEqual(stage_gates["metric_split"], "validation")
        self.assertEqual(stage_gates["replication_validation_top1"], 0.88)
        self.assertNotIn("test_threshold", stage_gates)

    def test_frozen_upstream_paths_match_design(self):
        self.assertEqual(
            self.config["frozen_upstream"]["base"],
            {
                "config": "output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json",
                "checkpoint": "output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt",
            },
        )
        self.assertEqual(
            self.config["frozen_upstream"]["lca"],
            {
                "config": "output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/configs/lca_finetune_config.json",
                "checkpoint": "output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt",
            },
        )


class ExperimentProvenanceTest(unittest.TestCase):
    def test_standard_run_layout_includes_cache_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = ensure_run_layout(Path(temp_dir) / "run")
            self.assertTrue((run_dir / "cache_manifest").is_dir())

    @staticmethod
    def command_result(returncode=0, stdout="", stderr=""):
        return {
            "args": [],
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    @patch("scripts.experiment_utils.importlib.metadata.version")
    @patch("scripts.experiment_utils.run_command")
    def test_environment_metadata_records_clean_git_and_package_versions(
        self,
        mock_run_command,
        mock_version,
    ):
        mock_run_command.side_effect = [
            self.command_result(stdout="abc123"),
            self.command_result(stdout=""),
        ]
        mock_version.side_effect = lambda distribution: f"{distribution}-version"

        metadata = collect_environment_metadata(PROJECT_ROOT)
        json.dumps(metadata)

        self.assertEqual(
            metadata["git"],
            {"revision": "abc123", "dirty": False, "error": None},
        )
        self.assertEqual(
            metadata["packages"]["torch"],
            {"version": "torch-version", "error": None},
        )
        self.assertEqual(
            metadata["packages"]["sklearn"],
            {"version": "scikit-learn-version", "error": None},
        )
        self.assertEqual(set(metadata["python"]), {"version"})
        self.assertIsInstance(metadata["platform"], str)
        mock_run_command.assert_has_calls(
            [
                call(
                    ["git", "rev-parse", "HEAD"],
                    cwd=PROJECT_ROOT.resolve(),
                    timeout=20,
                ),
                call(
                    ["git", "status", "--porcelain"],
                    cwd=PROJECT_ROOT.resolve(),
                    timeout=20,
                ),
            ]
        )

    @patch("scripts.experiment_utils.importlib.metadata.version", return_value="1.0")
    @patch("scripts.experiment_utils.run_command")
    def test_environment_metadata_records_dirty_git(
        self,
        mock_run_command,
        _mock_version,
    ):
        mock_run_command.side_effect = [
            self.command_result(stdout="abc123"),
            self.command_result(stdout=" M private_file.py"),
        ]

        metadata = collect_environment_metadata(PROJECT_ROOT)

        self.assertEqual(
            metadata["git"],
            {"revision": "abc123", "dirty": True, "error": None},
        )

    @patch("scripts.experiment_utils.importlib.metadata.version", return_value="1.0")
    @patch("scripts.experiment_utils.run_command")
    def test_environment_metadata_sanitizes_git_failure(
        self,
        mock_run_command,
        _mock_version,
    ):
        mock_run_command.return_value = self.command_result(
            returncode=128,
            stderr="fatal: C:/Users/private-user/secret-repo is unavailable",
        )

        metadata = collect_environment_metadata(PROJECT_ROOT)
        serialized = json.dumps(metadata)

        self.assertEqual(
            metadata["git"],
            {
                "revision": None,
                "dirty": None,
                "error": "git rev-parse HEAD failed (returncode=128)",
            },
        )
        self.assertNotIn("private-user", serialized)
        self.assertNotIn("secret-repo", serialized)

    @patch("scripts.experiment_utils.importlib.metadata.version", return_value="1.0")
    @patch("scripts.experiment_utils.run_command")
    def test_environment_metadata_sanitizes_git_status_failure(
        self,
        mock_run_command,
        _mock_version,
    ):
        mock_run_command.side_effect = [
            self.command_result(stdout="abc123"),
            self.command_result(
                returncode=None,
                stderr="PermissionError at C:/Users/private-user/secret-repo",
            ),
        ]

        metadata = collect_environment_metadata(PROJECT_ROOT)
        serialized = json.dumps(metadata)

        self.assertEqual(
            metadata["git"],
            {
                "revision": "abc123",
                "dirty": None,
                "error": "git status --porcelain failed (returncode=unavailable)",
            },
        )
        self.assertNotIn("private-user", serialized)
        self.assertNotIn("secret-repo", serialized)

    @patch("scripts.experiment_utils.importlib.metadata.version")
    @patch("scripts.experiment_utils.run_command")
    def test_environment_metadata_uses_stable_package_error_structure(
        self,
        mock_run_command,
        mock_version,
    ):
        mock_run_command.side_effect = [
            self.command_result(stdout="abc123"),
            self.command_result(stdout=""),
        ]

        def package_version(distribution):
            if distribution == "speechbrain":
                raise importlib.metadata.PackageNotFoundError(
                    "speechbrain at C:/Users/private-user/env"
                )
            return "1.0"

        mock_version.side_effect = package_version

        metadata = collect_environment_metadata(PROJECT_ROOT)
        serialized = json.dumps(metadata)

        self.assertEqual(
            metadata["packages"]["speechbrain"],
            {"version": None, "error": "PackageNotFoundError"},
        )
        for package_metadata in metadata["packages"].values():
            self.assertEqual(set(package_metadata), {"version", "error"})
        self.assertNotIn("private-user", serialized)


if __name__ == "__main__":
    unittest.main()
