import unittest
import io
import sys
from pathlib import Path

import torch


class TeacherGuidedProxyMetricTests(unittest.TestCase):
    def test_alignment_and_rvq_metrics_are_reported(self):
        from nas.teacher_guided_proxy import compute_teacher_guided_metrics

        teacher_latent = torch.tensor(
            [
                [
                    [1.0, 2.0, 4.0],
                    [2.0, 3.0, 5.0],
                ]
            ]
        )
        student_latent = teacher_latent.clone()
        teacher_codes = torch.tensor(
            [
                [[1, 2, 3]],
                [[4, 5, 6]],
                [[7, 8, 9]],
            ]
        )
        student_codes = teacher_codes.clone()
        student_codes[1, 0, 2] = 42
        teacher_quantized = torch.zeros(1, 2, 3)
        student_quantized = torch.ones(1, 2, 3)

        metrics = compute_teacher_guided_metrics(
            teacher_latent=teacher_latent,
            student_latent=student_latent,
            teacher_codes=teacher_codes,
            student_codes=student_codes,
            teacher_quantized=teacher_quantized,
            student_quantized=student_quantized,
        )

        self.assertEqual(metrics["teacher_latent_smooth_l1"], 0.0)
        self.assertAlmostEqual(metrics["teacher_latent_cosine_distance"], 0.0, places=6)
        self.assertEqual(metrics["teacher_temporal_delta_loss"], 0.0)
        self.assertAlmostEqual(metrics["rvq_code_agreement"], 8 / 9)
        self.assertAlmostEqual(metrics["rvq_code_flip_rate"], 1 / 9)
        self.assertEqual(metrics["rvq_quantized_feature_l1"], 1.0)


class StagedTeacherArgumentTests(unittest.TestCase):
    def test_staged_runner_accepts_teacher_guided_arguments(self):
        from nas.run_staged_encoder_nas import build_parser

        args = build_parser().parse_args(
            [
                "--run-dir",
                "run",
                "--config",
                "config.json",
                "--manifest",
                "train.txt",
                "--teacher-config",
                "teacher_config.json",
                "--teacher-checkpoint",
                "teacher.pt",
                "--teacher-target",
                "pre_rvq",
                "--teacher-cache-mode",
                "disk",
                "--distill-stage2-steps",
                "10",
                "--distill-stage3-steps",
                "20",
                "--distill-final-steps",
                "30",
                "--distill-lr",
                "0.0002",
            ]
        )

        self.assertEqual(args.teacher_config, "teacher_config.json")
        self.assertEqual(args.teacher_checkpoint, "teacher.pt")
        self.assertEqual(args.teacher_target, "pre_rvq")
        self.assertEqual(args.teacher_cache_mode, "disk")
        self.assertEqual(args.distill_stage2_steps, 10)
        self.assertEqual(args.distill_stage3_steps, 20)
        self.assertEqual(args.distill_final_steps, 30)
        self.assertEqual(args.distill_lr, 0.0002)


class ExperimentTeeLoggingTests(unittest.TestCase):
    def test_tee_logging_preserves_terminal_streams_and_writes_log_files(self):
        from scripts.experiment_utils import install_tee_logging

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        console_stdout = io.StringIO()
        console_stderr = io.StringIO()

        run_dir = Path.cwd() / "tmp" / f"tee_logging_test_{id(self)}"
        try:
            sys.stdout = console_stdout
            sys.stderr = console_stderr
            install_tee_logging(run_dir)

            print("stdout-visible")
            print("stderr-visible", file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()

            self.assertIn("stdout-visible", console_stdout.getvalue())
            self.assertIn("stderr-visible", console_stderr.getvalue())
            self.assertIn("stdout-visible", (run_dir / "logs" / "stdout.log").read_text(encoding="utf-8"))
            self.assertIn("stderr-visible", (run_dir / "logs" / "stderr.log").read_text(encoding="utf-8"))
        finally:
            tee_stdout = sys.stdout
            tee_stderr = sys.stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            for stream in (tee_stdout, tee_stderr):
                close_log = getattr(stream, "close_log", None)
                if close_log:
                    close_log()


class ShortDistillationValidationTests(unittest.TestCase):
    def test_distillation_losses_are_zero_for_matching_latents(self):
        from nas.validate_short_distill import compute_distillation_losses

        teacher = torch.randn(2, 1024, 5)
        losses = compute_distillation_losses(student_latent=teacher.clone(), teacher_latent=teacher)

        self.assertAlmostEqual(float(losses["latent_smooth_l1"].item()), 0.0, places=6)
        self.assertAlmostEqual(float(losses["cosine_distance"].item()), 0.0, places=6)
        self.assertAlmostEqual(float(losses["temporal_delta_loss"].item()), 0.0, places=6)
        self.assertAlmostEqual(float(losses["total"].item()), 0.0, places=6)

    def test_short_distill_parser_accepts_validation_arguments(self):
        from nas.validate_short_distill import build_parser

        args = build_parser().parse_args(
            [
                "--source-run-dir",
                "output/experiments/source",
                "--run-dir",
                "output/experiments/distill",
                "--config",
                "config.json",
                "--manifest",
                "train.txt",
                "--teacher-config",
                "teacher.json",
                "--teacher-checkpoint",
                "teacher.pt",
                "--candidate-limit",
                "2",
                "--train-steps",
                "5",
                "--eval-samples",
                "3",
            ]
        )

        self.assertEqual(args.source_run_dir, "output/experiments/source")
        self.assertEqual(args.candidate_limit, 2)
        self.assertEqual(args.train_steps, 5)
        self.assertEqual(args.eval_samples, 3)


if __name__ == "__main__":
    unittest.main()
