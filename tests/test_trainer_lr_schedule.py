import math
import unittest

import torch
from accelerate import Accelerator

from speechtokenizer.trainer.trainer import (
    align_cosine_scheduler_to_training_plan,
    compute_lr_scheduler_plan,
)


class TrainerLrScheduleTests(unittest.TestCase):
    def test_scheduler_horizon_matches_batch_steps_used_by_current_training_loop(self):
        plan = compute_lr_scheduler_plan(
            dataset_size=27039,
            batch_size=8,
            epochs=60,
            gradient_accumulation_steps=4,
            drop_last=True,
        )

        self.assertEqual(plan["batches_per_epoch"], 3379)
        self.assertEqual(plan["legacy_update_steps_total"], 50640)
        self.assertEqual(plan["scheduler_total_steps"], 202740)
        self.assertEqual(plan["scheduler_step_unit"], "batch_step")

    def test_max_train_steps_caps_scheduler_horizon_for_debug_runs(self):
        plan = compute_lr_scheduler_plan(
            dataset_size=1000,
            batch_size=8,
            epochs=60,
            gradient_accumulation_steps=4,
            drop_last=True,
            max_train_steps=2,
        )

        self.assertEqual(plan["scheduler_total_steps"], 2)
        self.assertEqual(plan["batch_steps_total"], 2)

    def test_loaded_legacy_cosine_scheduler_is_rebased_to_new_horizon(self):
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([param], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50640)

        legacy_lr = 4.5e-6
        optimizer.param_groups[0]["lr"] = legacy_lr
        scheduler.last_epoch = 57500
        scheduler.T_max = 50640
        scheduler._last_lr = [legacy_lr]

        align_cosine_scheduler_to_training_plan(
            scheduler=scheduler,
            total_steps=202740,
            current_step=57500,
        )

        expected_lr = 0.5e-4 * (1.0 + math.cos(math.pi * 57500 / 202740))
        self.assertEqual(scheduler.T_max, 202740)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected_lr, places=12)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], expected_lr, places=12)

    def test_loaded_accelerated_scheduler_is_rebased_to_new_horizon(self):
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([param], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50640)
        accelerator = Accelerator(cpu=True)
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

        legacy_lr = 4.5e-6
        optimizer.param_groups[0]["lr"] = legacy_lr
        scheduler.scheduler.last_epoch = 57500
        scheduler.scheduler.T_max = 50640
        scheduler.scheduler._last_lr = [legacy_lr]

        align_cosine_scheduler_to_training_plan(
            scheduler=scheduler,
            total_steps=202740,
            current_step=57500,
        )

        expected_lr = 0.5e-4 * (1.0 + math.cos(math.pi * 57500 / 202740))
        self.assertEqual(scheduler.scheduler.T_max, 202740)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected_lr, places=12)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], expected_lr, places=12)


if __name__ == "__main__":
    unittest.main()
