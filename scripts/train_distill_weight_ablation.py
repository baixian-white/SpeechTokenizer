import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speechtokenizer import SpeechTokenizerTrainer
from speechtokenizer.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)
from scripts.experiment_utils import write_json
from scripts.exp2_supplementary import (
    DISTILL_WEIGHT_ABLATION_SCOPE,
    apply_generator_train_scope,
    load_generator_checkpoint,
    read_json,
)
from scripts.package_exp2_outputs import build_model


def build_discriminators():
    return {
        "mpd": MultiPeriodDiscriminator(),
        "msd": MultiScaleDiscriminator(),
        "mstftd": MultiScaleSTFTDiscriminator(32),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Run Exp2 S2 semantic distillation-weight ablation.")
    parser.add_argument("--config", "-c", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = read_json(args.config)
    scope = cfg.get("finetune_scope")
    if scope != DISTILL_WEIGHT_ABLATION_SCOPE:
        raise ValueError(f"Expected finetune_scope={DISTILL_WEIGHT_ABLATION_SCOPE}, got {scope!r}")

    checkpoint = cfg.get("pretrained_generator_checkpoint")
    if not checkpoint:
        raise ValueError("distillation-weight ablation requires pretrained_generator_checkpoint in config")

    generator = build_model(cfg)
    load_generator_checkpoint(generator, checkpoint)
    train_scope_report = apply_generator_train_scope(generator, scope)

    results_folder = Path(cfg["results_folder"])
    write_json(results_folder.parent / "reports" / "train_scope_report.json", train_scope_report)

    discriminators = build_discriminators()
    accelerate_kwargs = {
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 1),
        "mixed_precision": cfg.get("mixed_precision", "no"),
        "cpu": cfg.get("device", "cuda") == "cpu",
    }

    trainer = SpeechTokenizerTrainer(
        generator=generator,
        discriminators=discriminators,
        cfg=cfg,
        accelerate_kwargs=accelerate_kwargs,
    )
    trainer.train()


if __name__ == "__main__":
    main()
