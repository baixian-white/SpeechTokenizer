"""LCA fine-tuning entry point.

Loads an Exp2 SCIT-Speech-Base packaged checkpoint as initialization and runs
end-to-end LCA fine-tuning with random L sampling and ChannelSim perturbations.

Mirrors the structure of scripts/train_example.py: parses --config and
--continue_train, builds generator + discriminators, hands them to LCATrainer.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speechtokenizer import SpeechTokenizer
from speechtokenizer.trainer.lca_trainer import LCATrainer
from speechtokenizer.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="LCA fine-tuning config (JSON)")
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=None,
        help="Path to Exp2 SCIT-Speech-Base packaged checkpoint (overrides cfg['lca_base_checkpoint'])",
    )
    parser.add_argument(
        "--continue_train",
        action="store_true",
        help="Continue from latest SpeechTokenizerTrainer_* in results_folder; ignores --base-checkpoint",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = json.load(f)

    if cfg.get("nas_encoder_config"):
        from nas.encoder_only_model_variant import NASEncoderOnlySpeechTokenizer
        generator = NASEncoderOnlySpeechTokenizer(cfg, cfg["nas_encoder_config"])
    else:
        generator = SpeechTokenizer(cfg)

    discriminators = {
        "mpd": MultiPeriodDiscriminator(),
        "msd": MultiScaleDiscriminator(),
        "mstftd": MultiScaleSTFTDiscriminator(32),
    }

    accelerate_kwargs = {
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 1),
        "mixed_precision": cfg.get("mixed_precision", "no"),
        "cpu": cfg.get("device", "cuda") == "cpu",
    }

    trainer = LCATrainer(
        generator=generator,
        discriminators=discriminators,
        cfg=cfg,
        accelerate_kwargs=accelerate_kwargs,
    )

    if args.continue_train:
        trainer.continue_train()
        return

    base_ckpt = args.base_checkpoint or cfg.get("lca_base_checkpoint")
    if not base_ckpt:
        raise SystemExit(
            "LCA fresh start requires --base-checkpoint or cfg['lca_base_checkpoint']. "
            "Use --continue_train to resume from the latest trainer pkg without reloading base."
        )
    trainer.load_base_checkpoint(base_ckpt)
    trainer.train()


if __name__ == "__main__":
    main()
