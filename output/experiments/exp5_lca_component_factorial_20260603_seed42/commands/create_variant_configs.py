import copy
import json
from pathlib import Path


RUN_DIR = Path("output/experiments/exp5_lca_component_factorial_20260603_seed42")
BASE_CONFIG = Path(
    "output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/"
    "configs/lca_finetune_config.json"
)


CHANNEL_STRONG = [
    {"name": "clean", "p_drop": 0.0, "p_sub": 0.0},
    {"name": "dropout-mid", "p_drop": 0.05, "p_sub": 0.0},
    {"name": "dropout-high", "p_drop": 0.10, "p_sub": 0.0},
    {"name": "substitution-mid", "p_drop": 0.0, "p_sub": 0.01},
    {"name": "substitution-high", "p_drop": 0.0, "p_sub": 0.03},
]


VARIANTS = {
    "V0_full_depth_clean_control": {
        "description": "Matched fine-tune control: full-depth only, clean channel, no communication branch weight.",
        "random_l_values": [3],
        "channel_conditions": [{"name": "clean", "p_drop": 0.0, "p_sub": 0.0}],
        "lambda_comm": 0.0,
        "lambda_consistency": 0.0,
    },
    "V1_random_l_only": {
        "description": "random-L enabled, clean channel only, no consistency loss.",
        "random_l_values": [1, 2, 3],
        "channel_conditions": [{"name": "clean", "p_drop": 0.0, "p_sub": 0.0}],
        "lambda_comm": 1.5,
        "lambda_consistency": 0.0,
    },
    "V2_channelsim_only": {
        "description": "ChannelSim enabled at fixed full depth L=3, no random-L, no consistency loss.",
        "random_l_values": [3],
        "channel_conditions": CHANNEL_STRONG,
        "lambda_comm": 1.5,
        "lambda_consistency": 0.0,
    },
    "V3_random_l_channelsim": {
        "description": "random-L + strong ChannelSim enabled, no consistency loss.",
        "random_l_values": [1, 2, 3],
        "channel_conditions": CHANNEL_STRONG,
        "lambda_comm": 1.5,
        "lambda_consistency": 0.0,
    },
}


def main() -> None:
    with BASE_CONFIG.open(encoding="utf-8-sig") as handle:
        base = json.load(handle)

    configs_dir = RUN_DIR / "configs" / "variants"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "commands").mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": RUN_DIR.name,
        "purpose": "LCA component factorial ablation",
        "base_config": str(BASE_CONFIG),
        "base_checkpoint": base["lca_base_checkpoint"],
        "existing_full_lca_reference": (
            "output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42"
        ),
        "variants": {},
    }

    command_lines = [
        "# Run from H:\\H-CODE\\speechtokenizer with conda env speechtokenizer.",
        "# Full LCA reference is already available in exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42.",
        "",
    ]

    for index, (name, spec) in enumerate(VARIANTS.items()):
        cfg = copy.deepcopy(base)
        variant_dir = RUN_DIR / "runs" / name
        cfg["results_folder"] = str(variant_dir / "checkpoints")
        cfg["train_files"] = str(RUN_DIR / "artifacts" / "train_files.txt")
        cfg["valid_files"] = str(RUN_DIR / "artifacts" / "valid_files.txt")
        cfg["nas_encoder_config"] = str(RUN_DIR / "configs" / "best_seanet_config.json")
        cfg["random_l_sampling"] = {
            "values": spec["random_l_values"],
            "strategy": "uniform_random" if len(spec["random_l_values"]) > 1 else "round_robin",
            "seed": 42 + index,
        }
        cfg["channel_sim"] = {
            "sample_strategy": "uniform_random" if len(spec["channel_conditions"]) > 1 else "round_robin",
            "conditions": spec["channel_conditions"],
            "dropout_implementation": "previous-index replacement",
            "substitution": "uniform legal codebook index",
        }
        cfg["lambda_comm"] = spec["lambda_comm"]
        cfg["lambda_consistency"] = spec["lambda_consistency"]
        cfg["exp3_experiment_tag"] = f"lca_factorial_{name}"
        cfg["exp3_experiment_note"] = spec["description"]
        cfg["seed"] = 42 + index

        config_path = configs_dir / f"{name}.json"
        config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

        manifest["variants"][name] = {
            "description": spec["description"],
            "config": str(config_path),
            "results_folder": cfg["results_folder"],
            "random_l_values": spec["random_l_values"],
            "channel_conditions": spec["channel_conditions"],
            "lambda_comm": spec["lambda_comm"],
            "lambda_consistency": spec["lambda_consistency"],
        }
        command_lines.append(
            "& 'C:\\Users\\Windows11\\.conda\\envs\\speechtokenizer\\Scripts\\accelerate.exe' "
            f"launch scripts/train_lca.py --config {config_path} --base-checkpoint {base['lca_base_checkpoint']}"
        )

    (RUN_DIR / "configs" / "ablation_matrix.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RUN_DIR / "commands" / "run_all_variants.ps1").write_text(
        "\n".join(command_lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
