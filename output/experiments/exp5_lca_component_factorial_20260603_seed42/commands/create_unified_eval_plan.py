import json
from pathlib import Path

ROOT = Path("output/experiments/exp5_lca_component_factorial_20260603_seed42")
BASE_CONFIG = Path("output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json")
BASE_CKPT = Path("output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt")
SAMPLE_LIST = ROOT / "artifacts" / "valid_files.txt"
CHANNELS = [
    {"name": "clean", "p_drop": 0.0, "p_sub": 0.0},
    {"name": "dropout-mid", "p_drop": 0.05, "p_sub": 0.0},
    {"name": "dropout-high", "p_drop": 0.10, "p_sub": 0.0},
    {"name": "substitution-mid", "p_drop": 0.0, "p_sub": 0.01},
    {"name": "substitution-high", "p_drop": 0.0, "p_sub": 0.03},
]
VARIANTS = [
    {"id": "V0_clean_control_step30000", "config": ROOT / "configs" / "variants" / "V0_full_depth_clean_control.json", "checkpoint": ROOT / "runs" / "V0_full_depth_clean_control" / "checkpoints" / "SpeechTokenizerTrainer_00030000", "note": "V0 checkpoint near best L3 clean comm-mel."},
    {"id": "V1_random_l_only_step25000", "config": ROOT / "configs" / "variants" / "V1_random_l_only.json", "checkpoint": ROOT / "runs" / "V1_random_l_only" / "checkpoints" / "SpeechTokenizerTrainer_00025000", "note": "V1 checkpoint near best L2/L3 clean comm-mel."},
    {"id": "V2_channelsim_only_step17500", "config": ROOT / "configs" / "variants" / "V2_channelsim_only.json", "checkpoint": ROOT / "runs" / "V2_channelsim_only" / "checkpoints" / "SpeechTokenizerTrainer_00017500", "note": "V2 checkpoint near best L3 perturbation dev-mel."},
    {"id": "V3_random_l_channelsim_step32500", "config": ROOT / "configs" / "variants" / "V3_random_l_channelsim.json", "checkpoint": ROOT / "runs" / "V3_random_l_channelsim" / "checkpoints" / "SpeechTokenizerTrainer_00032500", "note": "V3 final checkpoint; best for most L2/L3 dev metrics."},
    {"id": "V4_full_lca_step30000", "config": Path("output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/configs/lca_finetune_config.json"), "checkpoint": Path("output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt"), "note": "Full LCA reference with random-L + ChannelSim + consistency."},
]

def main():
    eval_cfg_dir = ROOT / "configs" / "eval"
    eval_cfg_dir.mkdir(parents=True, exist_ok=True)
    eval_run_dir = ROOT / "eval_unified_20260605"
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    plan = {"run_id": "eval_unified_20260605", "base_config": str(BASE_CONFIG), "base_checkpoint": str(BASE_CKPT), "sample_list": str(SAMPLE_LIST), "max_samples": 8, "L_values": [1, 2, 3], "channel_conditions": CHANNELS, "variants": []}
    commands = ["$ErrorActionPreference = 'Stop'", "Set-Location 'H:\\H-CODE\\speechtokenizer'", ""]
    for variant in VARIANTS:
        with variant["config"].open(encoding="utf-8-sig") as handle:
            cfg = json.load(handle)
        cfg["random_l_sampling"] = {"values": [1, 2, 3], "strategy": "round_robin", "seed": 42}
        cfg["channel_sim"] = {"sample_strategy": "round_robin", "conditions": CHANNELS, "dropout_implementation": "previous-index replacement", "substitution": "uniform legal codebook index"}
        cfg_path = eval_cfg_dir / f"{variant['id']}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        run_dir = eval_run_dir / variant["id"]
        plan["variants"].append({"id": variant["id"], "note": variant["note"], "eval_config": str(cfg_path), "checkpoint": str(variant["checkpoint"]), "run_dir": str(run_dir)})
        commands.append(f"Write-Host '=== Evaluating {variant['id']} ==='")
        commands.append("& 'C:\\Users\\Windows11\\.conda\\envs\\speechtokenizer\\python.exe' scripts/evaluate_lca_vs_base.py " +
                        f"--run-dir {run_dir} --base-config {BASE_CONFIG} --base-checkpoint {BASE_CKPT} " +
                        f"--lca-config {cfg_path} --lca-checkpoint {variant['checkpoint']} --sample-list {SAMPLE_LIST} " +
                        "--max-samples 8 --device cuda --channel-seed 42")
        commands.append("")
    (ROOT / "configs" / "eval_unified_20260605.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "commands" / "run_unified_eval_20260605.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
if __name__ == "__main__":
    main()
