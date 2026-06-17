import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from scripts.experiment_utils import ensure_run_layout, write_json, write_text
except ImportError:
    from experiment_utils import ensure_run_layout, write_json, write_text


DECODER_ONLY_SCOPE = "decoder_only_acoustic"
DISTILL_WEIGHT_ABLATION_SCOPE = "distill_weight_ablation"
FULL_TRAINABLE_MODULES = ["encoder", "quantizer", "transform", "decoder"]


@dataclass
class SupplementaryRunPreparation:
    run_dir: Path
    config_path: Path
    command_path: Path
    launch_guide_path: Path


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _count_params(parameters):
    return sum(int(p.numel()) for p in parameters)


def _set_module_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = bool(trainable)


def apply_generator_train_scope(generator, scope):
    """Apply the generator freeze policy for an Exp2 supplementary run."""
    if scope not in (DECODER_ONLY_SCOPE, DISTILL_WEIGHT_ABLATION_SCOPE):
        raise ValueError(f"Unsupported generator train scope: {scope}")

    if scope == DECODER_ONLY_SCOPE:
        for param in generator.parameters():
            param.requires_grad = False

        if not hasattr(generator, "decoder"):
            raise AttributeError("decoder_only_acoustic requires generator.decoder")
        _set_module_trainable(generator.decoder, True)
    else:
        for param in generator.parameters():
            param.requires_grad = True

    module_status = {}
    for name in ("encoder", "quantizer", "transform", "decoder"):
        module = getattr(generator, name, None)
        if module is None:
            continue
        params = list(module.parameters())
        module_status[name] = {
            "param_count": _count_params(params),
            "trainable_param_count": _count_params(p for p in params if p.requires_grad),
            "trainable": any(p.requires_grad for p in params),
        }

    trainable_params = [p for p in generator.parameters() if p.requires_grad]
    frozen_params = [p for p in generator.parameters() if not p.requires_grad]
    return {
        "scope": scope,
        "trainable_modules": [name for name, item in module_status.items() if item["trainable"]],
        "frozen_modules": [name for name, item in module_status.items() if not item["trainable"]],
        "module_status": module_status,
        "trainable_param_count": _count_params(trainable_params),
        "frozen_param_count": _count_params(frozen_params),
    }


def _copy_nas_config_if_available(cfg, run_dir):
    nas_path = cfg.get("nas_encoder_config")
    if not nas_path:
        return cfg

    source = Path(nas_path)
    if not source.exists():
        return cfg

    target = run_dir / "configs" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    cfg["nas_encoder_config"] = str(target)
    return cfg


def _launch_command(config_path):
    return f"accelerate launch scripts/train_decoder_only_finetune.py --config {config_path}"


def _s2_launch_command(config_path):
    return f"accelerate launch scripts/train_distill_weight_ablation.py --config {config_path}"


def _launch_guide_text(run_dir, config_path, command_path):
    return "\n".join(
        [
            "# S1 decoder-only acoustic finetune 启动指南",
            "",
            "## 目的",
            "",
            "从实验二 best-dev checkpoint 出发，冻结 encoder 与 RVQ/codebook，只训练 decoder 和判别器，验证听感质量是否能在不改变 token 表示的前提下提升。",
            "",
            "## 启动命令",
            "",
            "```powershell",
            command_path.read_text(encoding="utf-8").strip(),
            "```",
            "",
            "## 关键文件",
            "",
            f"- 运行目录：`{run_dir}`",
            f"- 配置文件：`{config_path}`",
            f"- 命令文件：`{command_path}`",
            f"- 训练日志：`{run_dir / 'logs'}`",
            f"- checkpoint 输出：`{run_dir / 'checkpoints'}`",
            "",
            "## 训练完成后的建议评估",
            "",
            "1. 选取 finetune run 中的 best-dev checkpoint 或最终 checkpoint。",
            "2. 复用实验二固定样本与完整句子样本，生成 L1/L2/L3 重建语音。",
            "3. 运行样本级 PESQ-WB、STOI、SI-SNR、Mel L1 评估。",
            "4. 将结果追加到 `exp2_supplementary_experiments.md` 的“后续追加记录”小节。",
            "",
        ]
    )


def _s2_launch_guide_text(run_dir, config_path, command_path, variant_id, distill_note):
    return "\n".join(
        [
            f"# {variant_id} 语义蒸馏权重 ablation 启动指南",
            "",
            "## 目的",
            "",
            "从实验二 best-dev checkpoint 出发，调整 `distill_loss_lambda`，验证语义蒸馏权重是否过度挤压声学重建质量。",
            "",
            "## 当前变体",
            "",
            f"- 变体：`{variant_id}`",
            f"- 蒸馏权重：{distill_note}",
            "- 训练范围：generator 全量可训练，判别器正常训练。",
            "- 对照对象：实验二 best-dev 基线样本指标。",
            "",
            "## 启动命令",
            "",
            "```powershell",
            command_path.read_text(encoding="utf-8").strip(),
            "```",
            "",
            "## 关键文件",
            "",
            f"- 运行目录：`{run_dir}`",
            f"- 配置文件：`{config_path}`",
            f"- 命令文件：`{command_path}`",
            f"- 训练日志：`{run_dir / 'logs'}`",
            f"- checkpoint 输出：`{run_dir / 'checkpoints'}`",
            "",
            "## 训练完成后的评估",
            "",
            "1. 使用本 run 的 `checkpoints/SpeechTokenizer_best_dev.pt` 或最后一个 checkpoint 导出 fixed/full_utterance 样本。",
            "2. 运行 `scripts/evaluate_sample_audio_quality.py`，与实验二 best-dev 基线做同口径对比。",
            "3. 优先判定 L3 的 PESQ-WB、STOI、SI-SNR 与 Mel L1。",
            "4. 将结果追加到 `exp2_supplementary_experiments.md`。",
            "",
        ]
    )


def prepare_decoder_only_acoustic_run(
    base_config_path,
    base_run_dir,
    run_dir,
    checkpoint_path,
    learning_rate=3e-6,
    distill_loss_lambda=0,
    epochs=3,
    max_train_steps=None,
    seed=42,
):
    base_config_path = Path(base_config_path)
    base_run_dir = Path(base_run_dir)
    run_dir = ensure_run_layout(run_dir)
    checkpoint_path = Path(checkpoint_path)

    cfg = read_json(base_config_path)
    cfg = dict(cfg)
    cfg["results_folder"] = str(run_dir / "checkpoints")
    cfg["pretrained_generator_checkpoint"] = str(checkpoint_path)
    cfg["finetune_scope"] = DECODER_ONLY_SCOPE
    cfg["frozen_modules"] = ["encoder", "quantizer", "transform"]
    cfg["trainable_modules"] = ["decoder"]
    cfg["learning_rate"] = float(learning_rate)
    cfg["intial_learning_rate"] = float(learning_rate)
    cfg["distill_loss_lambda"] = float(distill_loss_lambda)
    cfg["epochs"] = int(epochs)
    cfg["seed"] = int(seed)
    cfg["valid_num_workers"] = int(cfg.get("valid_num_workers", 0))
    cfg["exp2_supplementary_parent_run"] = str(base_run_dir)
    cfg["exp2_supplementary_note"] = (
        "Decoder-only acoustic finetune: encoder/RVQ/codebooks are frozen; decoder is trained from the Exp2 best-dev checkpoint."
    )
    if max_train_steps is not None:
        cfg["max_train_steps"] = int(max_train_steps)
        short_run_save_steps = max(1, int(max_train_steps) // 2)
        cfg["save_model_steps"] = min(int(cfg.get("save_model_steps", short_run_save_steps)), short_run_save_steps)

    cfg = _copy_nas_config_if_available(cfg, run_dir)

    config_path = run_dir / "configs" / "decoder_only_acoustic_config.json"
    write_json(config_path, cfg)

    command_path = run_dir / "commands" / "run_decoder_only_acoustic_finetune.txt"
    write_text(command_path, _launch_command(config_path) + "\n")

    launch_guide_path = run_dir / "reports" / "launch_guide.md"
    write_text(launch_guide_path, _launch_guide_text(run_dir, config_path, command_path))

    write_json(
        run_dir / "reports" / "preparation.json",
        {
            "run_dir": str(run_dir),
            "base_config": str(base_config_path),
            "base_run_dir": str(base_run_dir),
            "checkpoint": str(checkpoint_path),
            "config_path": str(config_path),
            "command_path": str(command_path),
            "scope": DECODER_ONLY_SCOPE,
        },
    )

    return SupplementaryRunPreparation(
        run_dir=run_dir,
        config_path=config_path,
        command_path=command_path,
        launch_guide_path=launch_guide_path,
    )


def _normalize_distill_schedule(schedule, max_train_steps):
    if not schedule:
        return None
    schedule = dict(schedule)
    schedule_type = schedule.get("type")
    if schedule_type != "linear_decay":
        raise ValueError(f"Unsupported distill_loss_schedule type: {schedule_type!r}")
    if "start_value" not in schedule or "end_value" not in schedule:
        raise ValueError("linear_decay distill_loss_schedule requires start_value and end_value")
    decay_steps = schedule.get("decay_steps", max_train_steps)
    if decay_steps is None:
        raise ValueError("linear_decay distill_loss_schedule requires decay_steps or max_train_steps")
    return {
        "type": "linear_decay",
        "start_value": float(schedule["start_value"]),
        "end_value": float(schedule["end_value"]),
        "decay_steps": int(decay_steps),
    }


def prepare_distill_weight_ablation_run(
    base_config_path,
    base_run_dir,
    run_dir,
    checkpoint_path,
    variant_id,
    distill_loss_lambda,
    learning_rate=1e-5,
    epochs=3,
    max_train_steps=3000,
    save_model_steps=500,
    seed=42,
    distill_loss_schedule=None,
):
    base_config_path = Path(base_config_path)
    base_run_dir = Path(base_run_dir)
    run_dir = ensure_run_layout(run_dir)
    checkpoint_path = Path(checkpoint_path)

    cfg = read_json(base_config_path)
    cfg = dict(cfg)
    cfg["results_folder"] = str(run_dir / "checkpoints")
    cfg["pretrained_generator_checkpoint"] = str(checkpoint_path)
    cfg["finetune_scope"] = DISTILL_WEIGHT_ABLATION_SCOPE
    cfg["exp2_supplementary_variant"] = str(variant_id)
    cfg["frozen_modules"] = []
    cfg["trainable_modules"] = list(FULL_TRAINABLE_MODULES)
    cfg["learning_rate"] = float(learning_rate)
    cfg["intial_learning_rate"] = float(learning_rate)
    cfg["distill_loss_lambda"] = float(distill_loss_lambda)
    cfg["epochs"] = int(epochs)
    cfg["seed"] = int(seed)
    cfg["valid_num_workers"] = int(cfg.get("valid_num_workers", 0))
    cfg["max_train_steps"] = int(max_train_steps)
    cfg["save_model_steps"] = min(int(save_model_steps), int(max_train_steps))
    cfg["exp2_supplementary_parent_run"] = str(base_run_dir)
    cfg["exp2_supplementary_note"] = (
        "Distillation-weight ablation: start from Exp2 best-dev; train the full generator and discriminators with a lower or scheduled semantic distillation weight."
    )

    normalized_schedule = _normalize_distill_schedule(distill_loss_schedule, cfg["max_train_steps"])
    if normalized_schedule:
        cfg["distill_loss_schedule"] = normalized_schedule
        distill_note = (
            f"`{normalized_schedule['type']}`: "
            f"{normalized_schedule['start_value']} -> {normalized_schedule['end_value']} "
            f"over {normalized_schedule['decay_steps']} steps"
        )
    else:
        cfg.pop("distill_loss_schedule", None)
        distill_note = f"`distill_loss_lambda={cfg['distill_loss_lambda']}`"

    cfg = _copy_nas_config_if_available(cfg, run_dir)

    config_path = run_dir / "configs" / "distill_weight_ablation_config.json"
    write_json(config_path, cfg)

    command_path = run_dir / "commands" / "run_distill_weight_ablation.txt"
    write_text(command_path, _s2_launch_command(config_path) + "\n")

    launch_guide_path = run_dir / "reports" / "launch_guide.md"
    write_text(launch_guide_path, _s2_launch_guide_text(run_dir, config_path, command_path, variant_id, distill_note))

    write_json(
        run_dir / "reports" / "preparation.json",
        {
            "run_dir": str(run_dir),
            "base_config": str(base_config_path),
            "base_run_dir": str(base_run_dir),
            "checkpoint": str(checkpoint_path),
            "config_path": str(config_path),
            "command_path": str(command_path),
            "scope": DISTILL_WEIGHT_ABLATION_SCOPE,
            "variant_id": str(variant_id),
            "distill_loss_lambda": float(distill_loss_lambda),
            "distill_loss_schedule": normalized_schedule,
        },
    )

    return SupplementaryRunPreparation(
        run_dir=run_dir,
        config_path=config_path,
        command_path=command_path,
        launch_guide_path=launch_guide_path,
    )


def load_generator_checkpoint(generator, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    generator.load_state_dict(state)
    return generator
