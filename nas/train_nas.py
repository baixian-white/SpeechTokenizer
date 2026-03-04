# train_nas.py
"""
脚本名称: train_nas.py
功能描述:
    该脚本用于训练(Retrain)经过 NAS (神经网络架构搜索) 搜索出的最优 SpeechTokenizer 模型结构。
"""

from pathlib import Path
import sys
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAS_DIR = Path(__file__).resolve().parent
DEFAULT_NAS_CONFIG = NAS_DIR / "best_seanet_config.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speechtokenizer import SpeechTokenizerTrainer
from speechtokenizer.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)

try:
    from .custom_model import NASSpeechTokenizer
except ImportError:
    from custom_model import NASSpeechTokenizer


def _resolve_path(path_str: str, prefer_nas_dir: bool = False) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p

    candidates = []
    if prefer_nas_dir:
        candidates.append((NAS_DIR / p).resolve())
    candidates.append((PROJECT_ROOT / p).resolve())
    candidates.append((Path.cwd() / p).resolve())

    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Config file path")
    parser.add_argument("--continue_train", action="store_true", help="Continue to train from checkpoints")
    parser.add_argument(
        "--nas_config",
        type=str,
        default=str(DEFAULT_NAS_CONFIG),
        help="Path to NAS best structure json",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    nas_config_path = _resolve_path(args.nas_config, prefer_nas_dir=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"🚀 Initializing NAS Model using {nas_config_path}...")
    if not nas_config_path.exists():
        raise FileNotFoundError(f"❌ 找不到 NAS 配置文件: {nas_config_path}，请先运行 nas/export_best_model.py")

    generator = NASSpeechTokenizer(cfg, nas_config_path=nas_config_path)

    discriminators = {
        "mpd": MultiPeriodDiscriminator(),
        "msd": MultiScaleDiscriminator(),
        "mstftd": MultiScaleSTFTDiscriminator(32),
    }

    accelerate_kwargs = {
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 1),
        "mixed_precision": cfg.get("mixed_precision", "no"),
    }

    print("⚙️ Initializing Trainer...")
    trainer = SpeechTokenizerTrainer(
        generator=generator,
        discriminators=discriminators,
        cfg=cfg,
        accelerate_kwargs=accelerate_kwargs,
    )

    print("🔥 Start Training NAS Model...")
    if args.continue_train:
        trainer.continue_train()
    else:
        trainer.train()
