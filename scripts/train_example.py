# scripts/train_example.py
from speechtokenizer import SpeechTokenizer, SpeechTokenizerTrainer
from speechtokenizer.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)
import json
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file path')
    parser.add_argument('--continue_train', action='store_true', help='Continue to train from checkpoints')
    args = parser.parse_args()

    # 读取配置文件
    with open(args.config) as f:
        cfg = json.load(f)

    # 构造生成器与判别器
    generator = SpeechTokenizer(cfg)
    discriminators = {
        'mpd': MultiPeriodDiscriminator(),
        'msd': MultiScaleDiscriminator(),
        'mstftd': MultiScaleSTFTDiscriminator(32),
    }

    # ========= ⭐ 关键新增：透传 Accelerate 的关键参数 =========
    # - gradient_accumulation_steps: 用于梯度累积，模拟大 batch
    # - mixed_precision（可选）   : 'no' / 'fp16' / 'bf16'，若未在 cfg 指定，默认不启用
    accelerate_kwargs = {
        'gradient_accumulation_steps': cfg.get('gradient_accumulation_steps', 1),
        'mixed_precision': cfg.get('mixed_precision', 'no'),
        # 你也可以按需加更多 Accelerate 参数，例如：
        # 'project_dir': cfg.get('results_folder', 'Log/spt_base'),
    }
    # =====================================================

    # 初始化 Trainer，并把 accelerate_kwargs 传进去
    trainer = SpeechTokenizerTrainer(
        generator=generator,
        discriminators=discriminators,
        cfg=cfg,
        accelerate_kwargs=accelerate_kwargs,   # ⭐ 关键：让梯度累积真正生效
    )

    # 继续训练或从头训练
    if args.continue_train:
        trainer.continue_train()
    else:
        trainer.train()
