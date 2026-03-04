from transformers import HubertModel, Wav2Vec2FeatureExtractor
from pathlib import Path
import torchaudio
import torch
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
import os

if __name__ == '__main__':
    # ========= 1) 解析命令行参数 =========
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file path')
    parser.add_argument('--audio_dir', type=str, help='Audio folder path')
    parser.add_argument('--rep_dir', type=str, help='Path to save representation files')
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='flac')
    parser.add_argument('--split_seed', type=int, help="Random seed", default=0)
    parser.add_argument('--valid_set_size', type=float, default=1000)  # 可为绝对数量或比例(<1)
    args = parser.parse_args()

    # 支持多种音频后缀，通过逗号拆分，例如 "flac,wav,mp3"
    exts = args.exts.split(',')

    # ========= 2) 设备选择（优先GPU） =========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========= 3) 读取配置文件（JSON） =========
    # 期望包含：sample_rate、semantic_model_path、semantic_model_layer、
    #           train_files（训练清单路径）、valid_files（验证清单路径）、segment_size 等键
    with open(args.config) as f:
        cfg = json.load(f)

    sample_rate = cfg.get('sample_rate')
    # 加载 HuBERT 前端：特征提取器（Wav2Vec2FeatureExtractor）与模型（HubertModel）
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.get('semantic_model_path'))
    model = HubertModel.from_pretrained(cfg.get('semantic_model_path')).eval().to(device)
    # 目标层（可为具体整数层索引，也可为字符串 'avg' 表示对所有隐藏层取均值）
    target_layer = cfg.get('semantic_model_layer') #semantic_model_layer = avg

    # ========= 4) 构建音频文件列表 =========
    path = Path(args.audio_dir)
    # 递归收集指定后缀的文件
    file_list = [str(file) for ext in exts for file in path.glob(f'**/*.{ext}')]

    # ========= 5) 计算验证集大小 =========
    # 如果 0 < valid_set_size < 1，则按比例；否则按绝对数量（含 0 表示不切分）
    if args.valid_set_size != 0 and args.valid_set_size < 1:
        valid_set_size = int(len(file_list) * args.valid_set_size)
    else:
        valid_set_size = int(args.valid_set_size)

    # 训练/验证清单文件路径（由配置给出）
    train_file_list = cfg.get('train_files')
    valid_file_list = cfg.get('valid_files')

    # segment_size：如果原始音频太短，不足该长度则右侧补零到该长度
    segment_size = cfg.get('segment_size')

    # ========= 6) 随机打乱文件列表并确定划分 =========
    random.seed(args.split_seed)
    random.shuffle(file_list)

    print(
        f'A total of {len(file_list)} samples will be processed, '
        f'and {valid_set_size} of them will be included in the validation set.'
    )

    # ========= 7) 遍历音频并导出 HuBERT 表征 =========
    # 说明：
    # - 对每个音频：
    #   a) 读入并必要时重采样到 sample_rate
    #   b) 若长度不足 segment_size，则补零到该长度
    #   c) 送入 feature_extractor 与 HubertModel，取指定层的 hidden_states
    #   d) 将该隐藏表示保存为 .hubert.npy
    #   e) 按顺序写入 valid 或 train 清单（前 valid_set_size 个作为验证）
    with torch.no_grad():
        for i, audio_file in tqdm(enumerate(file_list), total=len(file_list)):
            # ---- a) 读入音频 ----
            wav, sr = torchaudio.load(audio_file)  # wav: [channels, time]
            # 如果是多通道，这里后续会 squeeze(0)，假设单通道，若存在多通道可考虑取均值
            # wav = wav.mean(dim=0, keepdim=True)  # 若需要显式混为单通道可取消注释

            # ---- b) 重采样到配置的 sample_rate ----
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)

            # ---- c) 右侧补零到 segment_size（若不足）----
            if wav.size(-1) < segment_size:
                wav = torch.nn.functional.pad(
                    wav, (0, segment_size - wav.size(-1)), 'constant'
                )

            # ---- d) 调用特征提取器：返回字典，取 input_values 张量 ----
            # squeeze(0) 将 [1, T] -> [T]，假设音频为单通道
            input_values = feature_extractor(
                wav.squeeze(0), sampling_rate=sample_rate, return_tensors="pt"
            ).input_values  # shape: [1, T]

            # ---- e) 前向计算 HuBERT，取隐藏层 ----
            output = model(input_values.to(model.device), output_hidden_states=True)

            # 指定层选择：'avg' 表示对所有层取均值；否则取指定层 index
            if target_layer == 'avg':
                rep = torch.mean(torch.stack(output.hidden_states), dim=0)  # [B, T', C]
            else:
                rep = output.hidden_states[target_layer]  # [B, T', C]

            # ---- f) 生成保存路径（将 audio_dir 路径前缀替换为 rep_dir，并把扩展名改为 .hubert.npy）----
            rep_file = audio_file.replace(args.audio_dir, args.rep_dir).split('.')[0] + '.hubert.npy'
            rep_sub_dir = '/'.join(rep_file.split('/')[:-1])
            if not os.path.exists(rep_sub_dir):
                os.makedirs(rep_sub_dir, exist_ok=True)

            # ---- g) 保存表示到 .npy 文件（CPU 上的 numpy）----
            np.save(rep_file, rep.detach().cpu().numpy())

            # ---- h) 写入清单：前 valid_set_size 个样本写入验证集，其余写入训练集 ----
            if i < valid_set_size:
                with open(valid_file_list, 'a+', encoding='utf-8') as f:
                    f.write(f'{audio_file}\t{rep_file}\n')
            else:
                with open(train_file_list, 'a+', encoding='utf-8') as f:
                    f.write(f'{audio_file}\t{rep_file}\n')
