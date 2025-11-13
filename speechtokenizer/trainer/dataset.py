from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import torch
import numpy as np

# 常见的“特征维”候选集合，用于判断 [D, T] 是否需要转置成 [T, D]
_FEAT_DIMS = {80, 128, 256, 512, 768, 1024}


def _normalize_feature_shape_to_TD(x: torch.Tensor) -> torch.Tensor:
    """
    将特征张量统一为 [T, D] 形状。
    允许输入形状：
      - [T, D]
      - [1, T, D]  -> squeeze(0)
      - [D, T]     -> transpose(0, 1)
      - [D]        -> 视作 [T, D] 中 T=1 的退化情况（几乎不会出现）
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    # [1, T, D] -> [T, D]
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0).contiguous()

    # [D, T] -> [T, D]（当第0维看起来像特征维，而第1维不是）
    if x.ndim == 2:
        t0, t1 = x.shape
        if t0 in _FEAT_DIMS and t1 not in _FEAT_DIMS:
            x = x.transpose(0, 1).contiguous()

    # [D] -> [1, D]
    if x.ndim == 1:
        x = x.unsqueeze(0).contiguous()

    # 现在应当是 [T, D]
    return x


def collate_fn(data):
    """
    data: List[ (audio[T], feature[T,D]) ] 或 List[tensor] 等
    目标：将变长的 T 在 batch 维做 padding：
      - audio -> [B, T]
      - feature -> [B, T, D]
    """
    # data 可能是 [(audio, feat), (audio, feat), ...]
    # 或者单列数据（极少用到）。这里按你的训练代码默认二元组处理。
    is_one_data = not isinstance(data[0], tuple)
    outputs = []

    if is_one_data:
        # 兼容单列情况（极少用）
        for datum in data:
            if isinstance(datum, torch.Tensor):
                output = datum.unsqueeze(0)
            else:
                output = torch.tensor([datum])
            outputs.append(output)
        return tuple(outputs)

    # 典型路径：音频与特征各自打包
    # datum[0] 是一批 audio；datum[1] 是一批 feature
    for datum in zip(*data):
        if isinstance(datum[0], torch.Tensor):
            # 将 batch 里的每个样本先规范形状，再 pad
            items = []
            for x in datum:
                if x.ndim == 3:
                    # 特征兜底：可能是 [1,T,D]
                    x = _normalize_feature_shape_to_TD(x)
                elif x.ndim == 2 and x.shape[1] > x.shape[0]:
                    # 兜底：疑似 [D,T]，转 [T,D]（仅特征会出现）
                    x = x.transpose(0, 1).contiguous()
                items.append(x)

            # audio: [T] -> pad -> [B, T]
            # feature: [T, D] -> pad -> [B, T, D]
            output = pad_sequence(items, batch_first=True)

        else:
            output = torch.tensor(list(datum))
        outputs.append(output)

    return tuple(outputs)


def get_dataloader(ds, **kwargs):
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)


class audioDataset(Dataset):
    """
    file_list: 形如 ["<path/to/audio.flac>\t<path/to/xxx.hubert.npy>", ...]
    segment_size: 以“采样点”为单位（例如 48000 表示 3 秒@16kHz）
    downsample_rate: 特征帧步长对应的采样点数（HuBERT@16kHz 通常是320，对应 20ms）
    """

    def __init__(
        self,
        file_list,
        segment_size,
        sample_rate,
        downsample_rate=320,
        valid=False
    ):
        super().__init__()
        self.file_list = file_list
        self.segment_size = int(segment_size)
        self.sample_rate = int(sample_rate)
        self.valid = bool(valid)
        self.downsample_rate = int(downsample_rate)

    def __len__(self):
        return len(self.file_list)

    def _load_audio(self, audio_file: str) -> torch.Tensor:
        """
        读取音频 -> 单声道 float32，并重采样到 self.sample_rate。
        返回 shape: [T]
        """
        audio, sr = torchaudio.load(audio_file)  # [C, T]
        # 转单声道
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio.squeeze(0)
        # 重采样
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        # 保证 float32
        if audio.dtype != torch.float32:
            audio = audio.float()
        return audio  # [T]

    def _load_feature(self, feature_file: str) -> torch.Tensor:
        """
        读取 .npy 特征并规范为 [T, D]，float32
        """
        feat = np.load(feature_file)
        if feat.dtype != np.float32:
            feat = feat.astype(np.float32, copy=False)
        feat = torch.from_numpy(feat)
        feat = _normalize_feature_shape_to_TD(feat)  # -> [T, D]
        return feat

    def __getitem__(self, index):
        # 解析一行 "<audio>\t<feature>"
        file = self.file_list[index].strip()
        audio_file, feature_file = file.split('\t')

        # 加载音频与特征
        audio = self._load_audio(audio_file)        # [T]
        feature = self._load_feature(feature_file)  # [T_f, D]

        # 计算本次需要的特征帧数（与音频 segment 对齐）
        seg_T = self.segment_size
        seg_F = seg_T // self.downsample_rate  # 例如 48000/320=150帧

        # 训练 vs 验证 的取段策略
        if audio.size(-1) > seg_T:
            if self.valid:
                # 验证：固定从开头取
                audio_seg = audio[:seg_T]
                feat_seg = feature[:seg_F]
            else:
                # 训练：随机裁切音频，再按比例裁切特征
                max_audio_start = audio.size(-1) - seg_T
                audio_start = random.randint(0, int(max_audio_start))
                audio_end = audio_start + seg_T
                audio_seg = audio[audio_start:audio_end]

                # 将 audio_start 映射到特征起点（向下取整）
                feat_start = int(audio_start // self.downsample_rate)
                # 特征索引上限兜底（避免越界）
                max_feat_start = max(0, feature.size(0) - seg_F)
                feat_start = min(feat_start, max_feat_start)
                feat_end = feat_start + seg_F
                feat_seg = feature[feat_start:feat_end]
        else:
            # 音频不够长：训练时右侧 pad 到 seg_T；验证集不 pad 音频（但为了与特征对齐通常也建议 pad）
            if not self.valid:
                audio_seg = torch.nn.functional.pad(
                    audio, (0, seg_T - audio.size(-1)), 'constant'
                )
            else:
                # 验证也统一右侧 pad，保持长度一致，便于 val 时的 batch 处理
                audio_seg = torch.nn.functional.pad(
                    audio, (0, max(0, seg_T - audio.size(-1))), 'constant'
                )

            # 特征不足 seg_F 帧时，同样右侧 pad
            if feature.size(0) >= seg_F:
                feat_seg = feature[:seg_F]
            else:
                pad_len = seg_F - feature.size(0)
                D = feature.size(1)
                pad_feat = torch.zeros((pad_len, D), dtype=feature.dtype)
                feat_seg = torch.cat([feature, pad_feat], dim=0)

        # 最终返回：
        #   audio_seg: [T] (T=segment_size)
        #   feat_seg:  [T_f, D] (T_f=segment_size//downsample_rate)
        return audio_seg, feat_seg
