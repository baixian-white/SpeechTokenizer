import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import aligned_crop_bounds, crop_cache_item


class CachedSpeakerDataset(Dataset):
    def __init__(self, records, split, crop_samples=48000, for_training=False, seed=42):
        if for_training and split == 'test':
            raise ValueError('test manifest is forbidden during training')
        self.records = list(records)
        self.split = split
        self.crop_samples = int(crop_samples)
        self.for_training = bool(for_training)
        self.random = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        audio_length = record.valid_sample_count
        if self.for_training:
            maximum = max(0, audio_length - self.crop_samples)
            start = self.random.randint(0, maximum) if maximum else 0
        else:
            start = max(0, (audio_length - self.crop_samples) // 2)
        bounds = aligned_crop_bounds(audio_length, record.code_frame_count, start, self.crop_samples)
        item = crop_cache_item(record, bounds)
        waveform = item['audio'].astype(np.float32)
        active_layers = record.rvq_layers
        if self.for_training:
            gain_db = self.random.uniform(-3.0, 3.0)
            waveform = waveform * (10.0 ** (gain_db / 20.0))
            signal_rms = float(np.sqrt(np.mean(waveform ** 2) + 1e-8))
            snr_db = self.random.uniform(10.0, 30.0)
            noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
            noise_rng = np.random.default_rng(self.random.randrange(0, 2 ** 32))
            waveform = waveform + noise_rng.normal(0.0, noise_rms, size=waveform.shape).astype(np.float32)
            draw = self.random.random()
            active_layers = 3 if draw < 0.6 else (1 if draw < 0.8 else 2)
        return {
            'codes': item['codes'].astype(np.int64),
            'waveform': waveform.astype(np.float32),
            'label': int(record.label_index),
            'speaker_id': record.speaker_id,
            'utterance_group': record.utterance_group,
            'bounds': bounds,
            'active_layers': min(active_layers, record.rvq_layers),
        }


def collate_cache_batch(items, pad_index=1024):
    batch_size = len(items)
    layers = max(item['codes'].shape[0] for item in items)
    frames = max(item['codes'].shape[1] for item in items)
    samples = max(len(item['waveform']) for item in items)
    codes = torch.full((batch_size, layers, frames), pad_index, dtype=torch.long)
    frame_mask = torch.zeros(batch_size, frames, dtype=torch.bool)
    layer_mask = torch.zeros(batch_size, layers, dtype=torch.bool)
    waveform = torch.zeros(batch_size, samples, dtype=torch.float32)
    labels = torch.empty(batch_size, dtype=torch.long)
    for index, item in enumerate(items):
        layer_count, frame_count = item['codes'].shape
        sample_count = len(item['waveform'])
        codes[index, :layer_count, :frame_count] = torch.from_numpy(item['codes'])
        frame_mask[index, :frame_count] = True
        active_layers = min(int(item.get('active_layers', layer_count)), layer_count)
        layer_mask[index, :active_layers] = True
        waveform[index, :sample_count] = torch.from_numpy(item['waveform'])
        labels[index] = item['label']
    return {'codes': codes, 'frame_mask': frame_mask, 'layer_mask': layer_mask, 'waveform': waveform, 'labels': labels, 'items': items}


def replication_decision(validation_top1, stop_threshold=0.85, replicate_threshold=0.88):
    value = float(validation_top1)
    if value >= replicate_threshold:
        return 'freeze_and_replicate'
    if value >= stop_threshold:
        return 'bounded_revision'
    return 'stop'


def assert_training_manifest_is_validation_safe(split_name):
    if split_name == 'test':
        raise ValueError('test manifest is forbidden during training and stage gates')
