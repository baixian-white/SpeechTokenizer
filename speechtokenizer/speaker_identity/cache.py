import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


SAMPLE_RATE = 16000
CODEBOOK_SIZE = 1024
DEFAULT_CACHE_BUDGET_BYTES = 30 * 1024 ** 3


def _require_int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('%s must be an integer' % name)
    if value < minimum:
        raise ValueError('%s must be at least %d' % (name, minimum))
    return value


@dataclass(frozen=True)
class CropBounds:
    audio_start: int
    audio_end: int
    token_start: int
    token_end: int


def aligned_crop_bounds(audio_length, token_length, start_sample, crop_samples):
    audio_length = _require_int(audio_length, 'audio_length', 1)
    token_length = _require_int(token_length, 'token_length', 1)
    start_sample = _require_int(start_sample, 'start_sample', 0)
    crop_samples = _require_int(crop_samples, 'crop_samples', 1)
    audio_start = min(start_sample, audio_length - 1)
    audio_end = min(audio_length, audio_start + crop_samples)
    start_ratio = audio_start / float(audio_length)
    end_ratio = audio_end / float(audio_length)
    token_start = min(token_length - 1, int(math.floor(start_ratio * token_length)))
    token_end = min(token_length, int(math.ceil(end_ratio * token_length)))
    token_end = max(token_start + 1, token_end)
    return CropBounds(audio_start, audio_end, token_start, token_end)


@dataclass(frozen=True)
class CacheRecord:
    speaker_id: str
    label_index: int
    utterance_group: str
    split: str
    source_audio_path: Path
    model_name: str
    rvq_layers: int
    codes_path: Path
    reconstruction_path: Path
    original_sample_count: int
    code_frame_count: int
    valid_sample_count: int
    transcript_hash: object
    checkpoint_sha256: str
    config_sha256: str

    def __post_init__(self):
        for name in ('source_audio_path', 'codes_path', 'reconstruction_path'):
            object.__setattr__(self, name, Path(getattr(self, name)))

    def to_json_dict(self):
        payload = asdict(self)
        for name in ('source_audio_path', 'codes_path', 'reconstruction_path'):
            payload[name] = str(payload[name])
        return payload

    @classmethod
    def from_json_dict(cls, payload):
        allowed = {field.name for field in fields(cls)}
        if set(payload) != allowed:
            raise ValueError('cache record fields do not match the frozen schema')
        return cls(**payload)


def state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if not torch.is_tensor(tensor):
            raise TypeError('state_dict value %s is not a tensor' % key)
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(json.dumps(list(value.shape), separators=(',', ':')).encode('ascii'))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order='C'))
    return digest.hexdigest()


def assert_state_dict_unchanged(before, after):
    before_hash = state_dict_sha256(before)
    after_hash = state_dict_sha256(after)
    if before_hash != after_hash:
        raise RuntimeError('SpeechTokenizer state changed during frozen cache build')
    return before_hash


def encode_decode_frozen(model, waveform, rvq_layers):
    _require_int(rvq_layers, 'rvq_layers', 1)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    before_hash = state_dict_sha256(before)
    with torch.inference_mode():
        encoded = model.encode(waveform, n_q=rvq_layers, st=0)
        reconstruction = model.decode(encoded, st=0)
    after = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    after_hash = state_dict_sha256(after)
    assert_state_dict_unchanged(before, after)
    codes = encoded.detach()
    if codes.ndim == 3 and codes.shape[1] == 1:
        codes = codes[:, 0, :]
    if codes.ndim != 2:
        raise ValueError('encoded codes must have shape [L,T] after removing batch dimension')
    audio = reconstruction.detach()
    while audio.ndim > 1 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim != 1:
        raise ValueError('reconstruction must be mono')
    return codes, audio, {'before_sha256': before_hash, 'after_sha256': after_hash}


def _load_codes(path):
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            if 'codes' not in archive:
                raise ValueError('codes archive is missing the codes array')
            return np.asarray(archive['codes'])
    except ValueError:
        raise
    except Exception as error:
        raise ValueError('codes_path is unreadable: %s' % error)


def validate_cache_record(record):
    if not isinstance(record, CacheRecord):
        raise TypeError('record must be CacheRecord')
    for name in ('source_audio_path', 'codes_path', 'reconstruction_path'):
        path = getattr(record, name)
        if not path.is_file():
            raise ValueError('%s does not exist: %s' % (name, path))
    for name in ('original_sample_count', 'code_frame_count', 'valid_sample_count'):
        _require_int(getattr(record, name), name, 1)
    _require_int(record.label_index, 'label_index', 0)
    _require_int(record.rvq_layers, 'rvq_layers', 1)
    for name in ('speaker_id', 'utterance_group', 'split', 'model_name', 'checkpoint_sha256', 'config_sha256'):
        value = getattr(record, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError('%s must be a non-empty string' % name)
    codes = _load_codes(record.codes_path)
    if codes.ndim != 2:
        raise ValueError('codes must be two-dimensional [L,T]')
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError('codes must use an integer dtype')
    if codes.shape[0] != record.rvq_layers:
        raise ValueError('codes first dimension must equal rvq_layers')
    if codes.shape[1] != record.code_frame_count:
        raise ValueError('code_frame_count does not match cached codes')
    if codes.size == 0 or int(codes.min()) < 0 or int(codes.max()) >= CODEBOOK_SIZE:
        raise ValueError('codes must be in [0, 1024)')
    try:
        audio, sample_rate = sf.read(str(record.reconstruction_path), dtype='float32', always_2d=False)
    except Exception as error:
        raise ValueError('reconstruction_path is unreadable: %s' % error)
    if sample_rate != SAMPLE_RATE:
        raise ValueError('reconstruction sample rate must be 16000 Hz')
    if audio.ndim != 1:
        raise ValueError('reconstruction must be mono')
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError('reconstruction audio must be finite and non-empty')
    if record.valid_sample_count > len(audio):
        raise ValueError('valid_sample_count exceeds reconstruction length')
    return {
        'code_shape': list(codes.shape),
        'code_dtype': str(codes.dtype),
        'audio_samples': int(len(audio)),
        'sample_rate': int(sample_rate),
    }


def load_cache_item(record):
    validate_cache_record(record)
    codes = _load_codes(record.codes_path)
    audio, _ = sf.read(str(record.reconstruction_path), dtype='float32', always_2d=False)
    return {'record': record, 'codes': codes, 'audio': np.asarray(audio[:record.valid_sample_count])}


def crop_cache_item(record, bounds):
    if not isinstance(bounds, CropBounds):
        raise TypeError('bounds must be CropBounds')
    item = load_cache_item(record)
    if not (0 <= bounds.audio_start < bounds.audio_end <= len(item['audio'])):
        raise ValueError('audio crop bounds are outside the cached item')
    if not (0 <= bounds.token_start < bounds.token_end <= item['codes'].shape[1]):
        raise ValueError('token crop bounds are outside the cached item')
    return {
        'record': record,
        'audio': item['audio'][bounds.audio_start:bounds.audio_end],
        'codes': item['codes'][:, bounds.token_start:bounds.token_end],
        'bounds': bounds,
    }


def estimate_cache_storage(item_count, samples_per_item, frames_per_item, rvq_layers):
    item_count = _require_int(item_count, 'item_count', 0)
    samples_per_item = _require_int(samples_per_item, 'samples_per_item', 0)
    frames_per_item = _require_int(frames_per_item, 'frames_per_item', 0)
    rvq_layers = _require_int(rvq_layers, 'rvq_layers', 1)
    audio_bytes = item_count * samples_per_item * 2
    code_bytes = item_count * frames_per_item * rvq_layers * 2
    metadata_bytes = item_count * 2048
    return {
        'item_count': item_count,
        'audio_bytes': audio_bytes,
        'code_bytes': code_bytes,
        'metadata_bytes': metadata_bytes,
        'estimated_bytes': audio_bytes + code_bytes + metadata_bytes,
    }


def preflight_cache_storage(cache_root, estimated_bytes, max_bytes=DEFAULT_CACHE_BUDGET_BYTES, free_space_multiplier=1.2):
    estimated_bytes = _require_int(estimated_bytes, 'estimated_bytes', 0)
    max_bytes = _require_int(max_bytes, 'max_bytes', 1)
    if isinstance(free_space_multiplier, bool) or not isinstance(free_space_multiplier, (int, float)):
        raise TypeError('free_space_multiplier must be numeric')
    if free_space_multiplier <= 1.0:
        raise ValueError('free_space_multiplier must be greater than 1.0')
    if estimated_bytes > max_bytes:
        raise ValueError('estimated cache exceeds configured budget')
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(root))
    free_bytes = int(usage.free if hasattr(usage, 'free') else usage[2])
    required = int(math.ceil(estimated_bytes * float(free_space_multiplier)))
    if free_bytes < required:
        raise ValueError('insufficient free space for cache build')
    return {
        'estimated_bytes': estimated_bytes,
        'max_bytes': max_bytes,
        'free_bytes': free_bytes,
        'required_free_bytes': required,
        'free_space_multiplier': float(free_space_multiplier),
    }


def atomic_write_npz(path, codes):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix='.tmp-', suffix='.npz', dir=str(destination.parent), delete=False)
    temp_path = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(str(temp_path), codes=np.asarray(codes))
        os.replace(str(temp_path), str(destination))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_flac(path, audio, sample_rate=SAMPLE_RATE):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix='.tmp-', suffix='.flac', dir=str(destination.parent), delete=False)
    temp_path = Path(handle.name)
    handle.close()
    try:
        sf.write(str(temp_path), np.asarray(audio, dtype=np.float32), sample_rate, format='FLAC')
        os.replace(str(temp_path), str(destination))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def is_resumable_cache_item(record, manifest_payload):
    try:
        expected = CacheRecord.from_json_dict(manifest_payload)
    except Exception:
        return False
    if expected != record:
        return False
    try:
        validate_cache_record(record)
    except (OSError, TypeError, ValueError):
        return False
    return True
