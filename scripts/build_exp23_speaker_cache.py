import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
import numpy as np
import torch

from scripts.experiment_utils import collect_environment_metadata, ensure_run_layout, file_sha256, write_json
from scripts.evaluate_lca_vs_base import build_model as build_tokenizer_model, load_audio_for_inference, load_state as load_tokenizer_state
from speechtokenizer.speaker_identity.cache import (
    DEFAULT_CACHE_BUDGET_BYTES,
    CacheRecord,
    atomic_write_flac,
    atomic_write_npz,
    encode_decode_frozen,
    estimate_cache_storage,
    is_resumable_cache_item,
    preflight_cache_storage,
    state_dict_sha256,
    validate_cache_record,
)
from speechtokenizer.speaker_identity.data import build_canonical_split, build_text_controlled_split


def build_parser():
    parser = argparse.ArgumentParser(description='Build frozen Exp23 SpeechTokenizer caches')
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--audio-root')
    parser.add_argument('--transcript-root')
    parser.add_argument('--models', nargs='+', choices=('base', 'lca'), default=['lca'])
    parser.add_argument('--layers', nargs='+', type=int, default=[3])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-speakers', type=int)
    parser.add_argument('--max-items-per-split', type=int)
    parser.add_argument('--split-kind', choices=('canonical', 'text-controlled'), default='canonical')
    parser.add_argument('--dry-run-estimate', action='store_true')
    return parser


def build_split_manifest(args, config):
    if not args.audio_root:
        raise ValueError('--audio-root is required unless split construction is injected')
    root = Path(args.audio_root)
    if args.split_kind == 'canonical':
        counts = config['split']['counts']
        manifest = build_canonical_split(
            root,
            train_count=counts['train'],
            validation_count=counts['validation'],
            test_count=counts['test'],
            seed=config['split']['seed'],
        )
    else:
        if not args.transcript_root:
            raise ValueError('--transcript-root is required for text-controlled split')
        counts = config['text_controlled_split']['counts']
        manifest = build_text_controlled_split(
            root,
            Path(args.transcript_root),
            train_count=counts['train'],
            validation_count=counts['validation'],
            test_count=counts['test'],
            seed=config['text_controlled_split']['seed'],
            excluded_speakers=set(config['text_controlled_split']['excluded_speakers']),
        )
    if args.max_speakers is not None:
        allowed = set(manifest['speakers'][:args.max_speakers])
        manifest['speakers'] = [speaker for speaker in manifest['speakers'] if speaker in allowed]
        for split_name in manifest['splits']:
            manifest['splits'][split_name] = [row for row in manifest['splits'][split_name] if row['speaker_id'] in allowed]
    if args.max_items_per_split is not None:
        for split_name in manifest['splits']:
            manifest['splits'][split_name] = manifest['splits'][split_name][:args.max_items_per_split]
    return manifest


def estimate_manifest_storage(manifest, layers):
    item_count = sum(len(rows) for rows in manifest['splits'].values())
    return estimate_cache_storage(item_count, 10 * 16000, 500, max(layers))


def build_model(config_path, checkpoint_path, device):
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    return build_tokenizer_model(config).to(device)


def load_state(model, checkpoint_path):
    load_tokenizer_state(model, checkpoint_path)
    return model


def load_source_audio(path):
    waveform = load_audio_for_inference(path, 16000)
    audio = waveform.squeeze(0).detach().cpu().numpy().astype('float32', copy=False)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError('source audio must become finite mono audio after 16 kHz resampling')
    return audio


def _manifest_rows(manifest):
    for split_name, rows in manifest['splits'].items():
        for row in rows:
            yield split_name, row


def _row_path(row):
    return Path(row.get('audio_path') or row.get('path'))


def _write_cache(model, model_name, layers, manifest, cache_root, checkpoint_hash, config_hash, device):
    labels = {speaker: index for index, speaker in enumerate(manifest['speakers'])}
    records = []
    failures = []
    for split_name, row in _manifest_rows(manifest):
        source = _row_path(row)
        key = hashlib.sha256(('%s|%s|%s|%s' % (model_name, layers, split_name, source)).encode('utf-8')).hexdigest()[:20]
        item_root = cache_root / model_name / ('l%d' % layers) / split_name
        codes_path = item_root / (key + '.npz')
        reconstruction_path = item_root / (key + '.flac')
        audio = load_source_audio(source)
        record = CacheRecord(
            speaker_id=row['speaker_id'],
            label_index=labels[row['speaker_id']],
            utterance_group=row['utterance_group'],
            split=split_name,
            source_audio_path=source,
            model_name=model_name,
            rvq_layers=layers,
            codes_path=codes_path,
            reconstruction_path=reconstruction_path,
            original_sample_count=len(audio),
            code_frame_count=max(1, len(audio) // 320),
            valid_sample_count=len(audio),
            transcript_hash=row.get('transcript_hash'),
            checkpoint_sha256=checkpoint_hash,
            config_sha256=config_hash,
        )
        if is_resumable_cache_item(record, record.to_json_dict()):
            records.append(record.to_json_dict())
            continue
        try:
            waveform = torch.from_numpy(audio).to(device).view(1, 1, -1)
            codes, reconstruction, _ = encode_decode_frozen(model, waveform, layers)
            record = CacheRecord(**dict(record.to_json_dict(), code_frame_count=int(codes.shape[1]), valid_sample_count=min(len(audio), int(reconstruction.numel()))))
            atomic_write_npz(codes_path, codes.cpu().numpy())
            atomic_write_flac(reconstruction_path, reconstruction.cpu().numpy())
            validate_cache_record(record)
            records.append(record.to_json_dict())
        except Exception as error:
            failures.append({'audio_path': str(source), 'error': '%s: %s' % (type(error).__name__, error)})
    return records, failures


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    run_dir = Path(args.run_dir).resolve()
    cache_root = Path(args.cache_root).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    ensure_run_layout(run_dir)
    manifest_dir = run_dir / 'cache_manifest'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = build_split_manifest(args, config)
    estimate = estimate_manifest_storage(split_manifest, args.layers)
    budget_bytes = int(float(config['cache'].get('budget_gb', 30)) * 1024 ** 3)
    storage_audit = preflight_cache_storage(
        cache_root,
        estimate['estimated_bytes'],
        max_bytes=budget_bytes or DEFAULT_CACHE_BUDGET_BYTES,
        free_space_multiplier=float(config['cache'].get('free_space_multiplier', 1.2)),
    )
    write_json(manifest_dir / 'cache_storage_estimate.json', dict(estimate, **storage_audit))
    write_json(manifest_dir / 'split_manifest.json', split_manifest)
    write_json(manifest_dir / 'environment.json', collect_environment_metadata(Path.cwd()))
    write_json(manifest_dir / 'config.json', config)
    if args.dry_run_estimate:
        return 0
    all_records = []
    all_failures = []
    tokenizer_state_hashes = []
    device = torch.device(args.device)
    config_hash = file_sha256(config_path)
    for model_name in args.models:
        upstream = config['frozen_upstream'][model_name]
        upstream_config = Path(upstream['config'])
        checkpoint = Path(upstream['checkpoint'])
        checkpoint_hash = file_sha256(checkpoint)
        for layers in args.layers:
            model = build_model(upstream_config, checkpoint, device)
            load_state(model, checkpoint)
            before_hash = state_dict_sha256(model.state_dict())
            records, failures = _write_cache(model, model_name, layers, split_manifest, cache_root, checkpoint_hash, config_hash, device)
            after_hash = state_dict_sha256(model.state_dict())
            if before_hash != after_hash:
                raise RuntimeError('SpeechTokenizer state changed during cache build')
            all_records.extend(records)
            all_failures.extend(failures)
            tokenizer_state_hashes.append({'model': model_name, 'rvq_layers': layers, 'before_sha256': before_hash, 'after_sha256': after_hash})
    write_json(manifest_dir / 'cache_records.json', all_records)
    write_json(manifest_dir / 'cache_failures.json', all_failures)
    write_json(manifest_dir / 'tokenizer_state_hashes.json', tokenizer_state_hashes)
    return 1 if all_failures else 0


if __name__ == '__main__':
    sys.exit(main())
