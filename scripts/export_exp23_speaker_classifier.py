import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.train_exp23_speaker_classifier import build_model
from speechtokenizer.speaker_identity.inference import CodesWaveformLogits, write_export_bundle


def _required(parser, args, names):
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        parser.error('missing required arguments: %s' % ', '.join('--' + name.replace('_', '-') for name in missing))


def _speaker_labels(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(payload, dict):
        payload = payload.get('records', payload.get('items', []))
    labels = {}
    for row in payload:
        index = int(row['label_index'])
        speaker = str(row['speaker_id'])
        if index in labels and labels[index] != speaker:
            raise ValueError('label index maps to multiple speakers: %d' % index)
        labels[index] = speaker
    if sorted(labels) != list(range(len(labels))):
        raise ValueError('speaker label indices must be contiguous from zero')
    return [labels[index] for index in range(len(labels))]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Package a scripted Exp23 classifier bundle')
    parser.add_argument('--scripted-classifier')
    parser.add_argument('--speaker-labels')
    parser.add_argument('--preprocessing')
    parser.add_argument('--model-config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--cache-manifest')
    parser.add_argument('--calibration')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--ecapa-savedir', default='output/models/speechbrain_spkrec_ecapa_voxceleb')
    parser.add_argument('--window-samples', type=int, default=48000)
    parser.add_argument('--code-frames', type=int, default=150)
    args = parser.parse_args(argv)
    if args.checkpoint:
        _required(parser, args, ('cache_manifest', 'calibration'))
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        holder = argparse.Namespace(
            device=args.device,
            ecapa_source=checkpoint['ecapa_source'],
            ecapa_savedir=args.ecapa_savedir,
            ecapa_stage='A',
            ecapa_trainable_pattern=[],
        )
        model = build_model(checkpoint['config'], holder)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(args.device).eval()
        wrapper = CodesWaveformLogits(model).to(args.device).eval()
        example_codes = torch.zeros(1, 3, args.code_frames, dtype=torch.long, device=args.device)
        example_waveform = torch.zeros(1, args.window_samples, dtype=torch.float32, device=args.device)
        scripted = torch.jit.trace(wrapper, (example_codes, example_waveform), strict=False)
        labels = _speaker_labels(args.cache_manifest)
        calibration = json.loads(Path(args.calibration).read_text(encoding='utf-8'))
        preprocessing = {
            'sample_rate': 16000,
            'window_samples': args.window_samples,
            'frame_hop_samples': 320,
            'rvq_layers': 3,
        }
        model_config = {
            'speaker_count': len(labels),
            'input_mode': 'codes_waveform',
            'codebook_size': 1024,
            'pad_index': 1024,
            'checkpoint_seed': checkpoint.get('seed'),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / 'classifier.pt'
            scripted.save(str(artifact))
            write_export_bundle(args.output_dir, artifact, labels, preprocessing, calibration, model_config)
        return 0
    _required(parser, args, ('scripted_classifier', 'speaker_labels', 'preprocessing', 'calibration', 'model_config'))
    labels = json.loads(Path(args.speaker_labels).read_text(encoding='utf-8'))
    preprocessing = json.loads(Path(args.preprocessing).read_text(encoding='utf-8'))
    calibration = json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    model_config = json.loads(Path(args.model_config).read_text(encoding='utf-8'))
    write_export_bundle(args.output_dir, args.scripted_classifier, labels, preprocessing, calibration, model_config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
