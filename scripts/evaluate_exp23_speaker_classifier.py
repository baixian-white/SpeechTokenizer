import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_exp23_speaker_classifier import build_model, load_records, move_batch
from speechtokenizer.speaker_identity.metrics import assert_evaluation_allowed, classification_metrics, fit_temperature
from speechtokenizer.speaker_identity.training import CachedSpeakerDataset, collate_cache_batch


def build_parser():
    parser = argparse.ArgumentParser(description='Evaluate a frozen Exp23 speaker classifier')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    parser.add_argument('--allow-test-evaluation', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--ecapa-savedir', default='output/models/speechbrain_spkrec_ecapa_voxceleb')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    assert_evaluation_allowed(args.split, args.allow_test_evaluation)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / ('test_predictions.csv' if args.split == 'test' else 'validation_predictions.csv')
    if predictions_path.exists() and not args.force:
        raise FileExistsError('%s exists; use --force to overwrite' % predictions_path)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    holder = argparse.Namespace(device=args.device, ecapa_source=checkpoint['ecapa_source'], ecapa_savedir=args.ecapa_savedir, ecapa_stage='A', ecapa_trainable_pattern=[])
    model = build_model(checkpoint['config'], holder)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device).eval()
    records = [record for record in load_records(args.cache_manifest) if record.split == args.split]
    loader = DataLoader(CachedSpeakerDataset(records, args.split), batch_size=args.batch_size, collate_fn=collate_cache_batch)
    logits_rows = []
    labels = []
    metadata = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, args.device)
            output = model(batch)
            logits_rows.append(output.fusion_logits.cpu().numpy())
            labels.extend(batch['labels'].cpu().tolist())
            metadata.extend(batch['items'])
    logits = np.concatenate(logits_rows, axis=0)
    labels_array = np.asarray(labels)
    temperature_path = output_dir / 'temperature.json'
    if args.split == 'validation':
        temperature = fit_temperature(logits, labels_array)
        temperature_path.write_text(json.dumps({'temperature': temperature}, indent=2) + '\n', encoding='utf-8')
    else:
        if not temperature_path.is_file():
            raise FileNotFoundError('validation-fitted temperature.json is required for test evaluation')
        temperature = float(json.loads(temperature_path.read_text(encoding='utf-8'))['temperature'])
    metrics = classification_metrics(logits / temperature, labels_array)
    (output_dir / ('%s_metrics.json' % args.split)).write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    predictions = (logits / temperature).argmax(axis=1)
    with predictions_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['speaker_id', 'utterance_group', 'label', 'prediction', 'correct'])
        writer.writeheader()
        for item, label, prediction in zip(metadata, labels, predictions):
            writer.writerow({'speaker_id': item['speaker_id'], 'utterance_group': item['utterance_group'], 'label': label, 'prediction': int(prediction), 'correct': int(label == prediction)})
    return 0


if __name__ == '__main__':
    sys.exit(main())
