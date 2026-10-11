import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch._dynamo
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from scripts.experiment_utils import collect_environment_metadata, ensure_run_layout, write_json
from speechtokenizer.speaker_identity.audio_encoder import CodecAwareEcapaEncoder
from speechtokenizer.speaker_identity.cache import CacheRecord
from speechtokenizer.speaker_identity.losses import joint_classification_loss
from speechtokenizer.speaker_identity.model import SpeakerIdentityModel
from speechtokenizer.speaker_identity.token_encoder import RVQTokenEncoder
from speechtokenizer.speaker_identity.training import CachedSpeakerDataset, collate_cache_batch, replication_decision


def build_parser():
    parser = argparse.ArgumentParser(description='Train Exp23 fixed-speaker classifier from frozen caches')
    parser.add_argument('--config', required=True)
    parser.add_argument('--cache-manifest', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--seed', type=int, choices=(41, 42, 43), default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--ecapa-source', default='speechbrain/spkrec-ecapa-voxceleb')
    parser.add_argument('--ecapa-savedir', default='output/models/speechbrain_spkrec_ecapa_voxceleb')
    parser.add_argument('--ecapa-stage', choices=('A', 'B'), default='A')
    parser.add_argument('--ecapa-trainable-pattern', action='append', default=['embedding_model.blocks.4'])
    return parser


def load_records(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(payload, dict):
        payload = payload.get('records', payload.get('items', []))
    return [CacheRecord.from_json_dict(row) for row in payload]


def build_model(config, args):
    token_cfg = config['model']['token_branch']
    token_encoder = RVQTokenEncoder(
        rvq_layers=3,
        codebook_size=1024,
        embedding_dim=token_cfg['token_embedding_dim'],
        model_dim=token_cfg['model_dim'],
        output_dim=token_cfg['speaker_embedding_dim'],
        block_count=token_cfg['block_count'],
        pad_index=token_cfg['pad_index'],
    )
    audio_encoder = CodecAwareEcapaEncoder.from_hparams(args.ecapa_source, args.ecapa_savedir, output_dim=token_cfg['speaker_embedding_dim'], device=args.device)
    audio_encoder.configure_stage(args.ecapa_stage, args.ecapa_trainable_pattern)
    aam = config['model']['classification_head']['aam_softmax']
    return SpeakerIdentityModel(token_encoder, audio_encoder, config['task']['speaker_count'], token_cfg['speaker_embedding_dim'], aam['scale'], aam['margin'])


def optimizer_groups(model):
    groups = {'token_fusion': [], 'projection_head': [], 'ecapa': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('audio_encoder.embedding_model'):
            groups['ecapa'].append(parameter)
        elif 'head' in name or name.startswith('audio_encoder.projection') or name.startswith('audio_encoder.input_adapter'):
            groups['projection_head'].append(parameter)
        else:
            groups['token_fusion'].append(parameter)
    return [
        {'params': groups['token_fusion'], 'lr': 3e-4, 'name': 'token_fusion'},
        {'params': groups['projection_head'], 'lr': 1e-4, 'name': 'projection_head'},
        {'params': groups['ecapa'], 'lr': 1e-5, 'name': 'ecapa'},
    ]


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def evaluate(model, loader, device):
    model.eval()
    labels = []
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch)
            labels.extend(batch['labels'].cpu().tolist())
            predictions.extend(output.fusion_logits.argmax(dim=-1).cpu().tolist())
    return {
        'top1': float(accuracy_score(labels, predictions)),
        'macro_f1': float(f1_score(labels, predictions, average='macro', zero_division=0)),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    records = load_records(args.cache_manifest)
    train_records = [record for record in records if record.split == 'train']
    validation_records = [record for record in records if record.split == 'validation']
    if not train_records or not validation_records:
        raise ValueError('cache manifest must contain train and validation records')
    run_dir = Path(args.run_dir)
    ensure_run_layout(run_dir)
    write_json(run_dir / 'environment.json', collect_environment_metadata(Path.cwd()))
    write_json(run_dir / 'training_config.json', {'args': vars(args), 'config': config})
    train_loader = DataLoader(CachedSpeakerDataset(train_records, 'train', for_training=True, seed=args.seed), batch_size=args.batch_size, shuffle=True, collate_fn=collate_cache_batch)
    validation_loader = DataLoader(CachedSpeakerDataset(validation_records, 'validation'), batch_size=args.batch_size, shuffle=False, collate_fn=collate_cache_batch)
    device = torch.device(args.device)
    model = build_model(config, args).to(device)
    groups = [group for group in optimizer_groups(model) if group['params']]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    total_steps = args.max_steps or max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    best_f1 = -1.0
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            output = model(batch)
            loss = joint_classification_loss(output.fusion_logits, output.token_logits, output.audio_logits, batch['labels'], token_embedding=output.token_embedding, audio_embedding=output.audio_embedding)
            (loss / args.gradient_accumulation).backward()
            if (batch_index + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        validation_metrics = evaluate(model, validation_loader, device)
        macro_f1 = validation_metrics['macro_f1']
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_top1 = validation_metrics['top1']
            torch.save({'state_dict': model.state_dict(), 'config': config, 'seed': args.seed, 'validation_top1': best_top1, 'validation_macro_f1': macro_f1, 'ecapa_source': args.ecapa_source}, run_dir / 'checkpoints' / 'best_macro_f1.pt')
        if args.max_steps and step >= args.max_steps:
            break
    decision = replication_decision(best_top1)
    write_json(run_dir / 'metrics' / 'validation_summary.json', {'seed': args.seed, 'top1': best_top1, 'macro_f1': best_f1, 'replication_decision': decision})
    return 0


if __name__ == '__main__':
    sys.exit(main())
