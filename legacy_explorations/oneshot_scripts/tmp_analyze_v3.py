import json
from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = Path('output/experiments/exp5_lca_component_factorial_20260603_seed42/runs')
runs = {
    'V0_clean_control': base / 'V0_full_depth_clean_control',
    'V2_channelsim_only': base / 'V2_channelsim_only',
    'V3_random_l_channelsim': base / 'V3_random_l_channelsim',
}

print('# JSONL summaries')
for name, run in runs.items():
    p = run / 'metrics' / 'lca_train_metrics.jsonl'
    print('\n##', name)
    rows = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    print('records', len(rows), 'last_step', rows[-1]['step'], 'last_epoch', rows[-1]['epoch'], 'last_lr', rows[-1]['lr'])
    by_l = defaultdict(list)
    by_channel = defaultdict(list)
    by_l_channel = defaultdict(list)
    recent_cut = rows[-1]['step'] - 5000
    recent_by_l = defaultdict(list)
    for r in rows:
        by_l[r['sampled_L']].append(r)
        by_channel[r['channel']].append(r)
        by_l_channel[(r['sampled_L'], r['channel'])].append(r)
        if r['step'] >= recent_cut:
            recent_by_l[r['sampled_L']].append(r)
    print('by L all:')
    for L, rs in sorted(by_l.items()):
        print(f'  L{L}: n={len(rs):3d} full={sum(r["full_branch"] for r in rs)/len(rs):8.3f} comm={sum(r["comm_branch"] for r in rs)/len(rs):8.3f}')
    print('by L last5k:')
    for L, rs in sorted(recent_by_l.items()):
        print(f'  L{L}: n={len(rs):3d} full={sum(r["full_branch"] for r in rs)/len(rs):8.3f} comm={sum(r["comm_branch"] for r in rs)/len(rs):8.3f}')
    print('by channel all:')
    for ch, rs in sorted(by_channel.items()):
        print(f'  {ch:18s} n={len(rs):3d} comm={sum(r["comm_branch"] for r in rs)/len(rs):8.3f} drop={sum(r["channel_stats"]["actual_p_drop"] for r in rs)/len(rs):.4f} sub={sum(r["channel_stats"]["actual_p_sub"] for r in rs)/len(rs):.4f}')

print('\n# Dev scalar summaries')
for name, run in runs.items():
    logdir = run / 'checkpoints' / 'logs'
    print('\n##', name)
    acc = EventAccumulator(str(logdir), size_guidance={'scalars': 0})
    acc.Reload()
    tags = sorted(acc.Tags().get('scalars', []))
    for tag in tags:
        if tag.startswith('dev/mel error') or tag.startswith('dev/full_depth_mel_error') or tag.startswith('dev/comm_mel/'):
            vals = acc.Scalars(tag)
            last = vals[-1]
            best = min(vals, key=lambda e: e.value)
            print(f'{tag:42s} n={len(vals):2d} last=({last.step:5d},{last.value:.6f}) best=({best.step:5d},{best.value:.6f})')
