"""Large-set clean evaluation without writing reconstructed WAVs.

Evaluates Base vs LCA for clean L=1/2/3 and writes CSV/JSON summaries only.
Designed for full LibriSpeech test-clean/test-other to avoid huge sample output.
"""
import argparse, csv, json, math, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_lca_vs_base import load_json, build_model, load_state, load_audio_for_inference, read_sample_rows, hash_file
from scripts.evaluate_sample_audio_quality import si_snr_db, pearson_corr, compute_mel_l1, maybe_compute_stoi, maybe_compute_pesq


def model_inference_clean(model, x_np, L, n_q, device):
    x = torch.from_numpy(x_np).to(device).view(1,1,-1)
    with torch.no_grad():
        codes = model.encode(x, n_q=int(n_q), st=0)
        recon = model.decode(codes[:int(L)].contiguous().long(), st=0)
    return recon[0,0].detach().cpu().numpy().astype(np.float32)


def eval_one(model_name, model, audio_np, sample_id, L, n_q, cfg, sample_rate, device):
    recon = model_inference_clean(model, audio_np, L, n_q, device)
    length = min(len(audio_np), len(recon))
    ref = audio_np[:length].astype(np.float32, copy=False)
    est = recon[:length].astype(np.float32, copy=False)
    diff = est-ref
    ref_rms=float(np.sqrt(np.mean(ref**2))) if len(ref) else 0.0
    est_rms=float(np.sqrt(np.mean(est**2))) if len(est) else 0.0
    return {
        'model': model_name,
        'sample_id': sample_id,
        'L': int(L),
        'channel': 'clean',
        'ideal_bitrate_bps': int(L)*50*10,
        'duration_sec': float(length/sample_rate),
        'wave_l1': float(np.mean(np.abs(diff))) if len(diff) else float('nan'),
        'rmse': float(np.sqrt(np.mean(diff**2))) if len(diff) else float('nan'),
        'mel_l1': compute_mel_l1(ref, est, cfg),
        'si_snr_db': si_snr_db(ref, est),
        'corr': pearson_corr(ref, est),
        'stoi': maybe_compute_stoi(ref, est, sample_rate) or '',
        'pesq_wb': maybe_compute_pesq(ref, est, sample_rate) or '',
        'rms_ratio_db': 20.0*math.log10((est_rms+1e-8)/(ref_rms+1e-8)),
        'length_samples': int(length),
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as h:
        w=csv.DictWriter(h, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def mean(rows, key):
    vals=[]
    for r in rows:
        v=r.get(key)
        if isinstance(v,(int,float)) and not math.isnan(v): vals.append(v)
    return sum(vals)/len(vals) if vals else float('nan')


def write_summary(run_dir, rows, base_ckpt, lca_ckpt):
    buckets=defaultdict(list)
    for r in rows: buckets[(r['model'],r['L'])].append(r)
    lines=['# Full LibriSpeech clean evaluation summary','',f'- Base: `{base_ckpt}`',f'- LCA: `{lca_ckpt}`','']
    lines += ['| model | L | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ |','|---|---:|---:|---:|---:|---:|---:|']
    for model in ['base','lca']:
        for L in [1,2,3]:
            rs=buckets[(model,L)]
            lines.append(f"| {model} | {L} | {len(rs)} | {mean(rs,'mel_l1'):.3f} | {mean(rs,'stoi'):.3f} | {mean(rs,'pesq_wb'):.3f} | {mean(rs,'si_snr_db'):+.2f} |")
    lines += ['','| L | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |','|---:|---:|---:|---:|---:|']
    for L in [1,2,3]:
        b=buckets[('base',L)]; l=buckets[('lca',L)]
        lines.append(f"| {L} | {mean(l,'mel_l1')-mean(b,'mel_l1'):+.3f} | {mean(l,'stoi')-mean(b,'stoi'):+.3f} | {mean(l,'pesq_wb')-mean(b,'pesq_wb'):+.3f} | {mean(l,'si_snr_db')-mean(b,'si_snr_db'):+.2f} |")
    out=run_dir/'reports'/'full_clean_summary.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--base-config', required=True)
    ap.add_argument('--base-checkpoint', required=True)
    ap.add_argument('--lca-config', required=True)
    ap.add_argument('--lca-checkpoint', required=True)
    ap.add_argument('--sample-list', required=True)
    ap.add_argument('--max-samples', type=int, default=0, help='0 means all')
    ap.add_argument('--device', default='cuda')
    args=ap.parse_args()
    run_dir=Path(args.run_dir)
    base_cfg=load_json(args.base_config); lca_cfg=load_json(args.lca_config)
    sample_rate=int(base_cfg.get('sample_rate',16000)); n_q=int(base_cfg.get('n_q',3))
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('loading base', flush=True)
    base_model=build_model(base_cfg).to(device).eval(); load_state(base_model,args.base_checkpoint)
    print('loading lca', flush=True)
    lca_model=build_model(lca_cfg).to(device).eval(); load_state(lca_model,args.lca_checkpoint)
    max_samples=args.max_samples if args.max_samples and args.max_samples>0 else 10**9
    samples=read_sample_rows(args.sample_list, max_samples)
    print(f'evaluating {len(samples)} samples x 3 L x 2 models', flush=True)
    rows=[]; tic=time.time()
    for idx,s in enumerate(samples,1):
        audio=load_audio_for_inference(s['audio'], sample_rate).squeeze(0).numpy().astype(np.float32)
        for L in [1,2,3]:
            rows.append(eval_one('base',base_model,audio,s['sample_id'],L,n_q,base_cfg,sample_rate,str(device)))
            rows.append(eval_one('lca',lca_model,audio,s['sample_id'],L,n_q,base_cfg,sample_rate,str(device)))
        if idx % 50 == 0 or idx == len(samples):
            print(f'done {idx}/{len(samples)} elapsed={time.time()-tic:.1f}s', flush=True)
    fields=['model','sample_id','L','channel','ideal_bitrate_bps','duration_sec','wave_l1','rmse','mel_l1','si_snr_db','corr','stoi','pesq_wb','rms_ratio_db','length_samples']
    csv_path=run_dir/'metrics'/'full_clean_results.csv'; json_path=run_dir/'metrics'/'full_clean_results.json'
    write_csv(csv_path, rows, fields)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({'rows':rows,'sample_count':len(samples),'base_checkpoint':args.base_checkpoint,'lca_checkpoint':args.lca_checkpoint,'base_sha256':hash_file(args.base_checkpoint),'lca_sha256':hash_file(args.lca_checkpoint)}, ensure_ascii=False), encoding='utf-8')
    summary=write_summary(run_dir, rows, args.base_checkpoint, args.lca_checkpoint)
    print(json.dumps({'status':'completed','rows':len(rows),'csv':str(csv_path),'summary':str(summary)}, indent=2), flush=True)
if __name__ == '__main__': main()
