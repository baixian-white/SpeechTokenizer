"""Resumable, provenance-checked frozen-codec evaluation on generic manifests."""
import argparse
import collections
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import sys
import time
import traceback
from pathlib import Path

BASE_DIR = 'output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42'
LCA_DIR = 'output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42'
MODELS = {
    'base': (BASE_DIR+'/configs/scit_speech_base_config.json', BASE_DIR+'/checkpoints/SCIT-Speech-Base_best.pt',
             '8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6'),
    'lca': (LCA_DIR+'/configs/lca_finetune_config.json', LCA_DIR+'/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt',
            'af60223b2f44733f5a43d5a022871dad061dfe2421473c96f5d350c3d4b49a5a'),
}
ASR_OPTIONS = dict(language='en', task='transcribe', beam_size=5, best_of=5, patience=1,
    length_penalty=1, repetition_penalty=1, no_repeat_ngram_size=0, temperature=0,
    compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6,
    condition_on_previous_text=False, suppress_blank=True, suppress_tokens=[-1],
    without_timestamps=True, word_timestamps=False, vad_filter=False)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def key(row):
    return row['dataset']+'/'+row['sample_id']


def read_records(path):
    if not path.exists():
        return {}
    data = path.read_bytes()
    records = {}
    offset = 0
    lines = data.splitlines(keepends=True)
    for i,line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if i != len(lines)-1:
                raise
            path.with_suffix('.interrupted_tail.bin').write_bytes(line)
            with path.open('r+b') as f:
                f.truncate(offset)
            break
        k = key(row)
        if k in records:
            raise RuntimeError('Duplicate completed record '+k)
        records[k] = row
        offset += len(line)
        if i == len(lines)-1 and not line.endswith(b'\n'):
            with path.open('ab') as f:
                f.write(b'\n')
    return records


def metric_worker_init(project):
    sys.path.insert(0, project)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def objective_metrics(ref, est, cfg):
    import numpy as np
    from scripts.evaluate_sample_audio_quality import compute_mel_l1, si_snr_db
    from pesq import pesq
    from pystoi import stoi
    out = dict(errors={})
    fns = dict(mel_l1=lambda:compute_mel_l1(ref,est,cfg),
        si_snr_db=lambda:si_snr_db(ref,est),
        stoi=lambda:stoi(ref,est,16000,extended=False),
        pesq_wb=lambda:pesq(16000,ref,est,'wb'))
    for name, fn in fns.items():
        try:
            value = float(fn())
            if not math.isfinite(value):
                raise ValueError('nonfinite metric')
            out[name] = value
        except Exception as exc:
            out[name] = None
            out['errors'][name] = type(exc).__name__+': '+str(exc)
    out['wave_l1'] = float(np.mean(np.abs(est-ref)))
    out['length_samples'] = len(ref)
    return out


def resolved_config(project, path):
    cfg = json.loads((project/path).read_text(encoding='utf-8-sig'))
    cfg['nas_encoder_config'] = str(project/cfg['nas_encoder_config'].replace('\\','/'))
    return cfg


def make_identity(a, samples):
    identity = dict(manifest_sha256=sha(a.run/'manifests/all_samples.jsonl'),
        evaluator_sha256=sha(Path(__file__)), project=str(a.project),
        smoke=a.smoke, selected_ids=[key(r) for r in samples], phase=a.phase,
        sample_rate=16000, layers=[1,2,3], codec_dtype='float32',
        alignment='common-length truncation only', models={}, versions={})
    for model,(cfg_path,ckpt,expected) in MODELS.items():
        actual = sha(a.project/ckpt)
        assert actual == expected, (model,'checkpoint hash mismatch',actual)
        cfg = resolved_config(a.project,cfg_path)
        identity['models'][model] = dict(checkpoint_sha256=actual, config_sha256=sha(a.project/cfg_path),
            nas_config_sha256=sha(Path(cfg['nas_encoder_config'])))
    for name in ('torch','torchaudio','numpy','soundfile','pesq','pystoi','faster-whisper','ctranslate2','jiwer'):
        try:
            identity['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            identity['versions'][name] = None
    identity['runtime'] = dict(python=sys.version,platform=platform.platform(),
                              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if a.phase == 'asr':
        identity['asr'] = dict(model='Systran/faster-whisper-small.en',
            model_sha256=sha(a.run/'asr_model/model.bin'),compute_type='float16',options=ASR_OPTIONS,
            normalizer_sha256=sha(a.run/'vendor/normalizers/english.py'),
            normalizer_mapping_sha256=sha(a.run/'vendor/normalizers/english.json'))
    return identity


def update_status(a, done, total, started, status, **extra):
    elapsed = time.monotonic()-started
    data = dict(status=status,phase=a.phase,smoke=a.smoke,pid=os.getpid(),
        completed=done,total=total,elapsed_seconds=elapsed,updated_unix=time.time(),**extra)
    write_json(a.out/(a.phase+'_status.json'),data)
    print(json.dumps(data,ensure_ascii=False),flush=True)


def run_objective(a,samples,done,handle,started):
    import numpy as np
    import soundfile as sf
    import torch
    from scripts.evaluate_lca_vs_base import build_model, load_audio_for_inference
    assert torch.cuda.is_available(), 'CUDA required; no accidental CPU fallback'
    models = {}
    cfgs = {}
    for name,(cfg_path,checkpoint,_) in MODELS.items():
        cfg = resolved_config(a.project,cfg_path)
        model = build_model(cfg)
        state = torch.load(a.project/checkpoint,map_location='cpu',weights_only=False)
        if isinstance(state,dict) and 'generator' in state:
            state = state['generator']
        model.load_state_dict(state,strict=True)
        models[name] = model.to('cuda').eval()
        cfgs[name] = cfg
        print('Strict checkpoint load OK: '+name,flush=True)
    pending = collections.deque()
    completed = len(done)
    def commit_first():
        nonlocal completed
        sample, jobs, meta = pending.popleft()
        rows = []
        for name, layer, future in jobs:
            rows.append(dict(model=name,L=layer,ideal_bitrate_bps=layer*500,**future.result()))
        record = dict(dataset=sample['dataset'],sample_id=sample['sample_id'],
            cluster_id=sample['cluster_id'],cluster_type=sample['cluster_type'],
            duration_sec=sample['duration_sec'],**meta,metrics=rows)
        handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        handle.flush()
        completed += 1
        if completed % 20 == 0 or a.smoke or completed == len(samples):
            update_status(a,completed,len(samples),started,'running',last_sample=key(sample))
    ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.metric_workers,mp_context=ctx,
            initializer=metric_worker_init,initargs=(str(a.project),)) as pool:
        for sample in samples:
            if key(sample) in done:
                # Cached L3 audio is needed for the next phase.
                for name in MODELS:
                    assert (a.out/'cache'/sample['dataset']/name/(sample['sample_id']+'.wav')).exists()
                continue
            audio = load_audio_for_inference(a.run/sample['audio'],16000).squeeze(0).numpy().astype(np.float32)
            assert len(audio)>0 and np.isfinite(audio).all(), key(sample)
            x = torch.from_numpy(audio).view(1,1,-1).cuda()
            jobs = []
            lengths = {}
            with torch.inference_mode():
                for name,model in models.items():
                    codes = model.encode(x,n_q=3,st=0)
                    for layer in (1,2,3):
                        recon = model.decode(codes[:layer].contiguous().long(),st=0)
                        est = recon[0,0].detach().cpu().numpy().astype(np.float32)
                        assert np.isfinite(est).all(), (key(sample),name,layer)
                        lengths[name+'_L'+str(layer)] = len(est)
                        length = min(len(audio),len(est))
                        ref = audio[:length].copy()
                        est = est[:length].copy()
                        if layer == 3:
                            path = a.out/'cache'/sample['dataset']/name/(sample['sample_id']+'.wav')
                            path.parent.mkdir(parents=True,exist_ok=True)
                            sf.write(path,est,16000,subtype='FLOAT')
                        jobs.append((name,layer,pool.submit(objective_metrics,ref,est,cfgs['base'])))
            pending.append((sample,jobs,dict(original_length_samples=len(audio),reconstruction_lengths=lengths)))
            if len(pending) >= 8:
                commit_first()
        while pending:
            commit_first()
    return completed


def edit_counts(reference,hypothesis):
    import jiwer
    r = jiwer.process_words(reference,hypothesis)
    return dict(substitutions=r.substitutions,deletions=r.deletions,insertions=r.insertions,
        reference_words=r.hits+r.substitutions+r.deletions,
        errors=r.substitutions+r.deletions+r.insertions,wer=float(r.wer))


def run_asr(a,samples,done,handle,started):
    import numpy as np
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel
    from scripts.evaluate_lca_vs_base import load_audio_for_inference
    from normalizers import EnglishTextNormalizer
    model = WhisperModel(str(a.run/'asr_model'),device='cuda',device_index=0,
        compute_type='float16',cpu_threads=2,num_workers=1,local_files_only=True)
    normalizer = EnglishTextNormalizer()
    completed = len(done)
    for sample in samples:
        if key(sample) in done:
            continue
        record = dict(dataset=sample['dataset'],sample_id=sample['sample_id'],
            cluster_id=sample['cluster_id'],cluster_type=sample['cluster_type'],
            reference=sample.get('transcript'),asr={},errors={})
        reference = normalizer(sample.get('transcript') or '')
        record['normalized_reference'] = reference
        if not reference:
            record['exclusion_reason'] = 'missing_or_empty_normalized_reference'
        else:
            arrays = {'original':load_audio_for_inference(a.run/sample['audio'],16000).squeeze(0).numpy().astype(np.float32)}
            for name in MODELS:
                path = a.out/'cache'/sample['dataset']/name/(sample['sample_id']+'.wav')
                arrays[name],sr = sf.read(path,dtype='float32')
                assert sr == 16000
            for name,audio in arrays.items():
                try:
                    segments,info = model.transcribe(audio,**ASR_OPTIONS)
                    segments = list(segments)
                    text = ' '.join(s.text.strip() for s in segments).strip()
                    hypothesis = normalizer(text)
                    item = dict(text=text,normalized_hypothesis=hypothesis,
                        duration_sec=len(audio)/16000,**edit_counts(reference,hypothesis))
                    if name != 'original' and 'original' in record['asr']:
                        item['vs_original'] = edit_counts(record['asr']['original']['normalized_hypothesis'],hypothesis)
                    record['asr'][name] = item
                except Exception as exc:
                    # Runtime failures must be fixed and rerun, not quietly excluded from WER.
                    raise RuntimeError(f'ASR failure {key(sample)} {name}: {exc}') from exc
        handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        handle.flush()
        completed += 1
        if completed % 10 == 0 or a.smoke or completed == len(samples):
            update_status(a,completed,len(samples),started,'running',last_sample=key(sample))
    return completed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--phase',choices=['objective','asr'],required=True)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--metric-workers',type=int,default=6)
    a = ap.parse_args()
    a.run = a.run.resolve(); a.project = a.project.resolve()
    a.out = a.run/('smoke' if a.smoke else 'full')
    a.out.mkdir(parents=True,exist_ok=True)
    # Exclusive advisory lock prevents two writers for the same phase/output.
    import fcntl
    lock = (a.out/(a.phase+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(a.project));sys.path.insert(0,str(a.run/'vendor'))
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(42);np.random.seed(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    samples = [json.loads(s) for s in (a.run/'manifests/all_samples.jsonl').read_text().splitlines() if s.strip()]
    if a.smoke:
        chosen=[]
        for ds in sorted({r['dataset'] for r in samples}):
            rs=[r for r in samples if r['dataset']==ds]
            chosen += [rs[0],max(rs,key=lambda r:r['duration_sec'])]
        samples = list({key(r):r for r in chosen}.values())
    identity = make_identity(a,samples)
    identity_path=a.out/(a.phase+'_identity.json')
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, 'Resume refused: provenance changed'
    else:
        write_json(identity_path,identity)
    results=a.out/(a.phase+'.jsonl')
    done=read_records(results)
    assert set(done).issubset({key(r) for r in samples})
    started=time.monotonic()
    update_status(a,len(done),len(samples),started,'running')
    try:
        with results.open('a',encoding='utf-8') as handle:
            count=(run_objective if a.phase=='objective' else run_asr)(a,samples,done,handle,started)
        update_status(a,count,len(samples),started,'completed')
    except Exception as exc:
        update_status(a,len(read_records(results)),len(samples),started,'failed',error=str(exc))
        raise


if __name__ == '__main__':
    main()
