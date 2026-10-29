"""Durable remote supervisor: smoke verification, full evaluation, aggregation."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--project',type=Path,required=True)
    a=ap.parse_args();r=a.run.resolve();project=a.project.resolve()
    import fcntl
    lock=(r/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    expected_download_hashes={
        'downloads/fleurs_test.tar.gz':'d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356',
        'downloads/tedlium_test.parquet':'4fae5ef76a9ca97e719aed0a365db919684e32e6bf24d147250a80594621af5e',
        'asr_model/model.bin':'62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a',
    }
    for rel,expected in expected_download_hashes.items():
        assert sha(r/rel)==expected, 'Pinned source SHA256 mismatch: '+rel
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join([str(r/'deps'),str(r/'vendor'),str(project)]))
    vendor_libs=list(Path('/home/suer/miniconda3/envs/paper/lib/python3.10/site-packages/nvidia').glob('*/lib'))
    env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,vendor_libs))+os.pathsep+env.get('LD_LIBRARY_PATH','')
    (r/'provenance').mkdir(exist_ok=True)
    save(r/'provenance/source_lfs_sha256.json',expected_download_hashes)
    files=list((project/'speechtokenizer').rglob('*.py'))+list((project/'nas').rglob('*.py'))
    files += [project/'scripts'/f for f in ['evaluate_lca_vs_base.py','evaluate_sample_audio_quality.py','experiment_utils.py']]
    source_hashes={str(p.relative_to(project)):sha(p) for p in sorted(files)}
    snap=r/'provenance/source_hashes.json'
    if snap.exists():
        assert json.loads(snap.read_text())==source_hashes,'Model source changed since run began'
    else:
        save(snap,source_hashes)
        for rel in source_hashes:
            target=r/'provenance/source'/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(project/rel,target)
    (r/'provenance/pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],env=env,text=True))
    manifest=[json.loads(s) for s in (r/'manifests/all_samples.jsonl').read_text().splitlines()]
    sample_ids={v['sample_id'].replace('_mic1','') for v in manifest}
    train_audit={}
    for dirname in ['exp2_scit_speech_distill30_retrain_20260529_seed42','exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42']:
        for split in ['train','valid']:
            p=project/'output/experiments'/dirname/'artifacts'/(split+'_files.txt')
            lines=p.read_text(encoding='utf-8-sig').splitlines()
            ids={Path(s.split('\t')[0].replace('\\','/')).stem for s in lines if s.strip()}
            overlap=sorted(ids&sample_ids)
            train_audit[dirname+'/'+split]=dict(count=len(lines),sha256=sha(p),id_overlap=overlap,
                external_path_mentions=sum(any(t in s.lower() for t in ['vctk','fleurs','tedlium','ted-lium']) for s in lines))
            assert not overlap and not train_audit[dirname+'/'+split]['external_path_mentions'],'Train/test path overlap'
    save(r/'provenance/training_manifest_audit.json',train_audit)
    phases=[('smoke_objective','objective',True),('smoke_asr','asr',True),
            ('full_objective','objective',False),('full_asr','asr',False)]
    try:
        for name,phase,smoke in phases:
            assert all(sha(project/rel)==h for rel,h in source_hashes.items()),'Model source changed during run'
            cmd=[sys.executable,'-u',str(r/'commands/evaluate.py'),'--run',str(r),'--project',str(project),'--phase',phase]
            if smoke:cmd.append('--smoke')
            save(r/'pipeline_status.json',dict(status='running',phase=name,pid=os.getpid(),started_unix=started,updated_unix=time.time()))
            with (r/'logs'/(name+'.log')).open('a') as log:
                print('Starting '+name,flush=True)
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            output=r/('smoke' if smoke else 'full')
            status=json.loads((output/(phase+'_status.json')).read_text())
            assert status['status']=='completed' and status['completed']==status['total'],name
            if smoke and phase=='objective':
                rows=[json.loads(s) for s in (output/'objective.jsonl').read_text().splitlines()]
                assert not any(m['errors'] for row in rows for m in row['metrics']), 'Smoke metric failures'
            if smoke and phase=='asr':
                rows=[json.loads(s) for s in (output/'asr.jsonl').read_text().splitlines()]
                assert all(set(row['asr'])=={'original','base','lca'} for row in rows if row['reference']), 'Missing smoke ASR'
            subprocess.run([sys.executable,str(r/'commands/aggregate.py'),'--run',str(r)]+(['--smoke'] if smoke else []),env=env,check=True)
        summary=json.loads((r/'full/reports/summary.json').read_text())
        assert summary['status']=='completed'
        save(r/'pipeline_status.json',dict(status='completed',pid=os.getpid(),started_unix=started,updated_unix=time.time(),
            duration_seconds=time.time()-started,total_samples=len(manifest)))
    except Exception as exc:
        save(r/'pipeline_status.json',dict(status='failed',pid=os.getpid(),started_unix=started,updated_unix=time.time(),error=str(exc)))
        raise


if __name__=='__main__':main()
