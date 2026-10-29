"""Post-hoc, prespecified fixed-gain diagnostic on all FLEURS English test items."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True)
    a=ap.parse_args();parent=a.parent.resolve();root=parent/'level_control'
    assert not (root/'full').exists(), 'Never replace inputs of an already-started control experiment'
    for sub in ['manifests','data/fleurs_en_us','logs','commands','provenance']:(root/sub).mkdir(parents=True,exist_ok=True)
    for name in ['deps','vendor','asr_model','downloads']:
        if not (root/name).exists():(root/name).symlink_to(parent/name,target_is_directory=True)
    for name in ['evaluate.py','aggregate.py','pipeline.py']:
        shutil.copyfile(parent/'commands'/name,root/'commands'/name)
    rows=[json.loads(s) for s in (parent/'manifests/all_samples.jsonl').read_text().splitlines()]
    selected=[r for r in rows if r['dataset']=='fleurs_en_us']
    assert len(selected)==647
    target_rms=10**(-25/20)
    gains=[]
    for row in selected:
        original=parent/row['audio'];x,sr=sf.read(original,dtype='float32',always_2d=True)
        assert sr==16000 and x.shape[1]==1
        x=x[:,0];rms=float(np.sqrt(np.mean(x.astype(np.float64)**2)));peak=float(np.max(np.abs(x)))
        assert rms>0 and peak>0
        gain=min(target_rms/rms,0.99/peak)
        y=(x*gain).astype(np.float32)
        p=root/'data/fleurs_en_us'/(row['sample_id']+'.wav');sf.write(p,y,16000,subtype='FLOAT')
        row['parent_audio_sha256']=row['audio_sha256'];row['audio_sha256']=sha(p)
        row['audio']=str(p.relative_to(root));row['control']=dict(target_rms_dbfs=-25,
            gain=float(gain),gain_db=float(20*np.log10(gain)),original_rms_dbfs=float(20*np.log10(rms)),
            achieved_rms_dbfs=float(20*np.log10(np.sqrt(np.mean(y.astype(np.float64)**2)))),
            peak_capped=bool(target_rms/rms>0.99/peak))
        assert np.isfinite(y).all() and np.max(np.abs(y))<1
        gains.append(row['control'])
    path=root/'manifests/all_samples.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    spec=dict(status='fixed_before_control_inference',type='post_hoc_input_level_sensitivity',
        rationale='Original FLEURS median RMS around -60 dBFS vs LibriSpeech reference around -25 dBFS; original objective decline observed before defining this diagnostic.',
        target_rms_dbfs=-25,peak_ceiling=0.99,selection='all 647 FLEURS en_us test recordings; no performance-based exclusions',
        model='same frozen Base/LCA checkpoints; no training',
        interpretation='Supplement to raw-level primary experiment; cannot replace it or be called a preregistered independent replication.',
        parent_manifest_sha256=sha(parent/'manifests/all_samples.jsonl'),manifest_sha256=sha(path),
        recordings=len(selected),peak_capped=sum(g['peak_capped'] for g in gains),
        gain_db_quantiles=list(map(float,np.quantile([g['gain_db'] for g in gains],[0,.5,1]))))
    (root/'CONTROL_PROTOCOL.json').write_text(json.dumps(spec,indent=2))
    print(json.dumps(spec,indent=2))


if __name__=='__main__':main()
