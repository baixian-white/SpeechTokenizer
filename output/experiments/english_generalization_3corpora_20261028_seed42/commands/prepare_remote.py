"""Download pinned external test sets and ASR assets; freeze manifest before inference."""
import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

FLEURS = '70bb2e84b976b7e960aa89f1c648e09c59f894dd'
TED = '8083dbe64f82e17b6be070a02b9239e9b847b6ce'
ASR = 'd1d751a5f8271d482d14ca55d9e2deeebbae577f'
SIZES = {'fleurs_test.tar.gz':289851356, 'tedlium_test.parquet':351931487, 'model.bin':483545366}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def download(job):
    url, path = job
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_name(path.name + '.download.json')
    if path.exists() and marker.exists():
        meta = json.loads(marker.read_text())
        if meta['url'] == url and meta['sha256'] == sha(path):
            return meta
    part = path.with_name(path.name + '.partial')
    if path.name in SIZES:
        total = SIZES[path.name]
        prefix = part.stat().st_size if part.exists() else 0
        assert prefix <= total
        chunkdir = path.with_name(path.name+'.ranges')
        chunkdir.mkdir(exist_ok=True)
        ranges = [(start,min(start+8*1024*1024,total)-1) for start in range(prefix,total,8*1024*1024)]
        def fetch_range(bounds):
            start,end=bounds
            chunk=chunkdir/f'{start}-{end}'
            if chunk.exists() and chunk.stat().st_size==end-start+1:
                return chunk
            for attempt in range(5):
                try:
                    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(req,timeout=90) as r:
                        assert r.status==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{total}', (r.status,r.headers.get('Content-Range'))
                        with chunk.open('wb') as out:
                            shutil.copyfileobj(r,out,length=1024*1024)
                    assert chunk.stat().st_size==end-start+1
                    print(json.dumps({'range_complete':path.name,'start':start,'end':end}),flush=True)
                    return chunk
                except Exception:
                    if attempt==4:raise
                    time.sleep(attempt+1)
        with ThreadPoolExecutor(max_workers=6) as pool:
            chunks=list(pool.map(fetch_range,ranges))
        assembled=path.with_name(path.name+'.assembling')
        with assembled.open('wb') as out:
            if prefix:
                with part.open('rb') as f:shutil.copyfileobj(f,out,length=1024*1024)
            for chunk in chunks:
                with chunk.open('rb') as f:shutil.copyfileobj(f,out,length=1024*1024)
        assert assembled.stat().st_size==total
        assembled.replace(path)
        meta=dict(url=url,path=str(path),bytes=total,sha256=sha(path))
        marker.write_text(json.dumps(meta,indent=2))
        print(json.dumps({'download_complete':path.name,'bytes':total}),flush=True)
        return meta
    for attempt in range(5):
        try:
            n = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={'Range':f'bytes={n}-'} if n else {})
            with urllib.request.urlopen(req, timeout=90) as r:
                append = n > 0 and r.status == 206
                expected = int(r.headers.get('Content-Length', 0))
                start = n if append else 0
                print(json.dumps({'download':path.name,'start':start,'remaining':expected}), flush=True)
                with part.open('ab' if append else 'wb') as out:
                    while True:
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                if expected and part.stat().st_size != start + expected:
                    raise IOError('incomplete Content-Length')
            part.replace(path)
            meta = dict(url=url, path=str(path), bytes=path.stat().st_size, sha256=sha(path))
            marker.write_text(json.dumps(meta, indent=2))
            print(json.dumps({'download_complete':path.name,'bytes':meta['bytes']}), flush=True)
            return meta
        except Exception as exc:
            print(json.dumps({'retry':attempt+1,'file':path.name,'error':str(exc)}), flush=True)
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def audio_info(path):
    i = sf.info(str(path))
    return dict(audio_sha256=sha(path), source_sample_rate=i.samplerate,
                source_channels=i.channels, duration_sec=i.duration)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--normalizer-revision', required=True)
    a = ap.parse_args()
    root = a.run.resolve()
    for p in ('downloads','manifests','data','asr_model','vendor/normalizers','logs','reports','metrics'):
        (root/p).mkdir(parents=True, exist_ok=True)
    source = [
        (f'https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS}/data/en_us/audio/test.tar.gz',root/'downloads/fleurs_test.tar.gz'),
        (f'https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS}/data/en_us/test.tsv',root/'downloads/fleurs_test.tsv'),
        (f'https://huggingface.co/datasets/sanchit-gandhi/tedlium-data/resolve/{TED}/data/test-00000-of-00001-05350587dc944f81.parquet',root/'downloads/tedlium_test.parquet'),
        ('https://huggingface.co/datasets/distil-whisper/tedlium/resolve/31aa998c1cbc36063c5bd5c0ca87b082d443da02/tedlium.py',root/'downloads/tedlium_source_loader.py'),
    ]
    for name in ('config.json','model.bin','tokenizer.json','vocabulary.txt','README.md'):
        source.append((f'https://huggingface.co/Systran/faster-whisper-small.en/resolve/{ASR}/{name}',root/'asr_model'/name))
    for name in ('basic.py','english.py','english.json'):
        source.append((f'https://raw.githubusercontent.com/openai/whisper/{a.normalizer_revision}/whisper/normalizers/{name}',root/'vendor/normalizers'/name))
    source.append((f'https://raw.githubusercontent.com/openai/whisper/{a.normalizer_revision}/LICENSE',root/'vendor/normalizers/LICENSE'))
    with ThreadPoolExecutor(max_workers=3) as pool:
        provenance = list(pool.map(download, source))
    (root/'vendor/normalizers/__init__.py').write_text('from .english import EnglishTextNormalizer\n')
    (root/'manifests/downloads.json').write_text(json.dumps(provenance, indent=2))
    rows = []
    exclusions = []
    with tarfile.open(root/'downloads/fleurs_test.tar.gz','r:gz') as tar:
        for m in tar:
            if m.isfile() and Path(m.name).suffix == '.wav':
                path = root/'data/fleurs_en_us'/Path(m.name).name
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, path.open('wb') as out:
                    while True:
                        b = src.read(1024*1024)
                        if not b:
                            break
                        out.write(b)
    with (root/'downloads/fleurs_test.tsv').open(encoding='utf-8',newline='') as f:
        # FLEURS TSV uses literal quotes in transcription fields, not CSV quoting.
        for index, r in enumerate(csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE)):
            assert len(r) == 7, (index,len(r))
            prompt, filename, raw, normalized, chars, num_samples, gender = r
            path = root/'data/fleurs_en_us'/filename
            info = audio_info(path)
            assert info['source_sample_rate'] == 16000 and sf.info(str(path)).frames == int(num_samples)
            rows.append(dict(dataset='fleurs_en_us', sample_id=Path(filename).stem,
                cluster_id=str(prompt), cluster_type='prompt', transcript=raw,
                source_normalized_transcript=normalized, gender=gender, source_row=index,
                audio=str(path.relative_to(root)), source_revision=FLEURS, **info))
    assert len(rows) == 647
    n_ted = 0
    ted_dir = root/'data/tedlium'
    ted_dir.mkdir(parents=True, exist_ok=True)
    for batch in pq.ParquetFile(root/'downloads/tedlium_test.parquet').iter_batches(batch_size=32):
        for r in batch.to_pylist():
            index = n_ted
            n_ted += 1
            text = r['text'].strip()
            if not text or text == 'ignore_time_segment_in_scoring':
                exclusions.append(dict(dataset='tedlium', source_row=index, source_id=r['id'], reason=text or 'empty_reference'))
                continue
            sid = f'ted_{index:05d}'
            raw = r['audio']['bytes']
            assert raw, (sid,r['audio'])
            # Decode the embedded source and preserve samples losslessly as float WAV.
            audio, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
            path = ted_dir/(sid+'.wav')
            sf.write(path,audio,sr,subtype='FLOAT')
            rows.append(dict(dataset='tedlium', sample_id=sid, cluster_id=str(r['file']),
                cluster_type='talk', speaker_id=r['speaker_id'], transcript=text,
                source_row=index, source_id=r['id'], source_file=r['file'],
                audio=str(path.relative_to(root)), source_revision=TED,
                embedded_audio_sha256=hashlib.sha256(raw).hexdigest(), **audio_info(path)))
    assert n_ted == 1469, n_ted
    local = root/'manifests/local_samples.jsonl'
    if not local.exists():
        raise RuntimeError('Selected local audio/manifest must be transferred before finalizing')
    rows += [json.loads(s) for s in local.read_text().splitlines() if s.strip()]
    keys = [(r['dataset'],r['sample_id']) for r in rows]
    assert len(keys) == len(set(keys)), 'Duplicate sample IDs'
    for r in rows:
        p = root/r['audio']
        assert p.is_file() and sha(p) == r['audio_sha256'], str(p)
    rows.sort(key=lambda r:(r['dataset'],r['sample_id']))
    manifest = root/'manifests/all_samples.jsonl'
    content = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if manifest.exists():
        assert manifest.read_text() == content, 'Frozen manifest cannot be changed in place'
    else:
        manifest.write_text(content)
    (root/'manifests/exclusions.json').write_text(json.dumps(exclusions,indent=2))
    summary = {'count':len(rows),'manifest_sha256':sha(manifest), 'normalizer_revision':a.normalizer_revision,
               'ted_raw_rows':n_ted,'exclusions':len(exclusions),'datasets':{}}
    for ds in sorted({r['dataset'] for r in rows}):
        rs = [r for r in rows if r['dataset']==ds]
        summary['datasets'][ds] = dict(count=len(rs), clusters=len({r['cluster_id'] for r in rs}),
            cluster_type=rs[0]['cluster_type'], duration_hours=sum(r['duration_sec'] for r in rs)/3600,
            missing_transcripts=sum(not r.get('transcript') for r in rs),
            min_duration=min(r['duration_sec'] for r in rs),max_duration=max(r['duration_sec'] for r in rs))
    (root/'manifests/summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
