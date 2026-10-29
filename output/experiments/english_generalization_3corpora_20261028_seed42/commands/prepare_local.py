"""Freeze seeded VCTK/LibriSpeech selections and package only selected originals."""
import argparse
import hashlib
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path
import soundfile as sf


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args()
    dest = a.run / 'manifests'
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    selected = []
    vctk = a.project / 'data/VCTK'
    for speaker in sorted((vctk / 'wav48_silence_trimmed').iterdir()):
        if not speaker.is_dir():
            continue
        candidates = sorted(speaker.glob('*_mic1.flac'))
        rng.shuffle(candidates)
        assert len(candidates) >= 20, speaker
        for p in candidates[:20]:
            transcript = vctk / 'txt' / speaker.name / (p.stem.replace('_mic1', '') + '.txt')
            selected.append(dict(dataset='vctk', sample_id=p.stem, cluster_id=speaker.name,
                cluster_type='speaker', source=p,
                transcript=transcript.read_text(encoding='utf-8').strip() if transcript.exists() else None,
                transcript_missing_reason=None if transcript.exists() else 'no_matching_reference_file'))
    assert len(selected) == 2200
    for split in ('test-clean', 'test-other'):
        root = a.project / 'data/SpeechPretrain/LibriSpeech' / split
        texts = {}
        for p in sorted(root.glob('*/*/*.trans.txt')):
            for line in p.read_text(encoding='utf-8').splitlines():
                uid, text = line.split(' ', 1)
                texts[uid] = text
        groups = defaultdict(list)
        for p in sorted(root.glob('*/*/*.flac')):
            groups[p.stem.split('-')[0]].append(p)
        rng = random.Random(42)
        for key in sorted(groups):
            rng.shuffle(groups[key])
        pool = []
        i = 0
        while len(pool) < 300:
            for key in sorted(groups):
                if i < len(groups[key]) and len(pool) < 300:
                    pool.append(groups[key][i])
            i += 1
        for p in pool:
            selected.append(dict(dataset='librispeech_' + split.replace('-', '_'), sample_id=p.stem,
                cluster_id=p.stem.split('-')[0], cluster_type='speaker', source=p,
                transcript=texts[p.stem], transcript_missing_reason=None))
    manifest = []
    archive = a.run / 'selected_local_audio.tar'
    with tarfile.open(archive, 'w') as tar:
        for row in selected:
            p = row.pop('source')
            info = sf.info(str(p))
            audio = 'data/' + row['dataset'] + '/' + p.name
            row.update(audio=audio, source_path=str(p.relative_to(a.project)).replace('\\','/'),
                audio_sha256=sha(p), source_sample_rate=info.samplerate, source_channels=info.channels,
                duration_sec=info.duration)
            tar.add(p, arcname=audio, recursive=False)
            manifest.append(row)
    path = dest / 'local_samples.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in manifest), encoding='utf-8')
    summary = {'seed':42, 'manifest_sha256':sha(path), 'archive_sha256':sha(archive),
        'archive_bytes':archive.stat().st_size, 'datasets':{}}
    for ds in sorted({r['dataset'] for r in manifest}):
        rows = [r for r in manifest if r['dataset']==ds]
        summary['datasets'][ds] = dict(count=len(rows), clusters=len({r['cluster_id'] for r in rows}),
            missing_transcripts=sum(r['transcript'] is None for r in rows),
            duration_hours=sum(r['duration_sec'] for r in rows)/3600)
    (dest / 'local_selection.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
