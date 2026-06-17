from pathlib import Path
import soundfile as sf
root = Path('data/SpeechPretrain/LibriSpeech')
out = Path('output/experiments/exp7_librispeech_test_full_clean_20260606/artifacts')
out.mkdir(parents=True, exist_ok=True)
for split in ['test-clean','test-other']:
    files = sorted((root/split).rglob('*.flac'))
    rows=[]; total_dur=0.0
    for p in files:
        rows.append(str(p.resolve()))
        info=sf.info(str(p)); total_dur += info.frames/info.samplerate
    (out/f'{split}_all_files.txt').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    print(split, 'count', len(rows), 'hours', total_dur/3600)
