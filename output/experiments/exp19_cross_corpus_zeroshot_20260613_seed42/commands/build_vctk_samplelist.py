"""exp19: build a speaker-stratified VCTK sample list (300 utterances, seed=42).

Scans the extracted VCTK dir for audio files, groups by speaker (parent dir name like
pXXX), and samples evenly across speakers to 300 total. Writes one absolute audio path
per line (the eval reader `read_sample_rows` takes the path before any tab).

VCTK 0.92 layout is typically: <root>/wav48_silence_trimmed/pXXX/pXXX_NNN_micN.flac
but this script auto-discovers the audio dir and extension so it survives layout surprises.

Usage: conda run -n speechtokenizer python <this file> --vctk-root data/VCTK --out <list.txt> [--n 300] [--seed 42]
"""
import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

AUDIO_EXTS = (".flac", ".wav")


def find_audio_files(root):
    files = []
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            if n.lower().endswith(AUDIO_EXTS):
                files.append(os.path.join(dirpath, n))
    return files


def speaker_of(path):
    # VCTK filenames start with speaker id pXXX (or sXX); use leading token of filename
    stem = Path(path).stem
    return stem.split("_")[0] if "_" in stem else Path(path).parent.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vctk-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mic", default="mic1", help="prefer files containing this token (VCTK has mic1/mic2); set empty to disable")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    allf = find_audio_files(args.vctk_root)
    print(f"found {len(allf)} audio files under {args.vctk_root}")
    if not allf:
        raise SystemExit("no audio files found — check --vctk-root and extraction")

    # prefer a single mic to avoid near-duplicate recordings of same utterance
    if args.mic:
        miced = [f for f in allf if args.mic in Path(f).name]
        if miced:
            print(f"filtered to {len(miced)} files containing '{args.mic}' (was {len(allf)})")
            allf = miced

    by_spk = defaultdict(list)
    for f in allf:
        by_spk[speaker_of(f)].append(f)
    speakers = sorted(by_spk)
    for s in speakers:
        by_spk[s].sort()
        rng.shuffle(by_spk[s])
    print(f"{len(speakers)} speakers")

    # round-robin across speakers until we have n
    chosen = []
    i = 0
    while len(chosen) < args.n:
        progressed = False
        for s in speakers:
            if i < len(by_spk[s]):
                chosen.append(by_spk[s][i])
                progressed = True
                if len(chosen) >= args.n:
                    break
        if not progressed:
            break
        i += 1
    chosen = chosen[: args.n]
    # sample_id = filename stem; check uniqueness
    stems = [Path(c).stem for c in chosen]
    assert len(set(stems)) == len(stems), "duplicate sample_ids — would collide in eval CSV"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in chosen:
            f.write(os.path.abspath(c) + "\n")
    n_spk_used = len({speaker_of(c) for c in chosen})
    print(f"wrote {len(chosen)} utterances from {n_spk_used} speakers to {args.out}")


if __name__ == "__main__":
    main()
