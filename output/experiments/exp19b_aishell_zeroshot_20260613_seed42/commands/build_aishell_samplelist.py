"""exp19b: build speaker-stratified AISHELL test sample list (300 utts, seed=42).
AISHELL layout: <root>/wav/test/SXXXX/BAC009SXXXX_WNNN.wav  (speaker = SXXXX dir).
Writes one absolute wav path per line; sample_id = filename stem (= utt id, matches transcript).
"""
import argparse
import os
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", required=True, help="dir containing SXXXX speaker subdirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    wavs = []
    for dp, _d, names in os.walk(args.test_root):
        for n in names:
            if n.lower().endswith(".wav"):
                wavs.append(os.path.join(dp, n))
    print(f"found {len(wavs)} wavs under {args.test_root}")
    if not wavs:
        raise SystemExit("no wavs — check --test-root (inner tars extracted?)")

    by_spk = defaultdict(list)
    for w in wavs:
        by_spk[Path(w).parent.name].append(w)  # SXXXX dir
    speakers = sorted(by_spk)
    for s in speakers:
        by_spk[s].sort(); rng.shuffle(by_spk[s])
    print(f"{len(speakers)} speakers")

    chosen, i = [], 0
    while len(chosen) < args.n:
        prog = False
        for s in speakers:
            if i < len(by_spk[s]):
                chosen.append(by_spk[s][i]); prog = True
                if len(chosen) >= args.n:
                    break
        if not prog:
            break
        i += 1
    chosen = chosen[: args.n]
    stems = [Path(c).stem for c in chosen]
    assert len(set(stems)) == len(stems), "dup sample_ids"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in chosen:
            f.write(os.path.abspath(c) + "\n")
    print(f"wrote {len(chosen)} utts from {len({Path(c).parent.name for c in chosen})} speakers to {args.out}")


if __name__ == "__main__":
    main()
