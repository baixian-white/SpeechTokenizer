import soundfile as sf, glob, statistics
d = r"output/experiments/exp12_baseline_comparison_test300_20260610_seed42/runs/test-clean_300/samples/original"
durs = []
for f in sorted(glob.glob(d + "/*.wav")):
    info = sf.info(f); durs.append(info.frames / info.samplerate)
print(f"n={len(durs)} min={min(durs):.1f} max={max(durs):.1f} median={statistics.median(durs):.1f} mean={statistics.mean(durs):.1f}")
# how many in 3-8s sweet spot for listening tests
good = [x for x in durs if 3 <= x <= 8]
print(f"in 3-8s range: {len(good)}")
