"""Peek at the latest TensorBoard scalars for the exp13 run (ground-truth step/dev-mel)."""
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
events = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "logs", "events*")))
if not events:
    print("NO EVENT FILES")
    sys.exit(1)
f = events[-1]
ea = EventAccumulator(f, size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]
print("EVENT FILE:", os.path.basename(f))
print("TAGS:", tags)
for t in tags:
    s = ea.Scalars(t)
    if not s:
        continue
    last = s[-1]
    # for dev-type tags, also show the min (best) so far
    extra = ""
    if "dev" in t.lower() or "mel" in t.lower():
        vals = [(x.step, x.value) for x in s]
        bstep, bval = min(vals, key=lambda kv: kv[1])
        extra = f"  | best={bval:.5f}@{bstep}"
    print(f"{t}: last step={last.step} val={last.value:.5f} (n={len(s)}){extra}")
