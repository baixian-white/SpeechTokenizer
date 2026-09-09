"""One-time: dump exp13 dev/mel + key train-loss trajectory (last N points) to diagnose
whether the dev/mel rise (0.52->1.56->2.08) is GAN instability or eval noise.
Reads newest TB event file by streaming (no EventAccumulator)."""
import glob
import os
from collections import defaultdict

from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util

RUN = r"h:\H-CODE\speechtokenizer\output\experiments\exp13_handdesigned_encoder_distill30_20260611_seed42"
TAGS = ["dev/mel error", "train/mel error", "train/generator loss",
        "train/discriminators loss", "train/adversarial loss", "train/quantizer loss",
        "train/distillation loss"]

events = sorted(glob.glob(os.path.join(RUN, "checkpoints", "logs", "events*")), key=os.path.getmtime)
store = defaultdict(dict)
for ev_path in events[-2:]:
    for raw in event_file_loader.EventFileLoader(ev_path).Load():
        ev = raw if isinstance(raw, event_pb2.Event) else event_pb2.Event.FromString(raw)
        if not ev.HasField("summary"):
            continue
        for val in ev.summary.value:
            if val.tag in TAGS and val.HasField("tensor"):
                try:
                    store[val.tag][ev.step] = float(tensor_util.make_ndarray(val.tensor).reshape(-1)[0])
                except Exception:
                    pass

print("=== dev/mel error (all points) ===")
dev = store.get("dev/mel error", {})
for s in sorted(dev):
    print(f"  step {s}: {dev[s]:.4f}")

print("\n=== train losses (last 12 logged points) ===")
for tag in ["train/mel error", "train/generator loss", "train/discriminators loss",
            "train/adversarial loss", "train/quantizer loss", "train/distillation loss"]:
    d = store.get(tag, {})
    if not d:
        continue
    last = sorted(d)[-12:]
    vals = " ".join(f"{d[s]:.3f}" for s in last)
    print(f"  {tag}: steps {last[0]}..{last[-1]}: {vals}")
