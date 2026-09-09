"""Probe: what fields do the V0 event summaries actually use? (simple_value vs tensor)"""
import glob
import os

from tensorboard.compat.proto import event_pb2
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.util import tensor_util

LOGDIR = r"h:\H-CODE\speechtokenizer\output\experiments\exp5_lca_component_factorial_20260603_seed42\runs\V0_full_depth_clean_control\checkpoints\logs"
events = sorted(glob.glob(os.path.join(LOGDIR, "events*")), key=os.path.getsize)
loader = event_file_loader.EventFileLoader(events[-1])
seen = 0
for raw in loader.Load():
    ev = raw if isinstance(raw, event_pb2.Event) else event_pb2.Event.FromString(raw)
    if not ev.HasField("summary"):
        continue
    for val in ev.summary.value:
        has_simple = val.HasField("simple_value")
        has_tensor = val.HasField("tensor")
        tval = None
        if has_tensor:
            try:
                tval = float(tensor_util.make_ndarray(val.tensor))
            except Exception as e:
                tval = f"err:{e}"
        print(f"tag={val.tag!r} step={ev.step} simple={has_simple} tensor={has_tensor} tval={tval}")
        seen += 1
    if seen >= 12:
        break
