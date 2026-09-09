"""exp13 heartbeat v4: report real TB step + dev/mel + GPU + liveness, and detect a TRUE
stall by step non-movement across ticks (not by TB-file staleness, which a normal ~25min
dev eval triggers). Persists last step to a tmp file to compute delta between ticks.

Run once per heartbeat tick: conda run -n speechtokenizer python <this file>
Exit/print is one OK/WARN/ALERT line.
"""
import glob
import os
import subprocess

RUN = r"h:\H-CODE\speechtokenizer\output\experiments\exp13_handdesigned_encoder_distill30_20260611_seed42"
STATE = os.path.join(RUN, "commands", "_hb_last_step.txt")
STUCK = os.path.join(RUN, "commands", "_hb_stuck_count.txt")
# A normal dev eval (~25min) freezes train/mel-error step for up to ~3 ticks (10min each).
# Only flag a stall after this many consecutive ticks with NO step movement.
STUCK_TICKS_ALERT = 4  # ~40min > 25min eval + one train interval


def gpu_and_alive():
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20).stdout.strip().splitlines()[0]
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=20).stdout
        return gpu, ("speechtokenizer" in apps)
    except Exception as e:
        return f"gpu-query-failed:{e}", False


def tb_step_and_dev():
    try:
        from tensorboard.backend.event_processing import event_file_loader
        from tensorboard.compat.proto import event_pb2
        from tensorboard.util import tensor_util
        events = sorted(glob.glob(os.path.join(RUN, "checkpoints", "logs", "events*")),
                        key=os.path.getmtime)
        if not events:
            return None, None, None
        # scan newest file for last train/mel error step and last dev/mel error
        last_step, last_train_mel, dev_last, dev_best = None, None, None, None
        for ev_path in events[-2:]:  # last two files (resume created a new one)
            loader = event_file_loader.EventFileLoader(ev_path)
            for raw in loader.Load():
                ev = raw if isinstance(raw, event_pb2.Event) else event_pb2.Event.FromString(raw)
                if not ev.HasField("summary"):
                    continue
                for val in ev.summary.value:
                    if not val.HasField("tensor"):
                        continue
                    try:
                        v = float(tensor_util.make_ndarray(val.tensor).reshape(-1)[0])
                    except Exception:
                        continue
                    if val.tag == "train/mel error":
                        last_step, last_train_mel = ev.step, v
                    elif val.tag == "dev/mel error":
                        dev_last = (ev.step, v)
                        if dev_best is None or v < dev_best[1]:
                            dev_best = (ev.step, v)
        return last_step, last_train_mel, (dev_last, dev_best)
    except Exception as e:
        return None, None, f"tb-err:{e}"


def main():
    gpu, alive = gpu_and_alive()
    step, train_mel, dev = tb_step_and_dev()
    prev = None
    if os.path.exists(STATE):
        try:
            prev = int(open(STATE).read().strip())
        except Exception:
            prev = None
    if step is not None:
        open(STATE, "w").write(str(step))

    # track consecutive stuck ticks
    stuck = 0
    if os.path.exists(STUCK):
        try:
            stuck = int(open(STUCK).read().strip())
        except Exception:
            stuck = 0
    if step is not None and prev is not None and step == prev:
        stuck += 1
    else:
        stuck = 0
    open(STUCK, "w").write(str(stuck))

    dev_str = ""
    if isinstance(dev, tuple) and dev[0]:
        (ds, dv), (bs, bv) = dev[0], dev[1]
        dev_str = f" dev_mel={dv:.3f}@{ds}(best {bv:.3f}@{bs})"

    if not alive:
        print(f"ALERT exp13 DEAD: python not on GPU. tb_step={step} gpu=[{gpu}]")
    elif stuck >= STUCK_TICKS_ALERT:
        print(f"WARN exp13: TB step STUCK at {step} for {stuck} ticks (~{stuck*10}min, "
              f">eval time) — verify real stall. train_mel={train_mel} gpu=[{gpu}] alive=yes")
    else:
        delta = f"(+{step-prev})" if (step is not None and prev is not None) else ""
        mel = f"{train_mel:.3f}" if train_mel is not None else "?"
        print(f"exp13 OK: step={step}{delta} train_mel={mel}{dev_str} gpu=[{gpu}] "
              f"alive=yes stuck_ticks={stuck}")


if __name__ == "__main__":
    main()
