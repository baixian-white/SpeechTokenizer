"""One-line heartbeat for exp13: real step + train/dev mel from TensorBoard (live-flushed),
plus GPU + process liveness. Used by the persistent Monitor (buffered train.log lags ~900 steps)."""
import glob
import os
import subprocess

RUN = r"h:\H-CODE\speechtokenizer\output\experiments\exp13_handdesigned_encoder_distill30_20260611_seed42"


def gpu_line():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()[0]
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        alive = "speechtokenizer" in apps
        return out, alive
    except Exception as e:
        return f"gpu-query-failed:{e}", False


def tb_state():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = sorted(glob.glob(os.path.join(RUN, "checkpoints", "logs", "events*")))
        if not events:
            return "no-tb-events"
        ea = EventAccumulator(events[-1], size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        parts = []
        if "train/mel error" in tags:
            s = ea.Scalars("train/mel error")
            parts.append(f"step={s[-1].step} train_mel={s[-1].value:.3f}")
        if "dev/mel error" in tags:
            s = ea.Scalars("dev/mel error")
            best = min(s, key=lambda x: x.value)
            parts.append(f"dev_mel={s[-1].value:.3f}@{s[-1].step}(best {best.value:.3f}@{best.step})")
        return " ".join(parts) if parts else "no-scalars-yet"
    except Exception as e:
        return f"tb-read-failed:{e}"


gpu, alive = gpu_line()
tb = tb_state()
status = "ALERT-DEAD" if not alive else "OK"
print(f"exp13 {status}: {tb} | gpu=[{gpu}] | python {'alive' if alive else 'NOT on GPU'}")
