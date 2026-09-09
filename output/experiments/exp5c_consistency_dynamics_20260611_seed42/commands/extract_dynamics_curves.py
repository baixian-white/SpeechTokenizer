"""exp5c step 1: extract LCA factorial training-dynamics curves from TensorBoard.

Read-only. For each variant V0-V4, dumps per-step scalar series to CSV and computes
the per-step "robustness gap" = comm_mel(perturbed channel) - comm_mel(clean) at L3,
which traces how perturbation robustness evolves during training. Also pulls
train/loss_consistency + train/lambda_consistency_used (nonzero only for V4) and
dev/full_depth_mel_error (clean-quality "do not regress" guardrail).

Run: conda run -n speechtokenizer python <this file>
"""
import csv
import glob
import os
from collections import defaultdict

from tensorboard.compat.proto import event_pb2
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.util import tensor_util

EXP5 = r"h:\H-CODE\speechtokenizer\output\experiments\exp5_lca_component_factorial_20260603_seed42"
EXP3_V4 = r"h:\H-CODE\speechtokenizer\output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42"
OUT = r"h:\H-CODE\speechtokenizer\output\experiments\exp5c_consistency_dynamics_20260611_seed42"

VARIANT_LOGDIRS = {
    "V0": os.path.join(EXP5, "runs", "V0_full_depth_clean_control", "checkpoints", "logs"),
    "V1": os.path.join(EXP5, "runs", "V1_random_l_only", "checkpoints", "logs"),
    "V2": os.path.join(EXP5, "runs", "V2_channelsim_only", "checkpoints", "logs"),
    "V3": os.path.join(EXP5, "runs", "V3_random_l_channelsim", "checkpoints", "logs"),
    "V4": os.path.join(EXP3_V4, "checkpoints", "logs"),
}

# tags we care about for the dynamics story (extracted if present)
KEY_TRAIN_TAGS = ["train/loss_consistency", "train/lambda_consistency_used",
                  "train/mel loss full", "train/mel loss comm", "train/mel error"]
PERTURB_CHANNELS = ["dropout-mid", "dropout-high", "substitution-mid", "substitution-high"]


def load_variant(logdir):
    """Stream ALL event files (mtime order) once; return {tag: {step: value}}.
    Later files override earlier on the same step — needed for interrupted+resumed
    runs like V4 (exp3 v2) which has a pre-interruption and a post-resume event file."""
    events = sorted(glob.glob(os.path.join(logdir, "events*")), key=os.path.getmtime)
    if not events:
        return None
    store = defaultdict(dict)
    for ev_path in events:
        loader = event_file_loader.EventFileLoader(ev_path)
        for raw in loader.Load():
            # raw is already a parsed Event in recent TB; guard for bytes too
            ev = raw if isinstance(raw, event_pb2.Event) else event_pb2.Event.FromString(raw)
            if not ev.HasField("summary"):
                continue
            step = ev.step
            for val in ev.summary.value:
                v = None
                if val.HasField("simple_value"):
                    v = val.simple_value
                elif val.HasField("tensor"):
                    # TB2 stores scalars as 0-d tensors; skip text/hparams summaries
                    try:
                        arr = tensor_util.make_ndarray(val.tensor)
                        if arr.size == 1 and arr.dtype.kind in "fiu":
                            v = float(arr.reshape(-1)[0])
                    except Exception:
                        v = None
                if v is not None:
                    store[val.tag][step] = v
    return store


def series(store, tag):
    """Return {step: value} dict for a tag, or {} if absent."""
    return store.get(tag, {})


def main():
    os.makedirs(os.path.join(OUT, "metrics"), exist_ok=True)
    long_rows = []          # variant, tag, step, value
    gap_rows = []           # variant, L, channel, step, comm_mel_clean, comm_mel_pert, gap
    summary_rows = []       # variant, has_consistency, full_mel tags etc.

    for vname, logdir in VARIANT_LOGDIRS.items():
        store = load_variant(logdir)
        if store is None:
            print(f"{vname}: NO EVENT FILE at {logdir}")
            continue
        tags = list(store.keys())
        # 1) long-format dump of all dev/* tags + key train tags
        dump_tags = [t for t in tags if t.startswith("dev/")] + \
                    [t for t in KEY_TRAIN_TAGS if t in tags]
        for t in dump_tags:
            for step, val in series(store, t).items():
                long_rows.append([vname, t, step, val])

        # 2) robustness-gap dynamics: for each L and perturbed channel that exists,
        #    gap(step) = comm_mel[L, channel](step) - comm_mel[L, clean](step)
        for L in (1, 2, 3):
            clean_tag = f"dev/comm_mel/L{L}_clean"
            clean = series(store, clean_tag)
            if not clean:
                continue
            for ch in PERTURB_CHANNELS:
                pert = series(store, f"dev/comm_mel/L{L}_{ch}")
                if not pert:
                    continue
                for step in sorted(set(clean) & set(pert)):
                    gap_rows.append([vname, L, ch, step, clean[step], pert[step],
                                     pert[step] - clean[step]])

        # 3) per-variant summary
        consist = series(store, "train/loss_consistency")
        lam = series(store, "train/lambda_consistency_used")
        consist_max = max(consist.values()) if consist else 0.0
        lam_max = max(lam.values()) if lam else 0.0
        full_mel = series(store, "dev/full_depth_mel_error") or series(store, "dev/mel error")
        full_best = min(full_mel.values()) if full_mel else float("nan")
        full_last = full_mel[max(full_mel)] if full_mel else float("nan")
        summary_rows.append([vname, lam_max > 0, round(consist_max, 5), round(lam_max, 3),
                             round(full_best, 4), round(full_last, 4),
                             max(full_mel) if full_mel else 0])
        print(f"{vname}: consistency_active={lam_max>0} (lambda_max={lam_max}) "
              f"full_mel best={full_best:.4f} last={full_last:.4f} "
              f"perturb_channels_present={any(f'dev/comm_mel/L3_{c}' in tags for c in PERTURB_CHANNELS)}")

    m = os.path.join(OUT, "metrics")
    with open(os.path.join(m, "dynamics_long.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "tag", "step", "value"]); w.writerows(long_rows)
    with open(os.path.join(m, "robustness_gap_dynamics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "L", "channel", "step", "comm_mel_clean", "comm_mel_pert", "gap"])
        w.writerows(gap_rows)
    with open(os.path.join(m, "variant_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "consistency_active", "loss_consistency_max",
                    "lambda_consistency_max", "full_mel_best", "full_mel_last", "last_dev_step"])
        w.writerows(summary_rows)
    print(f"\nwrote: dynamics_long.csv ({len(long_rows)} rows), "
          f"robustness_gap_dynamics.csv ({len(gap_rows)} rows), "
          f"variant_summary.csv ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
