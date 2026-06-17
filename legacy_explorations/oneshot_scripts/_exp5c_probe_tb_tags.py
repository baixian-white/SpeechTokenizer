"""Read-only: dump TensorBoard scalar tags + step ranges for V0-V4 LCA factorial variants.
Used to design exp5c (consistency dynamics) before authoring the curve extractor.
Run with: conda run -n speechtokenizer python <this file>
"""
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EXP5 = r"h:\H-CODE\speechtokenizer\output\experiments\exp5_lca_component_factorial_20260603_seed42"
EXP3_V4 = r"h:\H-CODE\speechtokenizer\output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42"

VARIANTS = {
    "V0": os.path.join(EXP5, "runs", "V0_full_depth_clean_control", "checkpoints", "logs"),
    "V1": os.path.join(EXP5, "runs", "V1_random_l_only", "checkpoints", "logs"),
    "V2": os.path.join(EXP5, "runs", "V2_channelsim_only", "checkpoints", "logs"),
    "V3": os.path.join(EXP5, "runs", "V3_random_l_channelsim", "checkpoints", "logs"),
    "V4": os.path.join(EXP3_V4, "checkpoints", "logs"),
}


def main():
    for name, logdir in VARIANTS.items():
        print(f"\n========== {name} ==========")
        print("logdir:", logdir, "exists:", os.path.isdir(logdir))
        if not os.path.isdir(logdir):
            continue
        events = sorted(glob.glob(os.path.join(logdir, "events*")))
        print(f"event files ({len(events)}):")
        for e in events:
            print("   ", os.path.basename(e), f"{os.path.getsize(e)} bytes")
        if not events:
            continue
        ea = EventAccumulator(events[-1], size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        print(f"scalar tags ({len(tags)}):")
        for t in sorted(tags):
            s = ea.Scalars(t)
            if not s:
                print(f"    {t}: (empty)")
                continue
            steps = [x.step for x in s]
            vals = [x.value for x in s]
            print(f"    {t}: n={len(s)} step[{min(steps)}..{max(steps)}] "
                  f"last={vals[-1]:.4f} min={min(vals):.4f} max={max(vals):.4f}")


if __name__ == "__main__":
    main()
