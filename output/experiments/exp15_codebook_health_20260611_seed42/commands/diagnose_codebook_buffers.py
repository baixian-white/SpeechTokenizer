"""exp15 pre-diagnostic (CPU-only, no GPU): inspect RVQ codebook health buffers
in the exp2 distill30 best checkpoint.

Goal: decide exp15 direction. The codebook ALREADY has dead-code reinit
(expire_codes_, threshold_ema_dead_code=2) and kmeans_init=True, yet exp2 ended
with ~82% L1 dead code. This script reads the cluster_size / embed buffers directly
from the checkpoint (no forward pass, loads on CPU) to characterize WHY:
  - "never activated": cluster_size near 0 for dead codes (init residue)
  - "activated then decayed": broad distribution, many codes with small-but-non-trivial
    EMA mass that fell under threshold
Also reports embed-vector norms (collapsed/duplicate codes) per layer.

Run: conda run -n speechtokenizer python <this file>
"""
import os
import sys

import torch

CKPT = r"h:\H-CODE\speechtokenizer\output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt"
OUT = r"h:\H-CODE\speechtokenizer\output\experiments\exp15_codebook_health_20260611_seed42"


def find_codebook_buffers(sd):
    """Return list of (layer_idx, cluster_size_tensor, embed_tensor) found in state_dict."""
    # keys look like quantizer.vq.layers.<i>._codebook.cluster_size / .embed
    layers = {}
    for k, v in sd.items():
        if "_codebook.cluster_size" in k:
            idx = k.split(".layers.")[1].split(".")[0]
            layers.setdefault(idx, {})["cluster_size"] = v
        elif "_codebook.embed" in k and "embed_avg" not in k:
            idx = k.split(".layers.")[1].split(".")[0]
            layers.setdefault(idx, {})["embed"] = v
        elif "_codebook.embed_avg" in k:
            idx = k.split(".layers.")[1].split(".")[0]
            layers.setdefault(idx, {})["embed_avg"] = v
    return layers


def main():
    print("loading (CPU):", CKPT)
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    # checkpoint may be a dict with 'model'/'generator' or a raw state_dict
    if isinstance(obj, dict):
        for key in ("model", "generator", "state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                print("  using nested state_dict key:", key)
                break
        else:
            sd = obj
            print("  treating top-level dict as state_dict")
    else:
        sd = obj.state_dict()

    cb_keys = [k for k in sd if "_codebook" in k]
    print(f"  found {len(cb_keys)} codebook-related keys; sample: {cb_keys[:3]}")
    layers = find_codebook_buffers(sd)
    if not layers:
        print("NO codebook buffers found. All keys with 'quantizer':")
        for k in sd:
            if "quantizer" in k:
                print("   ", k, tuple(sd[k].shape) if hasattr(sd[k], "shape") else type(sd[k]))
        sys.exit(1)

    os.makedirs(os.path.join(OUT, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "reports"), exist_ok=True)
    print(f"\n{'layer':>5} {'K':>5} {'dead<2':>7} {'dead%':>7} {'cs=0':>6} "
          f"{'cs_med':>8} {'cs_p90':>9} {'cs_max':>9} {'embed_norm_med':>14}")
    lines = []
    for idx in sorted(layers, key=int):
        d = layers[idx]
        cs = d.get("cluster_size")
        embed = d.get("embed")
        if cs is None:
            continue
        K = cs.numel()
        dead = (cs < 2).sum().item()
        zero = (cs == 0).sum().item()
        cs_sorted = cs.sort().values
        med = cs.median().item()
        p90 = cs_sorted[int(0.9 * K)].item()
        mx = cs.max().item()
        enorm_med = float("nan")
        if embed is not None:
            enorm = embed.norm(dim=1)
            enorm_med = enorm.median().item()
        row = (f"{idx:>5} {K:>5} {dead:>7} {100*dead/K:>6.1f}% {zero:>6} "
               f"{med:>8.3f} {p90:>9.3f} {mx:>9.2f} {enorm_med:>14.4f}")
        print(row)
        lines.append({"layer": idx, "K": K, "dead_lt2": dead, "dead_pct": round(100*dead/K, 2),
                      "cs_zero": zero, "cs_median": round(med, 4), "cs_p90": round(p90, 4),
                      "cs_max": round(mx, 3), "embed_norm_median": round(enorm_med, 5)})

    # write csv
    import csv as _csv
    with open(os.path.join(OUT, "metrics", "codebook_buffer_health.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(lines[0].keys()))
        w.writeheader(); w.writerows(lines)
    print("\nwrote metrics/codebook_buffer_health.csv")

    # interpretation hint
    print("\nINTERPRETATION:")
    for r in lines:
        if r["cs_zero"] / r["K"] > 0.5:
            verdict = "mostly NEVER-ACTIVATED (cs==0) -> reinit not catching them / init residue"
        elif r["dead_pct"] > 50 and r["cs_zero"] / r["K"] < 0.2:
            verdict = "ACTIVATED-THEN-DECAYED (cs small but >0) -> EMA decay starves codes"
        else:
            verdict = "mixed"
        print(f"  L{int(r['layer'])+1}: dead {r['dead_pct']}%, cs==0 {r['cs_zero']}/{r['K']} -> {verdict}")


if __name__ == "__main__":
    main()
