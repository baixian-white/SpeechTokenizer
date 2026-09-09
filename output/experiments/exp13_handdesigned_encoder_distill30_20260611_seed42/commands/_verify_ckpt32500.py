"""CPU-only integrity check of the exp13 step-32500 trainer checkpoint before resume.
Verifies torch can fully deserialize it and that all keys needed by trainer.load()
(generator, discriminators, optim_g/d, scheduler_g/d) are present. No GPU, no forward."""
import torch

CKPT = r"h:\H-CODE\speechtokenizer\output\experiments\exp13_handdesigned_encoder_distill30_20260611_seed42\checkpoints\SpeechTokenizerTrainer_00032500"

print("loading (CPU, weights_only=False):", CKPT)
pkg = torch.load(CKPT, map_location="cpu", weights_only=False)
print("top-level type:", type(pkg).__name__)
if isinstance(pkg, dict):
    print("top-level keys:", list(pkg.keys()))
    for k in ("generator", "discriminators", "optim_g", "optim_d", "scheduler_g", "scheduler_d"):
        present = k in pkg
        print(f"  {k:14s}: {'OK' if present else 'MISSING'}", end="")
        if present and isinstance(pkg[k], dict):
            print(f"  ({len(pkg[k])} entries)")
        else:
            print()
    # sanity: generator has codebook + encoder params
    if "generator" in pkg and isinstance(pkg["generator"], dict):
        g = pkg["generator"]
        n_cb = sum(1 for kk in g if "_codebook" in kk)
        n_enc = sum(1 for kk in g if kk.startswith("encoder"))
        n_dec = sum(1 for kk in g if kk.startswith("decoder"))
        print(f"  generator: {len(g)} tensors  (encoder={n_enc}, decoder={n_dec}, codebook={n_cb})")
        # scheduler last_epoch tells us the resume step alignment
    if "scheduler_g" in pkg and isinstance(pkg["scheduler_g"], dict):
        print("  scheduler_g state:", {k: pkg["scheduler_g"][k] for k in pkg["scheduler_g"] if k in ("last_epoch", "_step_count", "T_max", "base_lrs")})
print("\nINTEGRITY: checkpoint deserialized fully without error.")
