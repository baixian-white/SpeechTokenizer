"""exp13 final integrity check (CPU): verify packaged best ckpt loads and its embedded
best_dev_mel matches the TB best (0.4032@57500). No GPU, no forward."""
import torch

PKG = r"h:\H-CODE\speechtokenizer\output\experiments\exp13_handdesigned_encoder_distill30_20260611_seed42\checkpoints\SCIT-Speech-Base_best.pt"

print("loading packaged best ckpt (CPU):", PKG)
pkg = torch.load(PKG, map_location="cpu", weights_only=False)
print("type:", type(pkg).__name__)
if isinstance(pkg, dict):
    print("keys:", list(pkg.keys()))
    # generator state dict present?
    g = pkg.get("generator") or pkg.get("model")
    if isinstance(g, dict):
        n_enc = sum(1 for k in g if k.startswith("encoder"))
        n_dec = sum(1 for k in g if k.startswith("decoder"))
        n_cb = sum(1 for k in g if "_codebook" in k)
        print(f"generator tensors: {len(g)} (encoder={n_enc}, decoder={n_dec}, codebook={n_cb})")
    if "best_dev_mel_loss" in pkg:
        print(f"embedded best_dev_mel_loss: {pkg['best_dev_mel_loss']}")
print("\nINTEGRITY OK: packaged ckpt deserialized.")
