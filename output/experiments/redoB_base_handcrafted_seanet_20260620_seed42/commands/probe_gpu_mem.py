import json, os, sys
from pathlib import Path
import torch

ROOT = Path("h:/H-CODE/speechtokenizer")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from speechtokenizer import SpeechTokenizer
from speechtokenizer.discriminators import (
    MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator,
)

CFG = "output/experiments/redoB_base_handcrafted_seanet_20260620_seed42/configs/scit_speech_base_config.json"
with open(CFG, encoding="utf-8-sig") as f:
    cfg = json.load(f)

dev = "cuda"
torch.cuda.reset_peak_memory_stats()
gen = SpeechTokenizer(cfg).to(dev)
discs = {
    'mpd': MultiPeriodDiscriminator().to(dev),
    'msd': MultiScaleDiscriminator().to(dev),
    'mstftd': MultiScaleSTFTDiscriminator(32).to(dev),
}
og = torch.optim.AdamW(gen.parameters(), lr=1e-4)
od = torch.optim.AdamW([p for d in discs.values() for p in d.parameters()], lr=1e-4)

B = cfg["batch_size"]; T = cfg["segment_size"]
for it in range(2):
    x = torch.randn(B, 1, T, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x_hat, loss_q, feat = gen(x)
        od.zero_grad(set_to_none=True)
        d_out = [d(x, x_hat.detach()) for d in discs.values()]
        loss_d = sum((o[0][0].float().mean()*0 + sum((dr-1).pow(2).mean() for dr in o[0]) + sum(dg.pow(2).mean() for dg in o[1])) for o in d_out)
    loss_d.backward(); od.step()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        d_out = [d(x, x_hat) for d in discs.values()]
        loss_g = x_hat.abs().mean() + loss_q*10 + sum(sum((1-dg).pow(2).mean() for dg in o[1]) for o in d_out)
    og.zero_grad(set_to_none=True); loss_g.backward(); og.step()
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated()/1e9
reserved = torch.cuda.max_memory_reserved()/1e9
print(f"peak_allocated_GB: {peak:.2f}")
print(f"peak_reserved_GB: {reserved:.2f}")
