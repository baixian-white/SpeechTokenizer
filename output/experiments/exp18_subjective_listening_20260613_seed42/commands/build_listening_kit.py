"""exp18 scaffold: build a self-contained subjective AB-preference listening kit.

Does NOT run any listening test — only prepares samples + form for external listeners.
Selects 40 utterances (3-8s sweet spot, seed=42) from exp12 test-clean_300 decoded wavs,
copies the needed methods into a flat listening dir, and generates an AB-preference HTML form.

Focused AB pairs (not all 15 — only those that support the paper's same-rate-wins claim):
  P1: SCIT-LCA L3 vs DAC n_q3      (both 1.5 kbps)
  P2: SCIT-LCA L3 vs EnCodec 1.5k  (both 1.5 kbps)
  P3: SCIT-LCA L3 vs Opus 6k       (cross-rate: SCIT 1/4 bandwidth)
  P4: SCIT-LCA L1 vs SCIT-LCA L3   (own rate ladder 500 vs 1500 bps)
Each pair is presented with A/B side randomized per (sample,pair) by seed so listeners
can't infer which is which. original is given as a hidden reference (optional anchor).
"""
import csv
import glob
import os
import random
import shutil
from pathlib import Path

ROOT = Path(r"h:\H-CODE\speechtokenizer")
SRC = ROOT / "output/experiments/exp12_baseline_comparison_test300_20260610_seed42/runs/test-clean_300/samples"
RUN = ROOT / "output/experiments/exp18_subjective_listening_20260613_seed42"
N = 40
SEED = 42

# method -> source subdir under SRC
METHODS = {
    "scit_lca_L1": "scit_lca/L1",
    "scit_lca_L3": "scit_lca/L3",
    "dac_1.5k": "dac/n_q3",
    "encodec_1.5k": "encodec/bw1.5kbps",
    "opus_6k": "opus/br6000",
    "original": "original",
}
PAIRS = [
    ("P1_lcaL3_vs_dac", "scit_lca_L3", "dac_1.5k", "both 1.5 kbps"),
    ("P2_lcaL3_vs_encodec", "scit_lca_L3", "encodec_1.5k", "both 1.5 kbps"),
    ("P3_lcaL3_vs_opus6k", "scit_lca_L3", "opus_6k", "SCIT 1.5k vs Opus 6k (1/4 bw)"),
    ("P4_lcaL1_vs_lcaL3", "scit_lca_L1", "scit_lca_L3", "own rate ladder 500 vs 1500 bps"),
]


def pick_samples():
    import soundfile as sf
    cands = []
    for f in sorted(glob.glob(str(SRC / "original" / "*.wav"))):
        info = sf.info(f)
        d = info.frames / info.samplerate
        if 3 <= d <= 8:
            cands.append(Path(f).name)
    rng = random.Random(SEED)
    rng.shuffle(cands)
    return sorted(cands[:N])


def copy_wavs(samples):
    """Copy needed method wavs into RUN/samples/<method>/<sample>. Returns missing list."""
    missing = []
    for m, sub in METHODS.items():
        dst = RUN / "samples" / m
        dst.mkdir(parents=True, exist_ok=True)
        for s in samples:
            src = SRC / sub / s
            if src.exists():
                shutil.copy2(src, dst / s)
            else:
                missing.append(f"{m}/{s}")
    return missing


def build_trials(samples):
    """One trial per (sample, pair); randomize which method is A vs B by seed."""
    rng = random.Random(SEED + 1)
    trials = []
    tid = 0
    for s in samples:
        for pid, ma, mb, note in PAIRS:
            tid += 1
            if rng.random() < 0.5:
                aside, bside = ma, mb
            else:
                aside, bside = mb, ma
            trials.append({"trial": tid, "sample_id": Path(s).stem, "wav": s,
                           "pair": pid, "note": note,
                           "A_method": aside, "B_method": bside})
    rng.shuffle(trials)  # randomize trial order
    for i, t in enumerate(trials, 1):
        t["order"] = i
    return trials


def write_key(trials):
    """Hidden answer key (which side is which method) — for analysis, NOT shown to listeners."""
    with open(RUN / "artifacts" / "trial_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["order", "trial", "sample_id", "wav", "pair",
                                          "note", "A_method", "B_method"])
        w.writeheader()
        for t in sorted(trials, key=lambda x: x["order"]):
            w.writerow(t)


def write_form(trials):
    """Self-contained AB-preference HTML. Audio paths relative to forms/ -> ../samples/.
    Listeners pick A / B / no-preference; JS exports responses.csv. No method names shown."""
    head = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>SCIT-Speech 主观 AB 听测</title><style>
body{font-family:sans-serif;max-width:820px;margin:20px auto;padding:0 12px;line-height:1.5}
.trial{border:1px solid #ccc;border-radius:8px;padding:14px;margin:14px 0}
.ab{display:flex;gap:24px;margin:8px 0}.ab>div{flex:1}
audio{width:100%}.q{font-weight:bold}label{margin-right:16px}
#bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #eee}
button{padding:8px 16px;font-size:15px}</style></head><body>
<div id="bar"><h2>SCIT-Speech 主观 AB 听测</h2>
<p>每题听 A、B 两段语音，选择你认为<b>整体音质更好/更自然</b>的一段。可平局。共 <span id="tot"></span> 题。
完成后点“导出结果”下载 CSV 回传。请戴耳机、安静环境。</p>
<p>听者ID：<input id="listener" placeholder="如 L01"> 进度：<span id="prog">0</span>/<span id="tot2"></span>
<button onclick="exportCSV()">导出结果 CSV</button></p></div>
<form id="f">
"""
    body = []
    for t in sorted(trials, key=lambda x: x["order"]):
        o = t["order"]
        a = f"../samples/{t['A_method']}/{t['wav']}"
        b = f"../samples/{t['B_method']}/{t['wav']}"
        body.append(f"""<div class="trial"><div class="q">第 {o} 题</div>
<div class="ab"><div>A<audio controls preload="none" src="{a}"></audio></div>
<div>B<audio controls preload="none" src="{b}"></audio></div></div>
<div>哪个更好？
<label><input type="radio" name="t{o}" value="A">A 更好</label>
<label><input type="radio" name="t{o}" value="B">B 更好</label>
<label><input type="radio" name="t{o}" value="tie">差不多</label></div></div>""")
    tail = """</form>
<script>
const N=%d;
document.getElementById('tot').textContent=N;document.getElementById('tot2').textContent=N;
document.getElementById('f').addEventListener('change',()=>{
 let c=0;for(let i=1;i<=N;i++){if(document.querySelector('input[name=t'+i+']:checked'))c++;}
 document.getElementById('prog').textContent=c;});
function exportCSV(){
 const lid=document.getElementById('listener').value||'anon';
 let rows=[['listener','order','choice']];
 for(let i=1;i<=N;i++){const v=document.querySelector('input[name=t'+i+']:checked');
  rows.push([lid,i,v?v.value:'']);}
 const csv=rows.map(r=>r.join(',')).join('\\n');
 const blob=new Blob([csv],{type:'text/csv'});const a=document.createElement('a');
 a.href=URL.createObjectURL(blob);a.download='responses_'+lid+'.csv';a.click();}
</script></body></html>""" % len(trials)
    (RUN / "forms").mkdir(parents=True, exist_ok=True)
    (RUN / "forms" / "ab_listening_form.html").write_text(head + "\n".join(body) + tail, encoding="utf-8")


def main():
    samples = pick_samples()
    print(f"selected {len(samples)} samples (3-8s, seed={SEED})")
    missing = copy_wavs(samples)
    if missing:
        print(f"WARNING {len(missing)} missing wavs: {missing[:5]}")
    else:
        print(f"copied {len(samples)} x {len(METHODS)} method wavs")
    trials = build_trials(samples)
    write_key(trials)
    write_form(trials)
    print(f"trials: {len(trials)} ({len(samples)} samples x {len(PAIRS)} pairs)")
    print(f"form: {RUN/'forms'/'ab_listening_form.html'}")
    print(f"key:  {RUN/'artifacts'/'trial_key.csv'}")


if __name__ == "__main__":
    main()


