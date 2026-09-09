#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 本次监控：SCIT-Speech 实验B 手工SEANet Base 正式训练(~3天)。
#   信号源不是 stdout 日志(被 nohup 块缓冲, 一直空)，而是实时落盘的:
#     - checkpoints/SpeechTokenizerTrainer_NNNNNNNN : 新存档 = 进度里程碑(每2500步)
#     - checkpoints/SpeechTokenizer_best_dev.pt     : mtime 变 = dev mel 刷新最优
#     - checkpoints/logs/events.out.tfevents.*       : mtime = 训练心跳(每~100步/~100s)
#   报警 = TB 心跳 > 1800s 无更新(疑似卡死/崩溃)。
#   里程碑 = 每个新 checkpoint 发一条。每 7200s 健康汇报。完成 = 到 step>=200000 或心跳长期停。
#   发到 feishu-main。
import os, re, sys, time, glob
sys.path.insert(0, r"C:/Users/Windows11/.claude/skills/feishu-monitor/scripts")
from watch_lib import StallWatch, Throttle, Reporter, make_logger, get_notifier

NAME    = "实验B-手工SEANet-Base"
CKPT_DIR = r"H:/H-CODE/speechtokenizer/output/experiments/redoB_base_handcrafted_seanet_20260620_seed42/checkpoints"
TB_GLOB  = os.path.join(CKPT_DIR, "logs", "events.out.tfevents.*")
BEST_PT  = os.path.join(CKPT_DIR, "SpeechTokenizer_best_dev.pt")
BOT     = "feishu-main"
POLL    = 60
STALL   = 1800          # TB 心跳无更新阈值(秒)
REPORT  = 7200          # 健康汇报间隔
TOTAL_STEPS = 202740    # 60 epoch
CKPT_RE = re.compile(r"SpeechTokenizerTrainer_(\d+)$")

wlog = make_logger(os.path.splitext(os.path.basename(__file__))[0] + ".log")
n    = get_notifier(BOT)

def latest_step():
    best = -1
    for p in glob.glob(os.path.join(CKPT_DIR, "SpeechTokenizerTrainer_*")):
        m = CKPT_RE.search(os.path.basename(p))
        if m: best = max(best, int(m.group(1)))
    return best

def tb_idle():
    fs = glob.glob(TB_GLOB)
    if not fs: return None
    return time.time() - max(os.path.getmtime(f) for f in fs)

def best_mtime():
    return os.path.getmtime(BEST_PT) if os.path.exists(BEST_PT) else None

def main():
    wlog(f"监控 '{NAME}' 启动 ckpt={CKPT_DIR} stall={STALL} report={REPORT}")
    step0 = latest_step()
    n.send(f"[{NAME}] 监控已启动(已改信号源)",
           f"原 stdout 日志被块缓冲一直空, 已改盯 checkpoint+TB心跳\n"
           f"当前已到 step={step0} / {TOTAL_STEPS}\n"
           f"里程碑: 每个新 checkpoint(2500步)\n心跳卡死阈值 {STALL//60}min · 健康汇报每 {REPORT//3600}h")
    sw  = StallWatch(STALL)
    rep = Reporter(REPORT)
    th  = Throttle(STALL)
    start = time.time()
    last_step = step0
    last_best = best_mtime()
    while True:
        step = latest_step()
        if step > last_step:
            pct = 100.0 * step / TOTAL_STEPS
            n.send(f"[{NAME}] 进度 step={step}",
                   f"{step}/{TOTAL_STEPS} ({pct:.1f}%)\n运行 {(time.time()-start)/3600:.2f}h")
            wlog(f"[里程碑] step={step}")
            last_step = step
        bm = best_mtime()
        if bm and last_best and bm > last_best:
            n.send(f"[{NAME}] dev mel 刷新最优",
                   f"best_dev.pt 在 step≈{step} 更新(dev mel 下降, 方向正确)")
            wlog(f"[best] mtime updated step={step}")
        last_best = bm

        idle = tb_idle()
        edge = sw.update(idle)
        if edge == "stall":
            n.send(f"[{NAME}] ⏸ 疑似卡死/崩溃",
                   f"TB 心跳 {idle/60:.1f} 分钟无更新, 训练可能已停\n最近 step={step}")
            wlog(f"[卡死] tb_idle={idle:.0f}s step={step}")
        elif edge == "recover":
            n.send(f"[{NAME}] ▶ 已恢复", f"TB 心跳恢复, step={step}"); wlog("[恢复]")

        if rep.due():
            idle_s = f"{idle:.0f}s 前" if idle is not None else "无TB文件"
            n.send(f"[{NAME}] 健康汇报",
                   f"运行 {(time.time()-start)/3600:.2f}h\n"
                   f"step={step}/{TOTAL_STEPS} ({100.0*step/TOTAL_STEPS:.1f}%)\n"
                   f"TB 心跳 {idle_s}更新" + ("（疑似卡死）" if sw.stalled else "（正常）"))
            wlog(f"[汇报] step={step} idle={idle}")

        if step >= TOTAL_STEPS - 1 and th.allow("done"):
            n.send(f"[{NAME}] ✅ 训练完成",
                   f"step={step} 已达 {TOTAL_STEPS}\n用时 {(time.time()-start)/3600:.2f}h\n请来做 B3 验收")
            wlog("[完成] 退出"); return
        time.sleep(POLL)

if __name__ == "__main__":
    main()
