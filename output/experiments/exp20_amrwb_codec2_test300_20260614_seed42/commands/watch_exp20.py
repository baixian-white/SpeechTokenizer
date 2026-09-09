#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 本次监控：exp20 AMR-WB+Codec2 编解码评测（600样本×15档=9000 wav）。
#   进度信号 = 磁盘已解码 wav 数 / 9000；活性 = 最新 wav mtime；
#   完成 = run_full.log 出现 "ALL DONE" 或 wav 数达 9000；
#   异常 = 日志 Traceback / 长时间无新 wav（卡死）。
#   汇报间隔 720s，卡死阈值 600s，发到 feishu-main。
import os, re, sys, time, glob

sys.path.insert(0, r"C:/Users/Windows11/.claude/skills/feishu-monitor/scripts")
from watch_lib import StallWatch, Throttle, Reporter, make_logger, get_notifier

NAME   = "exp20-AMRWB-Codec2"
RUNS   = r"H:/H-CODE/speechtokenizer/output/experiments/exp20_amrwb_codec2_test300_20260614_seed42/runs"
LOG    = r"H:/H-CODE/speechtokenizer/output/experiments/exp20_amrwb_codec2_test300_20260614_seed42/run_full.log"
BOT    = "feishu-main"
POLL   = 30
STALL  = 600        # 10 分钟无新 wav 视为疑似卡死
REPORT = 720        # 12 分钟一次健康汇报
TOTAL  = 9000       # 600 样本 × (9 amrwb + 6 codec2)

wlog = make_logger("watch_exp20.log")
n    = get_notifier(BOT)


def wav_count():
    return len(glob.glob(os.path.join(RUNS, "*", "samples", "*", "*", "*.wav")))


def newest_mtime():
    best = 0.0
    for p in glob.glob(os.path.join(RUNS, "*", "samples", "*", "*", "*.wav")):
        try:
            m = os.path.getmtime(p)
            if m > best:
                best = m
        except OSError:
            pass
    return best


def split_breakdown():
    out = []
    for s in ("test-clean_300", "test-other_300"):
        c = len(glob.glob(os.path.join(RUNS, s, "samples", "*", "*", "*.wav")))
        out.append(f"{s}={c}/4500")
    return " ".join(out)


def log_tail(nlines=20):
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            return f.read()[-4000:]
    except OSError:
        return ""


def main():
    wlog(f"监控 '{NAME}' 启动 runs={RUNS}")
    n.send(f"[{NAME}] 监控已启动",
           f"补 AMR-WB(9档)+Codec2(6档) baseline，共 {TOTAL} 个编解码评测。\n"
           f"每 {REPORT//60} 分钟汇报一次进度；卡死/崩溃/完成会立即推送。")
    sw  = StallWatch(STALL)
    th  = Throttle(900)
    rep = Reporter(REPORT)
    start = time.time()
    done_alerted = False
    while True:
        cnt = wav_count()
        nm  = newest_mtime()
        idle = (time.time() - nm) if nm > 0 else 0.0
        tail = log_tail()

        # 完成判定
        if cnt >= TOTAL or "ALL DONE" in tail:
            if not done_alerted:
                n.send(f"[{NAME}] ✅ 全部完成",
                       f"已解码 {cnt}/{TOTAL} 个 wav，用时 {(time.time()-start)/3600:.2f}h。\n"
                       f"{split_breakdown()}\n下一步：跑分析脚本出置信区间表 + 回写论文。")
                wlog(f"[完成] cnt={cnt}")
            wlog("[完成] 退出监控"); return

        # 异常：Traceback
        if "Traceback (most recent" in tail and th.allow("traceback"):
            # 取最后一段栈
            seg = tail[tail.rfind("Traceback (most recent"):][:600]
            n.send(f"[{NAME}] ⚠️ 检测到异常栈",
                   f"已解码 {cnt}/{TOTAL}。日志末尾:\n{seg}")
            wlog("[报警] traceback")

        # 卡死/恢复（基于最新 wav mtime）
        edge = sw.update(idle)
        if edge == "stall" and th.allow("stall"):
            n.send(f"[{NAME}] ⚠️ 疑似卡死",
                   f"{idle/60:.1f} 分钟没有新 wav 产生。已解码 {cnt}/{TOTAL}。\n{split_breakdown()}")
            wlog(f"[卡死] idle={idle:.0f}s cnt={cnt}")
        elif edge == "recover":
            n.send(f"[{NAME}] 已恢复", f"重新开始产出 wav。已解码 {cnt}/{TOTAL}。")
            wlog("[恢复]")

        # 定时健康汇报
        if rep.due():
            pct = 100.0 * cnt / TOTAL
            n.send(f"[{NAME}] 健康汇报",
                   f"进度 {cnt}/{TOTAL}（{pct:.0f}%），运行 {(time.time()-start)/3600:.2f}h\n"
                   f"{split_breakdown()}\n最新 wav {idle:.0f}s 前生成"
                   + ("（疑似卡死）" if sw.stalled else "（正常）"))
            wlog(f"[汇报] cnt={cnt} pct={pct:.0f}% idle={idle:.0f}s")

        time.sleep(POLL)


if __name__ == "__main__":
    main()
