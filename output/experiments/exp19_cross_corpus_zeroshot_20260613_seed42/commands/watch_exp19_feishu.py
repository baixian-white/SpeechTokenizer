#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# exp19 跨语料评估 飞书里程碑监控：阶段感知状态机（下载→解压→建list→评估→完成）。
# 每跨一个里程碑发一条飞书；异常(下载失败/评估崩溃/NaN)告警并 @全体。发到 feishu-main。
import os
import re
import sys
import time

SKILL = r"C:\Users\Windows11\.claude\skills\feishu-monitor\scripts"
sys.path.insert(0, SKILL)
from watch_lib import Throttle, Reporter, pid_alive, make_logger, get_notifier  # noqa: E402

ROOT = r"h:\H-CODE\speechtokenizer"
RUN = os.path.join(ROOT, r"output\experiments\exp19_cross_corpus_zeroshot_20260613_seed42")
ZIP = os.path.join(ROOT, r"data\_downloads\VCTK-Corpus-0.92.zip")
VCTK_DIR = os.path.join(ROOT, r"data\VCTK")
SAMPLE_LIST = os.path.join(RUN, r"artifacts\vctk_300.txt")
EVAL_CSV = os.path.join(RUN, r"eval_vctk_300\metrics\full_clean_results.csv")
EVAL_LOG = os.path.join(RUN, r"logs\eval.log")
DL_OUT = r"C:\Users\Windows11\AppData\Local\Temp\claude\h--H-CODE-speechtokenizer\e48702d9-12a5-4353-b5eb-88fbab61a70b\tasks\bodl62edf.output"
STATE_FILE = os.path.join(RUN, r"commands\_fb_stage.txt")
EXPECT_ROWS = 1800  # 300 samples x 3 L x 2 models
ZIP_TARGET = 11_000_000_000  # only for % display; completion is detected via curl DONE line
BOT = "feishu-main"
POLL = 120
AT_ALL = '<at user_id="all">所有人</at>'

wlog = make_logger("watch_exp19_feishu.log")
n = get_notifier(BOT)
th = Throttle(900)
dl_report = Reporter(300)  # download-phase progress: every 5 min


def read_stage():
    try:
        return open(STATE_FILE, encoding="utf-8").read().strip()
    except Exception:
        return ""


def write_stage(s):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    open(STATE_FILE, "w", encoding="utf-8").write(s)


def fsize(p):
    try:
        return os.path.getsize(p)
    except Exception:
        return 0


def csv_rows(p):
    try:
        with open(p, encoding="utf-8") as f:
            return sum(1 for _ in f) - 1  # minus header
    except Exception:
        return 0


def log_has_error(p):
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            tail = f.read()[-200000:]
        for pat in ("Traceback (most recent", "OutOfMemoryError", "CUDA error",
                    "MemoryError", "RuntimeError"):
            if pat in tail:
                for line in reversed(tail.replace("\r", "\n").split("\n")):
                    if pat.split()[0] in line:
                        return line.strip()[:400]
        return None
    except Exception:
        return None


def download_done():
    """True only when curl wrote its DONE line (i.e. process finished cleanly)."""
    try:
        return "DONE HTTP=" in open(DL_OUT, encoding="utf-8", errors="ignore").read()
    except Exception:
        return False


def main():
    wlog("exp19 feishu monitor started")
    n.send("[exp19] 飞书监控已启动",
           "里程碑模式：下载→解压→建list→评估→完成 各报一条；异常@全体。当前盯 VCTK 下载。")
    while True:
        stage = read_stage()
        # ---- detect current pipeline state from filesystem ----
        zsz = fsize(ZIP)
        vctk_extracted = os.path.isdir(VCTK_DIR) and any(
            f.lower().endswith((".flac", ".wav"))
            for _d, _s, fs in os.walk(VCTK_DIR) for f in fs[:1]
        ) if os.path.isdir(VCTK_DIR) else False
        list_ready = fsize(SAMPLE_LIST) > 0
        rows = csv_rows(EVAL_CSV)
        err = log_has_error(EVAL_LOG)

        # ---- download-phase periodic progress (every 5 min until curl done) ----
        if stage == "" and zsz > 0 and not download_done() and dl_report.due():
            n.send("[exp19] 下载中",
                   f"VCTK: {zsz/1e9:.2f}GB (~{zsz*100//ZIP_TARGET}% of est. 11GB, 实际略大)")
            wlog(f"[dl] {zsz/1e9:.2f}GB")

        # ---- error alerts (eval phase) ----
        if err and th.allow("eval_err"):
            n.send("[exp19] 评估异常", f"{AT_ALL}\n{err}\n已写 {rows}/{EXPECT_ROWS} 行")
            wlog(f"[ALERT] {err}")

        # ---- milestone transitions (one message each) ----
        if rows >= EXPECT_ROWS and stage != "done":
            n.send("[exp19] 评估完成 ✅",
                   f"VCTK 跨语料评估完成：{rows} 行 (300样本×3L×2模型)。\n"
                   f"产物: metrics/full_clean_results.csv\n等待完整性检查 + 写实验记录。")
            write_stage("done")
            wlog("[milestone] eval done"); return
        elif rows > 0 and stage in ("", "downloaded", "extracted", "list_ready"):
            n.send("[exp19] 评估进行中",
                   f"已写 {rows}/{EXPECT_ROWS} 行 (~{rows*100//EXPECT_ROWS}%)")
            write_stage("evaluating")
        elif list_ready and stage in ("", "downloaded", "extracted"):
            nlines = sum(1 for _ in open(SAMPLE_LIST, encoding="utf-8"))
            n.send("[exp19] 采样完成", f"VCTK sample list 就绪：{nlines} 条，准备启动评估。")
            write_stage("list_ready")
        elif vctk_extracted and stage in ("", "downloaded"):
            n.send("[exp19] 解压完成", "VCTK 已解压，开始建 sample list。")
            write_stage("extracted")
        elif zsz > 0 and download_done() and stage == "":
            n.send("[exp19] 下载完成",
                   f"VCTK zip 下载完成 ({zsz/1e9:.2f}GB)，准备校验+解压。")
            write_stage("downloaded")

        time.sleep(POLL)


if __name__ == "__main__":
    main()
