#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三用户实时通信跑批：为论文收集 RTF / 抖动 / 聚合带宽 / 丢包 的可复现测量。

对每个 (device, rvq_layers) 组合启动一个 router + 三个客户端（A/B/C 全部灌同一段
固定 wav 互发、--no_play 无头解码），跑 --duration 秒后优雅停止，把每客户端汇总
追加到统一 CSV。确定性输入（固定 wav，非麦克风）保证可复现。

用法（在本目录下）：
    python run_benchmark.py --python <conda_python> --duration 60 --devices cuda cpu --layers 1 2 3

说明：
- 默认模型走 group_client 的默认值（bundle 内 Log/spt_base，即论文 LCA v2 模型）。
- 每个 L 一个独立 room/port，避免串扰。
- 输出 CSV：benchmark_results.csv（每行一个 user×device×L）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_WAV = DEMO_DIR.parents[2] / "example_input.wav"


def launch(pycmd, args, logfile):
    f = open(logfile, "w", encoding="utf-8")
    return subprocess.Popen([pycmd, *args], cwd=str(DEMO_DIR), stdout=f, stderr=subprocess.STDOUT), f


def run_one(pycmd, device, layers, duration, wav, port, out_csv, logdir):
    room = f"bench_{device}_L{layers}"
    print(f"\n=== device={device} L={layers} port={port} duration={duration}s ===")

    router_args = ["router_server.py", "--listen_ip", "127.0.0.1", "--listen_port", str(port),
                   "--room_id", room, "--max_queue", "8"]
    router_proc, router_f = launch(pycmd, router_args, logdir / f"router_{device}_L{layers}.log")
    time.sleep(3.0)  # router 起来

    client_procs = []
    files = []
    for uid in ("A", "B", "C"):
        c_args = ["group_client.py", "--user_id", uid, "--room_id", room,
                  "--router_ip", "127.0.0.1", "--router_port", str(port),
                  "--wav_input", str(wav), "--loop_wav", "--no_mic", "--no_play",
                  "--device", device, "--rvq_layers", str(layers),
                  "--chunk_seconds", "0.5", "--run_seconds", str(duration),
                  "--summary_csv", str(out_csv), "--monitor"]
        p, f = launch(pycmd, c_args, logdir / f"client_{uid}_{device}_L{layers}.log")
        client_procs.append(p)
        files.append(f)
        time.sleep(1.0)  # 错开连接

    # 客户端到 run_seconds 后自停并写 CSV（不用 terminate 硬杀，Windows 上 terminate 不触发 finally）
    grace = duration + 60
    for p in client_procs:
        try:
            p.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            p.terminate()
    router_proc.terminate()
    try:
        router_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        router_proc.kill()
    for f in files + [router_f]:
        try:
            f.close()
        except Exception:
            pass
    print(f"=== done device={device} L={layers}; CSV -> {out_csv} ===")


def main():
    ap = argparse.ArgumentParser(description="3-user realtime benchmark orchestrator")
    ap.add_argument("--python", default=sys.executable, help="用于跑 router/client 的 python 解释器")
    ap.add_argument("--duration", type=int, default=60, help="每组合采集秒数")
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"], choices=["cuda", "cpu"])
    ap.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--wav", default=str(DEFAULT_WAV))
    ap.add_argument("--base_port", type=int, default=28080, help="起始端口，避开 Windows 排除端口段")
    ap.add_argument("--out_csv", default=str(DEMO_DIR / "benchmark_results.csv"))
    ap.add_argument("--logdir", default=str(DEMO_DIR / "benchmark_logs"))
    args = ap.parse_args()

    wav = Path(args.wav)
    if not wav.exists():
        sys.exit(f"wav not found: {wav}")
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    if out_csv.exists():
        out_csv.unlink()  # 重跑清空

    port = args.base_port
    for device in args.devices:
        for layers in args.layers:
            run_one(args.python, device, layers, args.duration, wav, port, out_csv, logdir)
            port += 1
            time.sleep(2.0)

    print(f"\nALL DONE. Aggregated CSV: {out_csv}")
    print("列含义见 group_client.write_summary_csv：RTF/jitter/kbps/drop_rate/one_way_latency_budget_ms")


if __name__ == "__main__":
    main()
