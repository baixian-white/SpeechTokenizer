#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单文件版：编码 → 存储 → 模拟传输（按信道码率 & npz 大小计算传输时长）→ 读入 → 解码 → 写出

新增：
- 参数 --rvq_layers：指定传输/解码使用的 RVQ 层数（从第 0 层起，取前 L 层）。
- 参数 --channel_bitrate_bps：信道传输码率（bit/s），用于由 npz 文件大小推算传输耗时。
- 编码阶段保存 .npz 后，打印 codes 的 shape / dtype / 文件大小。
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
import numpy as np
from scipy.io.wavfile import write
from speechtokenizer import SpeechTokenizer
from huggingface_hub import snapshot_download


# ========== 高精度时间点 → 墙钟时间 的一致映射 ==========
def make_time_mapper():
    t0_perf = time.perf_counter()
    t0_wall = time.time()

    def ts_from_perf(t_perf: float) -> str:
        wall = t0_wall + (t_perf - t0_perf)
        return datetime.fromtimestamp(wall).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def dur(sec: float) -> str:
        return f"{sec:.3f}s"

    return ts_from_perf, dur


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.2f} {u}"
        size /= 1024.0


def human_bitrate(bps: float) -> str:
    units = ["bps", "Kbps", "Mbps", "Gbps"]
    rate = float(bps)
    for u in units:
        if rate < 1000.0 or u == units[-1]:
            return f"{rate:.2f} {u}"
        rate /= 1000.0


def main():
    ts_from_perf, fmt_dur = make_time_mapper()

    # =========================================================
    # 0️⃣ 参数
    # =========================================================
    parser = argparse.ArgumentParser(description="单文件模拟：编码-传输-解码（含详细时间打印）")
    parser.add_argument("--config_path", type=str,
                        default="model_hub/speechtokenizer_hubert_avg/config.json")
    parser.add_argument("--ckpt_path", type=str,
                        default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    parser.add_argument("--speech_file", type=str, required=True,
                        help="输入语音文件路径（.wav 等）")
    parser.add_argument("--codes_file", type=str, default="codes_output.npz",
                        help="中间存储的离散 codes（供传输/解码读取）")
    parser.add_argument("--output_wav", type=str, default="decoded_output.wav",
                        help="解码后输出的 WAV 文件名")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备：cuda 或 cpu")
    parser.add_argument("--download", action="store_true",
                        help="如指定则从 Hugging Face 下载模型到 ./model_hub")
    parser.add_argument("--simulate_transfer", type=int, default=1,
                        help="是否模拟传输睡眠：1=是，0=否")
    parser.add_argument("--rvq_layers", type=int, default=0,
                        help="传输/解码的 RVQ 层数 L（取前 L 层，0 或不填表示使用全部层）")
    parser.add_argument("--channel_bitrate_bps", type=float, default=256_000,
                        help="信道码率（bit/s），用于根据 npz 文件大小计算传输时间。默认 256 kbps。")
    args = parser.parse_args()

    if args.download:
        snapshot_download(repo_id="fnlp/SpeechTokenizer", local_dir="model_hub")

    # =========================================================
    # 2️⃣ 加载模型
    # =========================================================
    model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
    model.eval()
    model.to(args.device)
    model_sr = int(model.sample_rate)

    t_total_start = time.perf_counter()

    # =========================================================
    # 3️⃣ 编码端：加载与重采样
    # =========================================================
    t_load_start = time.perf_counter()
    wav, sr_in = torchaudio.load(args.speech_file)
    ch_in = int(wav.shape[0])
    dur_in_sec = float(wav.shape[1]) / float(sr_in)

    print("\n[1] 输入音频信息")
    print(f"  - 路径：{Path(args.speech_file).resolve()}")
    print(f"  - 通道数：{ch_in}")
    print(f"  - 采样率：{sr_in} Hz")
    print(f"  - 音频时长：{fmt_dur(dur_in_sec)}")

    if sr_in != model_sr:
        wav = torchaudio.transforms.Resample(sr_in, model_sr)(wav)
    if wav.shape[0] > 1:
        wav = wav[:1, :]
    wav = wav.unsqueeze(0).to(args.device)

    t_load_end = time.perf_counter()
    print("\n[2] 编码端：音频加载与重采样")
    print(f"  - 开始时间：{ts_from_perf(t_load_start)}")
    print(f"  - 结束时间：{ts_from_perf(t_load_end)}")
    print(f"  - 耗时：{fmt_dur(t_load_end - t_load_start)}")

    # =========================================================
    # 4️⃣ 编码：语音 → 离散 codes
    # =========================================================
    t_enc_start = time.perf_counter()
    with torch.no_grad():
        codes = model.encode(wav)
    t_enc_end = time.perf_counter()

    codes_np_full = codes.detach().cpu().numpy()
    n_q_total = int(codes_np_full.shape[0])

    # 按层数截断
    if args.rvq_layers and args.rvq_layers > 0:
        L = min(args.rvq_layers, n_q_total)
        codes_np = codes_np_full[:L, :, :]
    else:
        L = n_q_total
        codes_np = codes_np_full

    # 保存 npz
    np.savez_compressed(
        args.codes_file,
        codes=codes_np,
        model_sample_rate=model_sr,
        n_q_total=n_q_total,
        rvq_layers_used=L,
        batch=int(codes_np.shape[1]),
        time_steps=int(codes_np.shape[2]),
        source=Path(args.speech_file).name,
    )

    # 输出 npz 文件内容信息
    npz_path = Path(args.codes_file).resolve()
    npz_size = npz_path.stat().st_size
    with np.load(npz_path) as f:
        codes_arr = f["codes"]
        shape = codes_arr.shape
        dtype = codes_arr.dtype

    print("\n[3] 编码阶段")
    print(f"  - 开始时间：{ts_from_perf(t_enc_start)}")
    print(f"  - 结束时间：{ts_from_perf(t_enc_end)}")
    print(f"  - 编码耗时：{fmt_dur(t_enc_end - t_enc_start)}")
    print(f"  - 模型总 RVQ 层数：{n_q_total}")
    print(f"  - 实际使用层数：{L}")
    print(f"  - codes 矩阵形状：{shape}")
    print(f"  - codes 数据类型：{dtype}")
    print(f"  - npz 文件大小：{human_size(npz_size)}")
    print(f"  - 文件路径：{npz_path}")

    # =========================================================
    # 5️⃣ 传输阶段（按码率计算）
    # =========================================================
    t_tx_start = time.perf_counter()

    if args.channel_bitrate_bps <= 0:
        raise ValueError("--channel_bitrate_bps 必须为正数（bit/s）。")

    # 基于文件大小与码率计算传输时间（秒）
    tx_dur = (npz_size * 8.0) / float(args.channel_bitrate_bps)

    if args.simulate_transfer == 1 and tx_dur > 0:
        time.sleep(tx_dur)

    t_tx_end = time.perf_counter()

    print("\n[4] 传输阶段（按码率模拟）")
    print(f"  - 信道码率：{human_bitrate(args.channel_bitrate_bps)}")
    print(f"  - 传输数据大小：{human_size(npz_size)}（{npz_size} bytes）")
    print(f"  - 计算得到的传输时长：{fmt_dur(tx_dur)}")
    print(f"  - 开始时间：{ts_from_perf(t_tx_start)}")
    print(f"  - 结束时间：{ts_from_perf(t_tx_end)}")

    # =========================================================
    # 6️⃣ 解码端：读入 codes
    # =========================================================
    t_readcodes_start = time.perf_counter()
    data = np.load(args.codes_file)
    codes_np_in = data["codes"]
    meta_sr = int(data["model_sample_rate"]) if "model_sample_rate" in data else model_sr
    codes_in = torch.from_numpy(codes_np_in).to(args.device)
    t_readcodes_end = time.perf_counter()

    print("\n[5] 解码端：读入数据")
    print(f"  - 开始时间：{ts_from_perf(t_readcodes_start)}")
    print(f"  - 结束时间：{ts_from_perf(t_readcodes_end)}")
    print(f"  - 读取耗时：{fmt_dur(t_readcodes_end - t_readcodes_start)}")

    # =========================================================
    # 7️⃣ 解码
    # =========================================================
    t_dec_start = time.perf_counter()
    with torch.no_grad():
        wav_out = model.decode(codes_in)
    t_dec_end = time.perf_counter()

    print("\n[6] 解码阶段")
    print(f"  - 开始时间：{ts_from_perf(t_dec_start)}")
    print(f"  - 结束时间：{ts_from_perf(t_dec_end)}")
    print(f"  - 解码耗时：{fmt_dur(t_dec_end - t_dec_start)}")

    # =========================================================
    # 8️⃣ 写出结果
    # =========================================================
    t_write_start = time.perf_counter()
    wav_np = wav_out.detach().cpu().numpy().astype(np.float32)
    write(args.output_wav, meta_sr, wav_np.squeeze())
    t_write_end = time.perf_counter()

    print("\n[7] 写出结果")
    print(f"  - 开始时间：{ts_from_perf(t_write_start)}")
    print(f"  - 结束时间：{ts_from_perf(t_write_end)}")
    print(f"  - 写出耗时：{fmt_dur(t_write_end - t_write_start)}")
    print(f"  - 输出文件：{Path(args.output_wav).resolve()}")
    print(f"  - 输出采样率：{meta_sr} Hz")

    # =========================================================
    # 9️⃣ 总时长
    # =========================================================
    t_total_end = time.perf_counter()
    print("\n[8] 总时长")
    print(f"  - 总耗时：{fmt_dur(t_total_end - t_total_start)}\n")


if __name__ == "__main__":
    main()
