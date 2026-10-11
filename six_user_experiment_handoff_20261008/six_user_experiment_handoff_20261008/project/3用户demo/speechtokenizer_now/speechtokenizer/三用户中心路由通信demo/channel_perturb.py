#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""信道索引扰动（与论文 §5.4 / scripts/evaluate_packet_burst_loss.py 逐行一致）。

用于在 demo 接收端 decode 前，对收到的 RVQ 索引按论文的确定性 packet/burst 丢失模型
施加扰动并以 previous-index 拼接隐藏，从而在真实实时链路上**精确重放**论文的丢包条件
（而非依赖 TCP/UDP 不可控的真实丢包，保证与论文同种子、可复现、语义一致）。

codes 形状 (L, B, T)，与论文一致；demo 单样本 B=1。
"""

from __future__ import annotations

import math

import torch


def previous_index_replace(codes, mask):
    out = codes.clone()
    if out.shape[-1] <= 1 or not mask.any():
        return out
    mask = mask.clone()
    mask[..., 0] = False
    prev = torch.cat([out[..., :1], out[..., :-1]], dim=-1)
    return torch.where(mask, prev, out)


def apply_packet_burst(codes, condition, generator):
    """codes (L, B, T)。返回 (扰动后 codes, stats)。与论文 apply_packet_burst 逐行一致。"""
    out = codes.clone()
    L, B, T = out.shape
    name = condition["name"]
    affected = 0

    if name == "clean":
        mask = torch.zeros_like(out, dtype=torch.bool)
    elif condition["type"] == "packet_loss":
        packet_frames = int(condition.get("packet_frames", 5))
        p_packet = float(condition.get("p_packet", 0.03))
        num_packets = int(math.ceil(T / packet_frames))
        packet_mask = torch.rand((B, num_packets), device=out.device, generator=generator) < p_packet
        frame_mask = torch.zeros((B, T), device=out.device, dtype=torch.bool)
        for packet_idx in range(num_packets):
            start = packet_idx * packet_frames
            end = min(T, start + packet_frames)
            frame_mask[:, start:end] = packet_mask[:, packet_idx:packet_idx + 1]
        mask = frame_mask.unsqueeze(0).expand(L, B, T)
        out = previous_index_replace(out, mask)
        affected = int(mask.sum().item())
    elif condition["type"] == "single_burst":
        burst_frames = int(condition.get("burst_frames", 5))
        burst_frames = max(1, min(burst_frames, T))
        frame_mask = torch.zeros((B, T), device=out.device, dtype=torch.bool)
        if T > burst_frames:
            starts = torch.randint(1, T - burst_frames + 1, (B,), device=out.device, generator=generator)
        else:
            starts = torch.zeros((B,), device=out.device, dtype=torch.long)
        for b in range(B):
            start = int(starts[b].item())
            frame_mask[b, start:start + burst_frames] = True
        frame_mask[:, 0] = False
        mask = frame_mask.unsqueeze(0).expand(L, B, T)
        out = previous_index_replace(out, mask)
        affected = int(mask.sum().item())
    else:
        raise ValueError(f"unknown condition type: {condition}")

    return out, {"affected": affected, "total": L * B * T}


# 论文 §4.3 / §5.4 评估端使用的扰动条件（正文主表强档 + 中弱档）
CONDITIONS = {
    "clean": {"name": "clean", "type": "clean"},
    "packet-loss-1p": {"name": "packet-loss-1p", "type": "packet_loss", "packet_frames": 5, "p_packet": 0.01},
    "packet-loss-3p": {"name": "packet-loss-3p", "type": "packet_loss", "packet_frames": 5, "p_packet": 0.03},
    "packet-loss-5p": {"name": "packet-loss-5p", "type": "packet_loss", "packet_frames": 5, "p_packet": 0.05},
    "burst-2f": {"name": "burst-2f", "type": "single_burst", "burst_frames": 2},
    "burst-5f": {"name": "burst-5f", "type": "single_burst", "burst_frames": 5},
    "burst-10f": {"name": "burst-10f", "type": "single_burst", "burst_frames": 10},
}
