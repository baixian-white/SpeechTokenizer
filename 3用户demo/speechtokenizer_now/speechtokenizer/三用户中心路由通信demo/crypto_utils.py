#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_key_bits(key_path: str) -> str:
    text = Path(key_path).read_text(encoding="utf-8", errors="ignore")
    bits = "".join(ch for ch in text if ch in "01")
    if not bits:
        raise ValueError(f"key file has no 0/1 bits: {key_path}")
    return bits


_UINT_VIEW = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def xor_int_ndarray(arr: np.ndarray, key_path: str) -> np.ndarray:
    """按数组自身位宽做逐元素 XOR。

    位宽由 dtype.itemsize 决定（uint16 → 16 bit/元素、int64 → 64 bit/元素），
    返回与输入相同 dtype。这样紧凑封装（uint16，对应 §3.1 的 10 bit 名义索引）
    时，每索引只消耗 16 个密钥位、字节流也按 2 字节/索引计，而不是被强制扩成 64 bit。
    注意：本演示仍每包从密钥起点复用同一前缀（无 offset 推进），非严格一次性密钥。
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("xor_int_ndarray only supports integer ndarray")
    itemsize = arr.dtype.itemsize
    if itemsize not in _UINT_VIEW:
        raise TypeError(f"unsupported integer itemsize: {itemsize}")

    bits_per = itemsize * 8
    uview = _UINT_VIEW[itemsize]
    arr_u = np.ascontiguousarray(arr).view(uview)
    total_bits = arr_u.size * bits_per
    key_bits = _load_key_bits(key_path)
    if len(key_bits) < total_bits:
        raise ValueError(f"key too short: need {total_bits} bits, got {len(key_bits)} bits")

    key_values = [int(key_bits[i : i + bits_per], 2) for i in range(0, total_bits, bits_per)]
    key_arr = np.array(key_values, dtype=uview).reshape(arr_u.shape)
    return np.bitwise_xor(arr_u, key_arr).view(arr.dtype)


def xor_encrypt_int_ndarray(arr: np.ndarray, key_path: str) -> np.ndarray:
    return xor_int_ndarray(arr, key_path)


def xor_decrypt_int_ndarray(arr: np.ndarray, key_path: str) -> np.ndarray:
    return xor_int_ndarray(arr, key_path)


def xor_bytes(data: bytes, key_path: str) -> bytes:
    """按字节流做 XOR（用于 10-bit 位打包后的紧凑字节流加解密）。

    位打包后整数元素边界消失，无法再按元素 XOR，故改为逐字节。
    注意：本演示仍每包从密钥起点复用同一前缀（无 offset 推进），非严格一次性密钥。
    """
    if not data:
        return b""
    key_bits = _load_key_bits(key_path)
    need_bits = len(data) * 8
    if len(key_bits) < need_bits:
        raise ValueError(f"key too short: need {need_bits} bits, got {len(key_bits)} bits")
    key_arr = np.array([1 if c == "1" else 0 for c in key_bits[:need_bits]], dtype=np.uint8)
    key_bytes = np.packbits(key_arr)
    data_arr = np.frombuffer(data, dtype=np.uint8)
    return np.bitwise_xor(data_arr, key_bytes).tobytes()
