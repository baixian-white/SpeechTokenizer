#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import json
import socket
import struct
from typing import Tuple


MAX_HEADER_BYTES = 64 * 1024
MAX_BODY_BYTES = 32 * 1024 * 1024


def pack_message(header: dict, body: bytes) -> bytes:
    header = dict(header)
    header["body_len"] = len(body)
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError(f"header too large: {len(header_bytes)} bytes")
    if len(body) > MAX_BODY_BYTES:
        raise ValueError(f"body too large: {len(body)} bytes")
    return struct.pack("!I", len(header_bytes)) + header_bytes + body


def send_message(sock: socket.socket, header: dict, body: bytes) -> None:
    data = pack_message(header, body)
    view = memoryview(data)
    while view:
        sent = sock.send(view)
        if sent <= 0:
            raise ConnectionError("socket send failed")
        view = view[sent:]


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while receiving")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Tuple[dict, bytes]:
    header_len = struct.unpack("!I", recv_exact(sock, 4))[0]
    if header_len <= 0 or header_len > MAX_HEADER_BYTES:
        raise ValueError(f"invalid header length: {header_len}")
    header = json.loads(recv_exact(sock, header_len).decode("utf-8"))
    body_len = int(header.get("body_len", 0))
    if body_len < 0 or body_len > MAX_BODY_BYTES:
        raise ValueError(f"invalid body length: {body_len}")
    body = recv_exact(sock, body_len)
    return header, body


async def read_message(reader: asyncio.StreamReader) -> Tuple[dict, bytes]:
    header_len_bytes = await reader.readexactly(4)
    header_len = struct.unpack("!I", header_len_bytes)[0]
    if header_len <= 0 or header_len > MAX_HEADER_BYTES:
        raise ValueError(f"invalid header length: {header_len}")
    header = json.loads((await reader.readexactly(header_len)).decode("utf-8"))
    body_len = int(header.get("body_len", 0))
    if body_len < 0 or body_len > MAX_BODY_BYTES:
        raise ValueError(f"invalid body length: {body_len}")
    body = await reader.readexactly(body_len)
    return header, body


async def write_message(writer: asyncio.StreamWriter, header: dict, body: bytes) -> None:
    writer.write(pack_message(header, body))
    await writer.drain()
