#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

from common_protocol import read_message, write_message


def packet_wire_size(header: dict, body: bytes) -> int:
    header_with_len = dict(header)
    header_with_len["body_len"] = len(body)
    header_bytes = json.dumps(header_with_len, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return 4 + len(header_bytes) + len(body)


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


@dataclass
class ClientSession:
    user_id: str
    room_id: str
    writer: asyncio.StreamWriter
    send_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    start_time: float = field(default_factory=time.time)
    packets_in: int = 0
    packets_out: int = 0
    drops: int = 0
    body_bytes_in: int = 0
    body_bytes_out: int = 0
    total_bytes_in: int = 0
    total_bytes_out: int = 0


class RouterState:
    def __init__(self, room_id: str, max_queue: int):
        self.room_id = room_id
        self.max_queue = int(max_queue)
        self.start_time = time.time()
        self.sessions: Dict[Tuple[str, str], ClientSession] = {}
        self.total_packets_in = 0
        self.total_packets_out = 0
        self.total_drops = 0
        self.total_body_bytes_in = 0
        self.total_body_bytes_out = 0
        self.total_wire_bytes_in = 0
        self.total_wire_bytes_out = 0
        self.per_user: Dict[str, dict] = {}
        self.lock = asyncio.Lock()

    def _user_stats(self, user_id: str) -> dict:
        stats = self.per_user.get(user_id)
        if stats is None:
            stats = {
                "in": 0,
                "out": 0,
                "drops": 0,
                "body_in": 0,
                "body_out": 0,
                "wire_in": 0,
                "wire_out": 0,
            }
            self.per_user[user_id] = stats
        return stats

    async def register(self, user_id: str, room_id: str, writer: asyncio.StreamWriter) -> ClientSession:
        key = (room_id, user_id)
        async with self.lock:
            old = self.sessions.get(key)
            if old is not None:
                old.writer.close()
            session = ClientSession(
                user_id=user_id,
                room_id=room_id,
                writer=writer,
                send_queue=asyncio.Queue(maxsize=self.max_queue),
            )
            self.sessions[key] = session
            self._user_stats(user_id)
            return session

    async def unregister(self, session: ClientSession) -> None:
        async with self.lock:
            key = (session.room_id, session.user_id)
            if self.sessions.get(key) is session:
                del self.sessions[key]

    async def record_in(self, session: ClientSession, body_len: int, total_len: int) -> None:
        async with self.lock:
            session.packets_in += 1
            session.body_bytes_in += body_len
            session.total_bytes_in += total_len
            self.total_packets_in += 1
            self.total_body_bytes_in += body_len
            self.total_wire_bytes_in += total_len
            stats = self._user_stats(session.user_id)
            stats["in"] += 1
            stats["body_in"] += body_len
            stats["wire_in"] += total_len

    async def record_out(self, session: ClientSession, body_len: int, total_len: int) -> None:
        async with self.lock:
            session.packets_out += 1
            session.body_bytes_out += body_len
            session.total_bytes_out += total_len
            self.total_packets_out += 1
            self.total_body_bytes_out += body_len
            self.total_wire_bytes_out += total_len
            stats = self._user_stats(session.user_id)
            stats["out"] += 1
            stats["body_out"] += body_len
            stats["wire_out"] += total_len

    async def record_drop(self, session: ClientSession) -> None:
        async with self.lock:
            session.drops += 1
            self.total_drops += 1
            stats = self._user_stats(session.user_id)
            stats["drops"] += 1

    async def forward(self, source: ClientSession, header: dict, body: bytes) -> int:
        count = 0
        async with self.lock:
            targets = [
                session
                for (room_id, user_id), session in self.sessions.items()
                if room_id == source.room_id and user_id != source.user_id
            ]

        for target in targets:
            if target.send_queue.full():
                try:
                    target.send_queue.get_nowait()
                    await self.record_drop(target)
                except asyncio.QueueEmpty:
                    pass
            await target.send_queue.put((header, body))
            count += 1
        return count

    def online_count(self) -> int:
        return len(self.sessions)

    def elapsed_sec(self) -> float:
        return max(1e-6, time.time() - self.start_time)

    def global_body_kbps(self) -> float:
        return (self.total_body_bytes_in + self.total_body_bytes_out) * 8.0 / self.elapsed_sec() / 1000.0

    def global_total_kbps(self) -> float:
        return (self.total_wire_bytes_in + self.total_wire_bytes_out) * 8.0 / self.elapsed_sec() / 1000.0

    def print_global_summary(self) -> None:
        duration = self.elapsed_sec()
        drop_rate = self.total_drops / max(1, self.total_packets_out + self.total_drops)
        print(f"\n[ROUTER SUMMARY] duration={duration:.1f}s")
        print(
            f"[ROUTER SUMMARY] in={self.total_packets_in} out={self.total_packets_out} "
            f"drops={self.total_drops} drop_rate={fmt_pct(drop_rate)}"
        )
        print(
            f"[ROUTER SUMMARY] body={self.global_body_kbps():.1f}kbps "
            f"total={self.global_total_kbps():.1f}kbps"
        )
        if self.per_user:
            print("[ROUTER SUMMARY] per_user:")
            for user_id, stats in sorted(self.per_user.items()):
                user_drop_rate = stats["drops"] / max(1, stats["out"] + stats["drops"])
                print(
                    f"  user={user_id} in={stats['in']} out={stats['out']} "
                    f"drops={stats['drops']} drop_rate={fmt_pct(user_drop_rate)}"
                )


async def writer_task(session: ClientSession, state: RouterState) -> None:
    try:
        while True:
            header, body = await session.send_queue.get()
            total_len = packet_wire_size(header, body)
            await write_message(session.writer, header, body)
            await state.record_out(session, len(body), total_len)
    except (ConnectionError, asyncio.CancelledError, OSError):
        return


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, state: RouterState) -> None:
    peer = writer.get_extra_info("peername")
    session = None
    task = None
    try:
        hello, body = await read_message(reader)
        if hello.get("kind") != "hello_v1":
            raise ValueError("first message must be hello_v1")
        user_id = str(hello["user_id"])
        room_id = str(hello.get("room_id", state.room_id))
        if room_id != state.room_id:
            raise ValueError(f"unsupported room_id={room_id}; router room_id={state.room_id}")
        if body:
            raise ValueError("hello_v1 body must be empty")

        session = await state.register(user_id=user_id, room_id=room_id, writer=writer)
        task = asyncio.create_task(writer_task(session, state))
        print(f"[ROUTER] registered user={user_id} room={room_id} peer={peer}")

        while True:
            header, body = await read_message(reader)
            if header.get("kind") != "codes_v1":
                continue
            if header.get("room_id") != session.room_id:
                continue
            if header.get("sender_id") != session.user_id:
                continue

            total_in = packet_wire_size(header, body)
            await state.record_in(session, len(body), total_in)
            header = dict(header)
            header["router_forward_ts_ms"] = int(time.time() * 1000)
            forwarded = await state.forward(session, header, body)
            if session.packets_in % 20 == 0:
                drop_rate = state.total_drops / max(1, state.total_packets_out + state.total_drops)
                print(
                    f"[ROUTER] from={session.user_id} seq={header.get('seq_id')} "
                    f"forwarded={forwarded} online={state.online_count()} "
                    f"in={state.total_packets_in} out={state.total_packets_out} "
                    f"drop={state.total_drops} drop_rate={fmt_pct(drop_rate)} "
                    f"body={state.global_body_kbps():.1f}kbps total={state.global_total_kbps():.1f}kbps"
                )

    except asyncio.IncompleteReadError:
        pass
    except Exception as exc:
        print(f"[ROUTER] client {peer} error: {exc}")
    finally:
        if task is not None:
            task.cancel()
        if session is not None:
            await state.unregister(session)
            duration = max(1e-6, time.time() - session.start_time)
            drop_rate = session.drops / max(1, session.packets_out + session.drops)
            body_kbps = (session.body_bytes_in + session.body_bytes_out) * 8.0 / duration / 1000.0
            total_kbps = (session.total_bytes_in + session.total_bytes_out) * 8.0 / duration / 1000.0
            print(
                f"[ROUTER] disconnected user={session.user_id} "
                f"in={session.packets_in} out={session.packets_out} drops={session.drops} "
                f"duration={duration:.1f}s drop_rate={fmt_pct(drop_rate)} "
                f"body={body_kbps:.1f}kbps total={total_kbps:.1f}kbps"
            )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def main_async(args: argparse.Namespace) -> None:
    state = RouterState(room_id=args.room_id, max_queue=args.max_queue)
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, state),
        host=args.listen_ip,
        port=args.listen_port,
    )
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(f"[ROUTER] listening on {addrs} room={args.room_id} max_queue={args.max_queue}")
    try:
        async with server:
            await server.serve_forever()
    finally:
        state.print_global_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-user encrypted RVQ token router")
    parser.add_argument("--listen_ip", default="0.0.0.0")
    parser.add_argument("--listen_port", type=int, default=12350)
    parser.add_argument("--room_id", default="demo")
    parser.add_argument("--max_queue", type=int, default=8)
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
