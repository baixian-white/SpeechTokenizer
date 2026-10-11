from __future__ import annotations

import pickle
import socket
import threading
import uuid

from common_protocol import recv_message, send_message


class GpuWorkerClient:
    def __init__(self, host: str, port: int):
        self.socket = socket.create_connection((host, int(port)), timeout=30)
        self.socket.settimeout(120)
        self.lock = threading.Lock()

    def request(self, op: str, payload: dict):
        request_id = uuid.uuid4().hex
        with self.lock:
            send_message(self.socket, {"kind": "gpu_request", "request_id": request_id, "op": op},
                         pickle.dumps(payload, protocol=5))
            header, body = recv_message(self.socket)
        if header.get("request_id") != request_id:
            raise RuntimeError("GPU worker response request_id mismatch")
        if header.get("status") != "ok":
            raise RuntimeError(header.get("error", "GPU worker request failed"))
        return pickle.loads(body)

    def close(self):
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
