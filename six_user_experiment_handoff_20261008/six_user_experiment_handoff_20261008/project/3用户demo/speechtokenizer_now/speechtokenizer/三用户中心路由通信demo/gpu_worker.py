"""Shared CUDA worker for six-client measurements.

One LCA instance is kept on the GPU. Clients remain independent Router peers and
submit encode/decode/predict jobs over a local control socket. Exp23 is kept on
CPU until its TorchScript bundle is re-exported with device-safe constants.
"""

from __future__ import annotations

import argparse
import pickle
import queue
import socketserver
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import torch

from common_protocol import recv_message, send_message
from metrics_protocol import Journal, write_json


class Request:
    def __init__(self, request_id, op, payload):
        self.request_id = request_id
        self.op = op
        self.payload = payload
        self.enqueued_ns = time.perf_counter_ns()
        self.event = threading.Event()
        self.result = None
        self.error = None


class SharedGpuWorker:
    def __init__(self, args):
        self.args = args
        self.queue = queue.Queue(maxsize=args.queue_size)
        self.stop = threading.Event()
        self.model_lock = threading.Lock()
        self.journal = Journal(Path(args.output) / "worker.jsonl")
        self.requests = 0
        self.model = None
        self.index_bits = 10
        self.classifier = None
        self.classifier_device = "disabled"
        self.classifier_output_space = None

    def emit(self, event, **values):
        return self.journal.emit(event, user_id="gpu_worker", **values)

    def load(self):
        from group_client import GroupClient
        model_args = SimpleNamespace(user_id="gpu_worker", room_id="gpu_worker", stage1_config="",
                                     device="cuda", config_path=self.args.config, ckpt_path=self.args.checkpoint)
        engine = GroupClient(model_args)
        engine.load_model()
        if engine.device.type != "cuda":
            raise RuntimeError("GPU worker requested CUDA, but PyTorch cannot use CUDA")
        self.model = engine.model
        self.index_bits = engine.index_bits
        if self.args.bundle:
            from speechtokenizer.speaker_identity.inference import SpeakerClassifierBundle
            candidate_labels = tuple(
                part.strip() for part in str(self.args.candidate_labels or "").split(",") if part.strip()
            )
            self.classifier = SpeakerClassifierBundle(
                self.args.bundle,
                device="cpu",
                threshold=.65,
                candidate_labels=candidate_labels or None,
            )
            self.classifier_device = "cpu_fallback_torchscript_constants"
            self.classifier_output_space = self.classifier.output_speaker_count
        self.emit("model_ready", device="cuda:0", classifier_device=self.classifier_device,
                  torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(0), classifier_enabled=self.classifier is not None,
                  classifier_output_space=self.classifier_output_space)

    def encode(self, payload):
        wav = np.asarray(payload["wav"], dtype=np.float32).reshape(1, 1, -1)
        value = torch.from_numpy(wav).to("cuda")
        with self.model_lock, torch.inference_mode():
            start = time.perf_counter_ns()
            codes = self.model.encode(value)[:3, 0].to("cpu").numpy().astype(np.int64)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
        return {"codes": codes, "device_start_ns": start, "device_end_ns": end,
                "device": "cuda:0", "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                "gpu_reserved_bytes": torch.cuda.memory_reserved()}

    def decode(self, payload):
        codes = torch.from_numpy(np.asarray(payload["codes"], dtype=np.int64)).reshape(3, 1, -1).to("cuda")
        with self.model_lock, torch.inference_mode():
            start = time.perf_counter_ns()
            pcm = self.model.decode(codes).flatten().to("cpu").numpy().astype(np.float32)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
        return {"pcm": pcm, "device_start_ns": start, "device_end_ns": end,
                "device": "cuda:0", "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                "gpu_reserved_bytes": torch.cuda.memory_reserved()}

    def predict(self, payload):
        if self.classifier is None:
            return {"prediction": None, "classifier_device": "disabled"}
        start = time.perf_counter_ns()
        prediction = self.classifier.predict(payload["pcm"], codes=payload["codes"])
        end = time.perf_counter_ns()
        return {"prediction": {"predicted_speaker": prediction.predicted_speaker, "score": prediction.score,
                               "margin": prediction.margin, "verified": prediction.verified},
                "classifier_device": self.classifier_device, "device_start_ns": start, "device_end_ns": end}

    def process(self, request):
        started = time.perf_counter_ns()
        self.requests += 1
        self.emit("gpu_request", request_id=request.request_id, op=request.op,
                  queue_wait_ms=(started - request.enqueued_ns) / 1e6, queue_depth=self.queue.qsize())
        try:
            if request.op == "encode":
                result = self.encode(request.payload)
            elif request.op == "decode":
                result = self.decode(request.payload)
            elif request.op == "predict":
                result = self.predict(request.payload)
            elif request.op == "health":
                result = {"device": "cuda:0", "classifier_device": self.classifier_device,
                          "classifier_output_space": self.classifier_output_space}
            else:
                raise ValueError("unknown GPU worker operation: " + request.op)
            request.result = result
            self.emit("gpu_complete", request_id=request.request_id, op=request.op,
                      total_ms=(time.perf_counter_ns() - started) / 1e6,
                      gpu_allocated_bytes=torch.cuda.memory_allocated(), gpu_reserved_bytes=torch.cuda.memory_reserved())
        except Exception as error:
            request.error = repr(error)
            self.emit("gpu_error", request_id=request.request_id, op=request.op, reason=request.error)
        finally:
            request.event.set()

    def loop(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                request = self.queue.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                self.process(request)
            finally:
                self.queue.task_done()

    def resource_loop(self):
        process = psutil.Process()
        while not self.stop.wait(1.0):
            gpu_util = None
            gpu_mem = None
            try:
                probe = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
                if probe.returncode == 0 and probe.stdout.strip():
                    values = [v.strip() for v in probe.stdout.strip().split(",")]
                    gpu_util, gpu_mem = float(values[0]), float(values[1]) * 1024 * 1024
            except Exception:
                pass
            self.emit("resource", phase="gpu_worker", cpu_percent=process.cpu_percent(),
                      rss_bytes=process.memory_info().rss, system_cpu_percent=psutil.cpu_percent(),
                      system_available_bytes=psutil.virtual_memory().available,
                      system_used_memory_bytes=psutil.virtual_memory().total - psutil.virtual_memory().available,
                      decode_queue_blocks=self.queue.qsize(), playback_queue_samples=0, playback_underruns=0,
                      gpu_status="cuda", gpu_percent=gpu_util, vram_bytes=gpu_mem,
                      gpu_allocated_bytes=torch.cuda.memory_allocated(), gpu_reserved_bytes=torch.cuda.memory_reserved(),
                      requests=self.requests)

    def serve(self):
        self.load()
        write_json(Path(self.args.output) / "worker_ready.json",
                   {"ready": True, "device": "cuda:0", "classifier_device": self.classifier_device,
                    "gpu": torch.cuda.get_device_name(0)})
        threading.Thread(target=self.loop, daemon=True, name="gpu-queue").start()
        threading.Thread(target=self.resource_loop, daemon=True, name="gpu-resources").start()
        worker = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                while not worker.stop.is_set():
                    header, body = recv_message(self.request)
                    if header.get("kind") != "gpu_request":
                        continue
                    request = Request(str(header.get("request_id")), str(header.get("op")), pickle.loads(body))
                    try:
                        worker.queue.put(request, timeout=10)
                    except queue.Full:
                        send_message(self.request, {"kind": "gpu_response", "request_id": request.request_id,
                                                    "status": "queue_full", "error": "GPU worker queue full"}, b"")
                        continue
                    if not request.event.wait(60):
                        send_message(self.request, {"kind": "gpu_response", "request_id": request.request_id,
                                                    "status": "timeout", "error": "GPU worker request timeout"}, b"")
                        continue
                    if request.error:
                        send_message(self.request, {"kind": "gpu_response", "request_id": request.request_id,
                                                    "status": "error", "error": request.error}, b"")
                    else:
                        send_message(self.request, {"kind": "gpu_response", "request_id": request.request_id,
                                                    "status": "ok", "op": request.op}, pickle.dumps(request.result, protocol=5))

        with socketserver.ThreadingTCPServer(("127.0.0.1", self.args.port), Handler) as tcp:
            tcp.daemon_threads = True
            write_json(Path(self.args.output) / "worker_listening.json", {"port": self.args.port})
            server_thread = threading.Thread(target=tcp.serve_forever, daemon=True, name="gpu-rpc-server")
            server_thread.start()
            try:
                while not self.stop.wait(.5):
                    if self.args.stop_file and Path(self.args.stop_file).exists():
                        break
            finally:
                self.stop.set()
                tcp.shutdown()
                server_thread.join(timeout=5)
        self.journal.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bundle", default="")
    parser.add_argument("--candidate-labels", default="",
                        help="Optional comma-separated known-roster labels for Exp23 inference")
    parser.add_argument("--stop-file", default="")
    parser.add_argument("--queue-size", type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    SharedGpuWorker(args).serve()


if __name__ == "__main__":
    main()
