"""Measured WAV replay client, using the existing codec, packet format and Exp23."""

from __future__ import annotations

import argparse
import queue
import socket
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from metrics_protocol import (BLOCK, SR, Journal, pairing_window, prediction_windows,
                              read_json, timeline_blocks, write_json)


class MeasuredClient:
    def __init__(self, job, user, output, fake_codec=False):
        import numpy as np
        self.np = np
        self.job, self.user = job, user
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.output / (user + ".jsonl"))
        self.stop = threading.Event()
        self.model_lock = threading.Lock()
        self.send_done = threading.Event()
        self.recv_done = threading.Event()
        self.queue = queue.Queue(maxsize=job["queue_blocks"])
        self.seen = set()
        self.parts = {}
        self.audio = {}
        self.errors = []
        self.socket = None
        self.identifier = None
        self.gpu_worker = None
        self.fake_codec = fake_codec
        self.play_queues = {}
        self.play_lock = threading.Lock()
        self.output_stream = None
        self.underruns = 0
        self.origin = None
        self.blocks = {s: {b["seq_id"]: b for b in timeline_blocks(parts)}
                       for s, parts in job["streams"].items()}
        self.windows = {s: {w["window_end_sample"]: w for w in prediction_windows(parts, job["continuous"])}
                        for s, parts in job["streams"].items()}
        self.resources_thread = threading.Thread(target=self.sample_resources, name="resources")

    def emit(self, event, **values):
        return self.journal.emit(event, user_id=self.user, **values)

    def sample_resources(self):
        import psutil
        proc = psutil.Process()
        proc.cpu_percent()
        while not self.stop.wait(self.job["resource_interval_seconds"]):
            with self.play_lock:
                play_size = sum(sum(len(p[0]) - p[2] for p in q) for q in self.play_queues.values())
            self.emit("resource", cpu_percent=proc.cpu_percent(), rss_bytes=proc.memory_info().rss,
                      system_cpu_percent=psutil.cpu_percent(), system_available_bytes=psutil.virtual_memory().available,
                      system_used_memory_bytes=psutil.virtual_memory().total - psutil.virtual_memory().available,
                      decode_queue_blocks=self.queue.qsize(), playback_queue_samples=play_size,
                      playback_underruns=self.underruns, gpu_status="not_used_cpu_run",
                      gpu_percent=None, vram_bytes=None)

    def load(self):
        import torch
        from group_client import GroupClient
        from speechtokenizer.speaker_identity.inference import SpeakerClassifierBundle
        candidate_labels = tuple(self.job.get("classifier_candidates") or ())
        torch.set_num_threads(self.job["torch_threads"])
        if self.job.get("compute_backend") == "gpu_worker":
            from gpu_client import GpuWorkerClient
            self.gpu_worker = GpuWorkerClient(self.job["worker_host"], self.job["worker_port"])
            health = self.gpu_worker.request("health", {})
            self.bits = 10
            self.codec = None
            if self.job["recognition"] and not self.fake_codec:
                self.identifier = "gpu_worker"
            self.emit("model_ready", fake_codec=self.fake_codec, classifier_enabled=bool(self.identifier),
                      recognition_requested=self.job["recognition"], device=health.get("device"),
                      compute_backend="shared_gpu_worker", classifier_device=health.get("classifier_device"),
                      classifier_output_space=health.get("classifier_output_space"),
                      classifier_candidates=list(candidate_labels))
            return
        self.engine = GroupClient(SimpleNamespace(user_id=self.user, room_id=self.job["job_id"],
                                                 stage1_config="", device="cpu",
                                                 config_path=self.job["model_config"], ckpt_path=self.job["checkpoint"]))
        if self.fake_codec:
            self.codec = None
            self.bits = 10
        else:
            self.engine.load_model()
            self.codec = self.engine.model
            self.bits = self.engine.index_bits
        if self.job["recognition"] and not self.fake_codec:
            self.identifier = SpeakerClassifierBundle(
                self.job["bundle"],
                device="cpu",
                threshold=.65,
                candidate_labels=candidate_labels or None,
            )
        self.emit("model_ready", fake_codec=self.fake_codec,
                  classifier_enabled=self.identifier is not None, recognition_requested=self.job["recognition"],
                  device="cpu",
                  classifier_output_space=(self.identifier.output_speaker_count if self.identifier else None),
                  classifier_candidates=list(candidate_labels))

    def connect(self):
        from common_protocol import send_message
        if self.job["transport"] == "local":
            return
        self.socket = socket.create_connection(("127.0.0.1", self.job["port"]), timeout=20)
        self.socket.settimeout(None)
        send_message(self.socket, {"kind": "hello_v1", "room_id": self.job["job_id"], "user_id": self.user}, b"")
        self.emit("connected")

    def gate(self):
        gate_path = Path(self.job["gate_path"])
        deadline = time.monotonic() + self.job["startup_timeout_seconds"]
        while not gate_path.is_file():
            if time.monotonic() > deadline or self.stop.wait(.05):
                raise TimeoutError("start barrier did not open")
        self.origin = int(read_json(gate_path)["origin_ns"])

    def wait_until(self, target):
        while time.perf_counter_ns() < target:
            if self.stop.wait(min(.05, (target - time.perf_counter_ns()) / 1e9)):
                return False
        return not self.stop.is_set()

    def send_loop(self):
        import torch
        import soundfile as sf
        from group_client import pack_indices_bits, packet_wire_size
        from common_protocol import send_message
        from crypto_utils import xor_bytes
        try:
            segments = self.job["streams"].get(self.user, [])
            current, wav = None, None
            for info in self.blocks.get(self.user, {}).values():
                if self.stop.is_set():
                    break
                if current != info["segment_index"]:
                    current = info["segment_index"]
                    wav, sr = sf.read(segments[current]["wav_path"], dtype="float32")
                    if sr != SR or wav.ndim != 1:
                        raise ValueError("frozen WAV must be 16 kHz mono")
                lo, hi = info["source_start_sample"], info["source_end_sample"]
                source_start = self.origin + round(lo / SR * 1e9)
                source_end = self.origin + round(hi / SR * 1e9)
                if not self.wait_until(source_end):
                    break
                chunk = wav[info["segment_offset"]:info["segment_offset"] + BLOCK]
                lock_request = time.perf_counter_ns()
                with self.model_lock, torch.inference_mode():
                    begin = time.perf_counter_ns()
                    if self.fake_codec:
                        codes = self.np.zeros((3, 25), dtype="int64")
                    elif self.gpu_worker:
                        response = self.gpu_worker.request("encode", {"wav": chunk})
                        codes = response["codes"]
                        begin = response["device_start_ns"]
                        finish = response["device_end_ns"]
                        self.emit("gpu_compute", op="encode", device=response.get("device"),
                                  device_start_ns=begin, device_end_ns=finish,
                                  gpu_allocated_bytes=response.get("gpu_allocated_bytes"),
                                  gpu_reserved_bytes=response.get("gpu_reserved_bytes"))
                    else:
                        codes = self.codec.encode(torch.from_numpy(chunk).reshape(1, 1, -1))[:3, 0].cpu().numpy()
                        finish = time.perf_counter_ns()
                body = xor_bytes(pack_indices_bits(codes, self.bits), self.job["key_path"])
                header = {"kind": "codes_v1", "room_id": self.job["job_id"], "sender_id": self.user,
                          "seq_id": info["seq_id"], "packing": "bitpack_v1", "rvq_layers": 3,
                          "bits": self.bits, "n_values": int(codes.size), "shape": list(codes.shape),
                          "source_start_sample": lo, "source_end_sample": hi}
                start_send = time.perf_counter_ns()
                if self.socket:
                    send_message(self.socket, header, body)
                end_send = time.perf_counter_ns()
                total = packet_wire_size(header, body) if self.socket else 0
                self.emit("send", sender_id=self.user, seq_id=info["seq_id"], stream_id=self.user,
                          source_start_sample=lo, source_end_sample=hi, source_start_ns=source_start,
                          source_end_ns=source_end, encode_start_ns=begin, encode_end_ns=finish,
                          codec_lock_wait_ms=(begin - lock_request) / 1e6,
                          send_start_ns=start_send, send_end_ns=end_send, payload_bytes=len(body) if self.socket else 0,
                          application_bytes=total, status="sent" if self.socket else "local_encoded")
                if not self.socket:
                    self.enqueue(header, body)
            self.emit("send_finished", expected_blocks=len(self.blocks.get(self.user, {})))
        except Exception as error:
            self.errors.append("send: " + repr(error))
            self.emit("worker_error", reason=self.errors[-1])
        finally:
            self.send_done.set()

    def enqueue(self, header, body):
        from group_client import packet_wire_size
        sender, seq = header["sender_id"], int(header["seq_id"])
        if sender not in self.blocks or seq not in self.blocks[sender]:
            self.emit("invalid_packet", reason="outside frozen source timeline", sender_id=sender, seq_id=seq)
            return
        if (sender, seq) in self.seen:
            self.emit("duplicate", sender_id=sender, seq_id=seq)
            return
        self.seen.add((sender, seq))
        received = time.perf_counter_ns()
        self.emit("receive", sender_id=sender, receiver_id=self.user, stream_id=sender, seq_id=seq,
                  receive_ns=received, payload_bytes=len(body) if self.socket else 0,
                  application_bytes=packet_wire_size(header, body) if self.socket else 0)
        try:
            self.queue.put_nowait((header, body, received))
        except queue.Full:
            self.emit("block_failure", sender_id=sender, receiver_id=self.user, seq_id=seq,
                      status="receiver_queue_drop", reason="bounded decode queue full")

    def recv_loop(self):
        from common_protocol import recv_message
        try:
            while not self.stop.is_set():
                header, body = recv_message(self.socket)
                if header.get("kind") == "codes_v1":
                    self.enqueue(header, body)
        except (OSError, ConnectionError) as error:
            if not self.stop.is_set():
                self.emit("disconnect", reason=repr(error))
                self.errors.append("unexpected disconnect")
        finally:
            self.recv_done.set()

    def decode_loop(self):
        import torch
        from group_client import unpack_indices_bits
        from crypto_utils import xor_bytes
        while not self.stop.is_set() or not self.queue.empty():
            try:
                header, body, received = self.queue.get(timeout=.1)
            except queue.Empty:
                continue
            sender, seq = header["sender_id"], int(header["seq_id"])
            info = self.blocks[sender][seq]
            source_end = self.origin + round(info["source_end_sample"] / SR * 1e9)
            expiry = source_end + round(self.job["block_deadline_seconds"] * 1e9)
            try:
                if time.perf_counter_ns() > expiry:
                    raise TimeoutError("block deadline expired in receive queue")
                codes = unpack_indices_bits(xor_bytes(body, self.job["key_path"]), int(header["bits"]),
                                            int(header["n_values"])).reshape(header["shape"])
                lock_request = time.perf_counter_ns()
                with self.model_lock, torch.inference_mode():
                    begin = time.perf_counter_ns()
                    if begin > expiry:
                        raise TimeoutError("block deadline expired while waiting for codec")
                    if self.fake_codec:
                        pcm = self.np.zeros(BLOCK, dtype="float32")
                    elif self.gpu_worker:
                        response = self.gpu_worker.request("decode", {"codes": codes})
                        pcm = response["pcm"]
                        begin = response["device_start_ns"]
                        finish = response["device_end_ns"]
                        self.emit("gpu_compute", op="decode", device=response.get("device"),
                                  device_start_ns=begin, device_end_ns=finish,
                                  gpu_allocated_bytes=response.get("gpu_allocated_bytes"),
                                  gpu_reserved_bytes=response.get("gpu_reserved_bytes"))
                    else:
                        pcm = self.codec.decode(torch.from_numpy(codes).reshape(3, 1, -1)).flatten().cpu().numpy()
                        finish = time.perf_counter_ns()
                actual_samples = len(pcm)
                pcm = self.np.pad(pcm[:BLOCK], (0, max(0, BLOCK - len(pcm)))).astype("float32")
                self.emit("decode", sender_id=sender, receiver_id=self.user, seq_id=seq,
                          decode_start_ns=begin, decode_end_ns=finish, codec_lock_wait_ms=(begin - lock_request) / 1e6,
                          received_samples=actual_samples, token_frames=int(codes.shape[-1]),
                          status="decoded" if finish <= expiry else "late_decoded")
                self.parts.setdefault(sender, {})[seq] = (pcm, codes)
                self.save_audio(sender, seq, info, pcm)
                if self.output_stream:
                    self.add_playback(sender, seq, pcm)
                window = self.windows[sender].get(info["source_end_sample"])
                if window and self.job["recognition"]:
                    pair = pairing_window(self.parts[sender], window["window_start_sample"], window["window_end_sample"])
                    if pair:
                        start_infer = time.perf_counter_ns()
                        try:
                            if self.gpu_worker and self.identifier == "gpu_worker":
                                response = self.gpu_worker.request("predict", {"pcm": pair[0], "codes": pair[1]})
                                item = response.get("prediction")
                                prediction = SimpleNamespace(**item) if item else None
                                self.emit("gpu_compute", op="predict", device=response.get("classifier_device"),
                                          device_start_ns=response.get("device_start_ns"),
                                          device_end_ns=response.get("device_end_ns"))
                            else:
                                prediction = self.identifier.predict(pair[0], codes=pair[1]) if self.identifier else None
                            end_infer = time.perf_counter_ns()
                            label = prediction.predicted_speaker if prediction else "fixture_identity"
                            status = "predicted" if end_infer <= source_end + self.job["prediction_deadline_seconds"] * 1e9 else "late_prediction"
                            if self.fake_codec:
                                status = "fixture"
                            self.emit("prediction", sender_id=sender, receiver_id=self.user,
                                      window_index=window["window_index"], inference_start_ns=start_infer,
                                      output_ns=end_infer, status=status, predicted_speaker=label,
                                      score=prediction.score if prediction else 0.0,
                                      margin=prediction.margin if prediction else 0.0,
                                      verified=prediction.verified if prediction else False,
                                      paired_samples=len(pair[0]), paired_token_frames=pair[1].shape[-1])
                        except Exception as error:
                            self.emit("prediction", sender_id=sender, receiver_id=self.user,
                                      window_index=window["window_index"], status="inference_failed", reason=repr(error))
                # Windows use complete paired blocks. Keep only the next three seconds.
                minimum_seq = seq - 6
                self.parts[sender] = {s: p for s, p in self.parts[sender].items() if s >= minimum_seq}
            except Exception as error:
                self.emit("block_failure", sender_id=sender, receiver_id=self.user, seq_id=seq,
                          status="deadline_expired" if isinstance(error, TimeoutError) else "decode_failed", reason=repr(error))
            finally:
                self.queue.task_done()

    def save_audio(self, sender, seq, info, pcm):
        import soundfile as sf
        key = (sender, info["segment_index"])
        if key not in self.audio:
            self.audio[key] = self.np.zeros(self.job["streams"][sender][key[1]]["samples"], dtype="float32")
        offset = info["segment_offset"]
        self.audio[key][offset:offset + BLOCK] = pcm
        if offset + BLOCK == len(self.audio[key]):
            path = self.output.parent / "audio" / self.user / sender / (str(key[1]).zfill(5) + ".wav")
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), self.audio.pop(key), SR, subtype="FLOAT")

    def add_playback(self, sender, seq, pcm):
        with self.play_lock:
            q = self.play_queues.setdefault(sender, deque())
            q.append([pcm, seq, 0])
            while sum(len(p[0]) - p[2] for p in q) > self.job["playback_max_seconds"] * SR:
                dropped = q.popleft()
                self.emit("playback_drop", sender_id=sender, receiver_id=self.user, seq_id=dropped[1], reason="playback buffer limit")

    def start_playback(self):
        if self.job["playback"] == "off":
            return
        import sounddevice as sd
        def callback(outdata, frames, timing, status):
            mixed = self.np.zeros(frames, dtype="float32")
            now = time.perf_counter_ns()
            with self.play_lock:
                for sender, q in self.play_queues.items():
                    filled = 0
                    while q and filled < frames:
                        item = q[0]
                        count = min(frames - filled, len(item[0]) - item[2])
                        mixed[filled:filled + count] += item[0][item[2]:item[2] + count]
                        item[2] += count
                        filled += count
                        if item[2] == len(item[0]):
                            q.popleft()
                            self.emit("playback_callback", sender_id=sender, receiver_id=self.user, seq_id=item[1],
                                      callback_ns=now, device_dac_time=timing.outputBufferDacTime)
                    if filled < frames:
                        self.underruns += 1
                if status:
                    self.emit("playback_status", reason=str(status))
            outdata[:, 0] = self.np.clip(mixed / max(1, len(self.play_queues)), -1, 1)
        self.output_stream = sd.OutputStream(samplerate=SR, channels=1, blocksize=320, callback=callback)
        self.output_stream.start()

    def run(self):
        import soundfile as sf
        threads = []
        self.resources_thread.start()
        try:
            self.emit("client_started")
            self.load()
            self.connect()
            self.start_playback()
            write_json(self.output / (self.user + ".ready.json"), {"ready": True, "user": self.user})
            self.gate()
            decoder = threading.Thread(target=self.decode_loop, name="decode")
            sender = threading.Thread(target=self.send_loop, name="send")
            threads = [decoder, sender]
            if self.socket:
                threads.append(threading.Thread(target=self.recv_loop, name="receive"))
            for thread in threads:
                thread.start()
            expected = sum(len(bs) for s, bs in self.blocks.items() if s != self.user or not self.socket)
            source_samples = max((sum(s["samples"] for s in segs) for segs in self.job["streams"].values()), default=0)
            deadline = self.origin + round((source_samples / SR + self.job["drain_seconds"]) * 1e9)
            while time.perf_counter_ns() < deadline:
                if self.send_done.is_set() and len(self.seen) >= expected and self.queue.unfinished_tasks == 0:
                    if self.output_stream:
                        time.sleep(self.job["playback_max_seconds"])
                    break
                if self.errors:
                    break
                time.sleep(.1)
            else:
                self.emit("timeout", expected_blocks=expected, received_blocks=len(self.seen))
                self.errors.append("session watchdog expired")
        except KeyboardInterrupt:
            self.errors.append("interrupted")
            self.emit("interrupted", reason="user interrupted the client")
        except Exception as error:
            self.errors.append(repr(error))
            self.emit("worker_error", reason=repr(error))
        finally:
            self.stop.set()
            if self.socket:
                try:
                    self.socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            for thread in threads + [self.resources_thread]:
                thread.join(timeout=15)
                if thread.is_alive():
                    self.errors.append("worker did not exit: " + thread.name)
            if self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()
            if self.socket:
                self.socket.close()
            if self.gpu_worker:
                self.gpu_worker.close()
            for (sender, index), pcm in self.audio.items():
                path = self.output.parent / "audio" / self.user / sender / (str(index).zfill(5) + ".wav")
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), pcm, SR, subtype="FLOAT")
            self.emit("client_exit", status="failed" if self.errors else "complete", errors=self.errors,
                      received_blocks=len(self.seen), playback_underruns=self.underruns)
            write_json(self.output / (self.user + ".done.json"), {"errors": self.errors, "exit_code": int(bool(self.errors))})
            self.journal.close()
        return int(bool(self.errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true", help="Synthetic codec for integration tests only")
    args = parser.parse_args()
    return MeasuredClient(read_json(args.job), args.user, args.output, args.fixture).run()


if __name__ == "__main__":
    raise SystemExit(main())
