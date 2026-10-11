"""Measured reconstruction scores; optional external PESQ worker for Windows."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path


def pesq_worker():
    import numpy as np
    import soundfile as sf
    from pesq import pesq
    for line in sys.stdin:
        try:
            request = json.loads(line)
            reference, sr = sf.read(request["reference"], dtype="float32")
            if request.get("estimate"):
                estimate, _ = sf.read(request["estimate"], dtype="float32")
            else:
                estimate = np.zeros(len(reference), dtype="float32")
            estimate = np.pad(estimate[:len(reference)], (0, max(0, len(reference) - len(estimate))))
            result = {"pesq_wb": float(pesq(sr, reference, estimate, "wb")), "pesq_status": "measured"}
        except Exception as error:
            result = {"pesq_wb": None, "pesq_status": "failed", "pesq_reason": repr(error)}
        print(json.dumps(result), flush=True)


class AudioScorer:
    def __init__(self, config, quality_python, log_path=None):
        self.config = config
        self.log_handle = None
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self.log_handle = Path(log_path).open("a", encoding="utf-8")
        self.process = subprocess.Popen([quality_python, "-u", str(Path(__file__).resolve()), "--pesq-worker"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_handle,
                                        text=True, encoding="utf-8")
        self.lines = queue.Queue()
        def read_output():
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)
        self.reader = threading.Thread(target=read_output, daemon=True)
        self.reader.start()

    def score(self, reference, estimate):
        import numpy as np
        import soundfile as sf
        import torch
        from pystoi import stoi
        from group_client import REPO_ROOT
        from speechtokenizer.trainer.loss import mel_loss
        torch.set_num_threads(1)
        reference_audio, sr = sf.read(str(reference), dtype="float32")
        reconstructed = sf.read(str(estimate), dtype="float32")[0] if estimate and Path(estimate).is_file() else np.zeros_like(reference_audio)
        reconstructed = np.pad(reconstructed[:len(reference_audio)], (0, max(0, len(reference_audio) - len(reconstructed))))
        result = {"stoi": None, "mel_l1": None, "evaluation_samples": len(reference_audio), "sample_rate": sr}
        for name, measure in (
            ("stoi", lambda: float(stoi(reference_audio, reconstructed, sr, extended=False))),
            ("mel_l1", lambda: float(mel_loss(torch.from_numpy(reference_audio).reshape(1, 1, -1),
                                            torch.from_numpy(reconstructed).reshape(1, 1, -1),
                                            n_fft=int(self.config["n_fft"]), num_mels=int(self.config["num_mels"]),
                                            sample_rate=sr, hop_size=int(self.config["hop_size"]),
                                            win_size=int(self.config["win_size"]), fmin=self.config.get("fmin", 0),
                                            fmax=self.config.get("fmax_for_loss")).item()))):
            try:
                value = measure()
                if not np.isfinite(value):
                    raise ValueError("metric returned nonfinite value")
                result.update({name: value, name + "_status": "measured"})
            except Exception as error:
                result.update({name + "_status": "failed", name + "_reason": repr(error)})
        self.process.stdin.write(json.dumps({"reference": str(reference), "estimate": str(estimate) if estimate and Path(estimate).is_file() else None}) + "\n")
        self.process.stdin.flush()
        diagnostics = []
        for _ in range(20):
            try:
                line = self.lines.get(timeout=30)
            except queue.Empty:
                self.process.kill()
                self.process.wait(timeout=5)
                result.update(pesq_wb=None, pesq_status="failed", pesq_reason="PESQ worker timed out after 30 seconds")
                break
            if not line:
                result.update(pesq_wb=None, pesq_status="failed", pesq_reason="PESQ subprocess exited")
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                diagnostics.append(line.strip())
                continue
            result.update(message)
            break
        else:
            result.update(pesq_wb=None, pesq_status="failed", pesq_reason="Invalid PESQ worker output")
        if diagnostics:
            result["pesq_worker_diagnostics"] = ";".join(diagnostics)
        return result

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.reader.join(timeout=5)
        if self.process.stdout:
            self.process.stdout.close()
        if self.log_handle:
            self.log_handle.close()


if __name__ == "__main__" and "--pesq-worker" in sys.argv:
    pesq_worker()
