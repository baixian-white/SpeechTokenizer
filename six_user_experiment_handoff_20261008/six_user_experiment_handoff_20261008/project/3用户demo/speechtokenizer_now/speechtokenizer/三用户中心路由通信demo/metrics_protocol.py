"""Frozen source timelines and append-only measurement records for system trials."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import threading
import time
from collections import Counter
from pathlib import Path

USERS = dict(zip("ABCDEF", ("p225", "p226", "p227", "p228", "p229", "p230")))
SR = 16000
BLOCK = 8000
WINDOW = 48000
HOP = 16000


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class Journal:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def emit(self, event, **values):
        row = {"event": event, "event_ns": time.perf_counter_ns(), **values}
        with self.lock:
            self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def close(self):
        with self.lock:
            self.handle.close()


def read_events(directory):
    rows = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A killed process can leave its last line incomplete.
                    rows.append({"event": "record_error", "file": str(path), "line": line_number})
    return rows


def timeline_blocks(segments):
    start = 0
    for index, segment in enumerate(segments):
        for offset in range(0, int(segment["samples"]), BLOCK):
            yield {"seq_id": (start + offset) // BLOCK, "segment_index": index,
                   "source_start_sample": start + offset, "source_end_sample": start + offset + BLOCK,
                   "segment_offset": offset, "utterance_group": segment["utterance_group"],
                   "expected_speaker": segment["speaker_id"]}
        start += int(segment["samples"])


def prediction_windows(segments, continuous=False):
    """Declare windows before reception; overlapping transition windows remain visible."""
    if not continuous:
        start = 0
        for index, segment in enumerate(segments):
            yield {"window_index": index, "window_start_sample": start,
                   "window_end_sample": start + WINDOW, "segment_index": index,
                   "expected_speaker": segment["speaker_id"], "utterance_group": segment["utterance_group"],
                   "transition": False}
            start += int(segment["samples"])
        return
    spans = []
    start = 0
    for index, segment in enumerate(segments):
        spans.append((start, start + int(segment["samples"]), index, segment))
        start = spans[-1][1]
    for number, end in enumerate(range(WINDOW, start + 1, HOP)):
        covered = [(i, item) for lo, hi, i, item in spans if lo < end and hi > end - WINDOW]
        index, item = covered[-1]
        yield {"window_index": number, "window_start_sample": end - WINDOW,
               "window_end_sample": end, "segment_index": index,
               "expected_speaker": item["speaker_id"], "utterance_group": item["utterance_group"],
               "transition": len({part["speaker_id"] for _, part in covered}) > 1}


def trial_jobs(condition, scenario, selected_users=None):
    users = list(selected_users or USERS)
    groups = [users] if condition != "B" else [users[:3], users[3:]]
    if condition == "A":
        groups = [[user] for user in users]
    jobs = []
    for group in filter(None, groups):
        active_sets = [[u] for u in group] if scenario == "sequential" else [group]
        for active in active_sets:
            jobs.append({"condition": condition, "scenario": scenario, "users": group,
                         "active_senders": active, "recognition": condition != "C",
                         "transport": "local" if condition == "A" else "tcp_loopback"})
    return jobs


def expected_links(job):
    if job["transport"] == "local":
        return [(u, u) for u in job["active_senders"]]
    return [(sender, receiver) for sender in job["active_senders"]
            for receiver in job["users"] if receiver != sender]


def quantiles(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    def percentile(p):
        position = (len(values) - 1) * p
        lo = int(position)
        fraction = position - lo
        return values[lo] * (1 - fraction) + values[min(lo + 1, len(values) - 1)] * fraction
    return {"count": len(values), "mean": sum(values) / len(values),
            "median": percentile(.5), "p95": percentile(.95)}


def recognition_metrics(rows, labels=None):
    labels = list(labels or USERS.values())
    expected = len(rows)
    valid = [r for r in rows if r.get("status") == "predicted"]
    correct = sum(r["expected_speaker"] == r.get("predicted_speaker") for r in valid)
    f1s = []
    confusion = Counter()
    for label in labels:
        tp = sum(r["expected_speaker"] == label and r.get("predicted_speaker") == label
                 and r.get("status") == "predicted" for r in rows)
        fp = sum(r["expected_speaker"] != label and r.get("predicted_speaker") == label
                 and r.get("status") == "predicted" for r in rows)
        fn = sum(r["expected_speaker"] == label for r in rows) - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    for row in rows:
        prediction = row.get("predicted_speaker", "") if row.get("status") == "predicted" else "no_output"
        if prediction not in labels and prediction != "no_output":
            prediction = "other_trained_identity"
        confusion[(row["expected_speaker"], prediction)] += 1
    return {"n_expected": expected, "n_predicted": len(valid), "n_correct": correct,
            "top1": correct / len(valid) if valid else None,
            "coverage": len(valid) / expected if expected else None,
            "end_to_end_correct_rate": correct / expected if expected else None,
            "macro_f1": sum(f1s) / len(f1s) if expected else None,
            "confusion": confusion}


def pairing_window(parts, start, end):
    """Pair block-aligned samples with every frame actually encoded for those blocks."""
    import numpy as np
    wanted = range(start // BLOCK, end // BLOCK)
    if any(seq not in parts for seq in wanted):
        return None
    chosen = [parts[seq] for seq in wanted]
    return np.concatenate([p[0] for p in chosen]), np.concatenate([p[1] for p in chosen], axis=1)
