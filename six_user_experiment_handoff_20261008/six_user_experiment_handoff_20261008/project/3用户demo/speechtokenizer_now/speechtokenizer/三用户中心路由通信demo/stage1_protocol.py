"""Small, dependency-free bookkeeping for the six-client Stage 1 replay."""

from __future__ import annotations

import json
import threading
from pathlib import Path


USERS = {"A": "p225", "B": "p226", "C": "p227", "D": "p228", "E": "p229", "F": "p230"}
CHUNKS_PER_TRIAL = 6


def load_stage1_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("users") != USERS or config.get("sender") not in USERS:
        raise ValueError("Stage 1 requires the fixed A-F to p225-p230 mapping")
    if config.get("chunks_per_trial") != CHUNKS_PER_TRIAL:
        raise ValueError("Stage 1 requires six 0.5-second packets per utterance")
    trials = config.get("trials")
    if not isinstance(trials, list) or len(trials) not in (1, 15):
        raise ValueError("Stage 1 requires one validation or 15 test utterances")
    if len({row.get("utterance_group") for row in trials}) != len(trials):
        raise ValueError("Stage 1 utterance groups must be unique")
    for row in trials:
        if not row.get("utterance_group") or not Path(row.get("wav_path", "")).is_file():
            raise ValueError("Stage 1 trial needs an existing WAV and utterance group")
    return config


class TrialLedger:
    """One receiver's results for one sending user; packet seq_id is the trial key."""

    def __init__(self, config, receiver):
        self.config = config
        self.receiver = receiver
        self.sender = config["sender"]
        self.expected_label = config["users"][self.sender]
        self.trials = [dict(received=set(), decoded=set(), prediction=None, decode_ms=[], errors=[]) for _ in config["trials"]]
        self.lock = threading.Lock()

    def trial_index(self, seq_id):
        seq_id = int(seq_id)
        if seq_id < 0 or seq_id >= len(self.trials) * CHUNKS_PER_TRIAL:
            raise ValueError(f"Stage 1 sequence out of range: {seq_id}")
        return seq_id // CHUNKS_PER_TRIAL

    def record_received(self, seq_id):
        index = self.trial_index(seq_id)
        with self.lock:
            self.trials[index]["received"].add(int(seq_id))
        return index

    def record_decoded(self, seq_id, prediction, decode_ms):
        index = self.trial_index(seq_id)
        with self.lock:
            trial = self.trials[index]
            trial["decoded"].add(int(seq_id))
            trial["decode_ms"].append(float(decode_ms))
            if prediction is not None and trial["prediction"] is None:
                trial["prediction"] = prediction

    def record_error(self, seq_id, error):
        index = self.trial_index(seq_id)
        with self.lock:
            self.trials[index]["errors"].append(str(error))

    def complete(self):
        with self.lock:
            return all(len(trial["decoded"]) == CHUNKS_PER_TRIAL for trial in self.trials)

    def rows(self):
        with self.lock:
            snapshot = [dict(received=set(t["received"]), decoded=set(t["decoded"]), prediction=t["prediction"], decode_ms=list(t["decode_ms"]), errors=list(t["errors"])) for t in self.trials]
        rows = []
        for index, (trial, item) in enumerate(zip(snapshot, self.config["trials"])):
            prediction = trial["prediction"]
            decoded = len(trial["decoded"])
            missing = CHUNKS_PER_TRIAL - decoded
            predicted_label = prediction.predicted_speaker if prediction is not None else ""
            correct = bool(prediction is not None and predicted_label == self.expected_label)
            values = sorted(trial["decode_ms"])
            p95 = values[min(len(values) - 1, int(round((len(values) - 1) * 0.95)))] if values else None
            rows.append({
                "receiver_id": self.receiver,
                "sender_id": self.sender,
                "expected_speaker_label": self.expected_label,
                "trial_index": index,
                "utterance_group": item["utterance_group"],
                "received_packets": len(trial["received"]),
                "decoded_packets": decoded,
                "missing_decoded_packets": missing,
                "predicted_speaker": predicted_label,
                "score": None if prediction is None else prediction.score,
                "margin": None if prediction is None else prediction.margin,
                "verified": None if prediction is None else int(prediction.verified),
                "evaluated": int(prediction is not None),
                "correct": int(correct),
                "end_to_end_correct": int(correct and missing == 0),
                "decode_ms_p95": p95,
                "errors": " | ".join(trial["errors"]),
            })
        return rows
