"""Recompute all system metrics from source plans and append-only raw events."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from metrics_protocol import (BLOCK, SR, USERS, expected_links, prediction_windows, quantiles, sha256,
                              read_events, read_json, recognition_metrics, timeline_blocks, write_csv, write_json)


def milliseconds(end, start):
    return (end - start) / 1e6 if end is not None and start is not None else None


def prefix_stats(prefix, values):
    return {prefix + "_" + k: v for k, v in quantiles(values).items()}


def classifier_output_space(job):
    candidates = job.get("classifier_candidates") or ()
    return len(candidates) if candidates else 110


def normalize_job(root, job):
    directory = root / "jobs" / job["job_id"]
    events = read_events(directory / "events")
    receipt = read_json(directory / "receipt.json") if (directory / "receipt.json").is_file() else {"status": "not_started"}
    origin = receipt.get("origin_ns")
    common = {"job_id": job["job_id"], "condition": job["condition"], "scenario": job["scenario"],
              "repeat": job["repeat"], "deployment": job["transport"], "job_status": receipt["status"]}
    common["compute_backend"] = job.get("compute_backend", "local_cpu")
    common["classifier_output_space"] = classifier_output_space(job)
    indexed = defaultdict(dict)
    predictions = {}
    resources, event_rows = [], []
    for event in events:
        event_rows.append({**common, **event})
        if event.get("event") == "resource":
            phase = "startup" if origin is None or event["event_ns"] < origin else "replay"
            duration = max(sum(s["samples"] for s in parts) / SR for parts in job["streams"].values())
            if origin and event["event_ns"] > origin + duration * 1e9:
                phase = "drain"
            resources.append({**common, **event, "phase": phase})
        elif event.get("event") == "prediction":
            predictions[(event["sender_id"], event["receiver_id"], event["window_index"])] = event
        elif "sender_id" in event and "seq_id" in event:
            receiver = event.get("receiver_id", "")
            indexed[(event["sender_id"], receiver, event["seq_id"])][event["event"]] = event
    blocks, outputs, audio = [], [], []
    for sender, receiver in expected_links(job):
        link_blocks = {}
        segment_blocks = defaultdict(list)
        for info in timeline_blocks(job["streams"][sender]):
            seq = info["seq_id"]
            source = indexed.get((sender, "", seq), {})
            remote = indexed.get((sender, receiver, seq), {})
            sent = source.get("send", {})
            received = remote.get("receive", {})
            decoded = remote.get("decode", {})
            forward = remote.get("router_forward", {})
            callback = remote.get("playback_callback", {})
            if decoded:
                status = decoded["status"]
            elif "block_failure" in remote:
                status = remote["block_failure"]["status"]
            elif "router_queue_drop" in remote:
                status = "router_queue_drop"
            elif received:
                status = "received_not_decoded"
            elif sent:
                status = "not_received_by_session_deadline"
            else:
                status = "not_sent" if origin else "not_started"
            first_sample = origin + round(info["source_start_sample"] / SR * 1e9) if origin else None
            last_sample = origin + round(info["source_end_sample"] / SR * 1e9) if origin else None
            row = {**common, "sender_id": sender, "receiver_id": receiver, "stream_id": sender, **info,
                   "status": status, "source_start_ns": first_sample, "source_end_ns": last_sample,
                   "encode_start_ns": sent.get("encode_start_ns"), "encode_end_ns": sent.get("encode_end_ns"),
                   "send_start_ns": sent.get("send_start_ns"), "send_end_ns": sent.get("send_end_ns"),
                   "router_receive_ns": source.get("router_receive", {}).get("event_ns"),
                   "router_forward_ns": forward.get("event_ns"), "receive_ns": received.get("receive_ns"),
                   "decode_start_ns": decoded.get("decode_start_ns"), "decode_end_ns": decoded.get("decode_end_ns"),
                   "playback_callback_ns": callback.get("callback_ns"), "device_dac_time": callback.get("device_dac_time"),
                   "payload_bytes": received.get("payload_bytes"), "application_bytes": received.get("application_bytes"),
                   "header_bytes": received.get("application_bytes", 0) - received.get("payload_bytes", 0) if received else None,
                   "send_payload_bytes": sent.get("payload_bytes"), "send_application_bytes": sent.get("application_bytes"),
                   "token_frames": decoded.get("token_frames"), "received_samples": decoded.get("received_samples"),
                   "codec_lock_wait_ms": decoded.get("codec_lock_wait_ms"),
                   "failure_reason": remote.get("block_failure", {}).get("reason", ""),
                   "playback_status": "callback_completed" if callback else "not_measured" if job["playback"] == "off" else "no_callback",
                   "encode_ms": milliseconds(sent.get("encode_end_ns"), sent.get("encode_start_ns")),
                   "transport_ms": milliseconds(received.get("receive_ns"), sent.get("send_start_ns")) if job["transport"] != "local" else None,
                   "queue_wait_ms": milliseconds(decoded.get("decode_start_ns"), received.get("receive_ns")),
                   "decode_ms": milliseconds(decoded.get("decode_end_ns"), decoded.get("decode_start_ns")),
                   "capture_to_decode_ms": milliseconds(decoded.get("decode_end_ns"), first_sample),
                   "block_tail_to_decode_ms": milliseconds(decoded.get("decode_end_ns"), last_sample),
                   "capture_to_callback_ms": milliseconds(callback.get("callback_ns"), first_sample),
                   "playback_wait_ms": milliseconds(callback.get("callback_ns"), decoded.get("decode_end_ns")),
                   "acoustic_output_latency_ms": None}
            blocks.append(row)
            link_blocks[seq] = row
            segment_blocks[info["segment_index"]].append(row)
        if job["recognition"]:
            for window in prediction_windows(job["streams"][sender], job["continuous"]):
                prediction = predictions.get((sender, receiver, window["window_index"]), {})
                status = prediction.get("status", "no_output" if origin else "not_started")
                window_blocks = [link_blocks[seq] for seq in range(window["window_start_sample"] // BLOCK,
                                                                  window["window_end_sample"] // BLOCK)]
                reason = prediction.get("reason", "")
                if not prediction:
                    reason = ";".join(sorted({r["status"] for r in window_blocks if r["status"] != "decoded"})) or "no prediction record"
                outputs.append({**common, "sender_id": sender, "receiver_id": receiver, "stream_id": sender,
                                **window, "status": status, "reason": reason,
                                "predicted_speaker": prediction.get("predicted_speaker", ""),
                                "score": prediction.get("score"), "margin": prediction.get("margin"),
                                "verified": prediction.get("verified"), "output_ns": prediction.get("output_ns"),
                                "inference_ms": milliseconds(prediction.get("output_ns"), prediction.get("inference_start_ns")),
                                "source_window_end_ns": origin + round(window["window_end_sample"] / SR * 1e9) if origin else None,
                                "paired_samples": prediction.get("paired_samples"), "paired_token_frames": prediction.get("paired_token_frames"),
                                "correct": int(status == "predicted" and prediction.get("predicted_speaker") == window["expected_speaker"]),
                                "all_window_blocks_decoded": all(r["status"] in ("decoded", "late_decoded") for r in window_blocks)})
        for index, segment in enumerate(job["streams"][sender]):
            rows = segment_blocks[index]
            reconstructed = directory / "audio" / receiver / sender / (str(index).zfill(5) + ".wav")
            audio.append({**common, "sender_id": sender, "receiver_id": receiver, "segment_index": index,
                          "utterance_group": segment["utterance_group"], "speaker_id": segment["speaker_id"],
                          "reference_path": segment["wav_path"], "reconstructed_path": str(reconstructed),
                          "missing_blocks": sum(r["status"] not in ("decoded", "late_decoded") for r in rows),
                          "audio_status": "recorded" if reconstructed.is_file() else "missing_zero_fill",
                          "alignment": "block_positions; trim_or_end_pad_8000; preserve_gaps"})
    return blocks, outputs, audio, resources, event_rows


def stream_summary(blocks, predictions, audio):
    result = []
    grouped = defaultdict(list)
    for b in blocks:
        grouped[(b["job_id"], b["sender_id"], b["receiver_id"])].append(b)
    for (job, sender, receiver), rows in grouped.items():
        outputs = [p for p in predictions if p["job_id"] == job and p["sender_id"] == sender and p["receiver_id"] == receiver and not p["transition"]]
        scores = [a for a in audio if a["job_id"] == job and a["sender_id"] == sender and a["receiver_id"] == receiver]
        recognition = recognition_metrics(outputs, sorted({p["expected_speaker"] for p in outputs}))
        recognition.pop("confusion")
        duration = len(rows) * .5
        sent = [r for r in rows if r["send_end_ns"] is not None]
        received = [r for r in rows if r["receive_ns"] is not None]
        decoded = [r for r in rows if r["decode_end_ns"] is not None]
        status_counts = Counter(r["status"] for r in rows)
        row = {k: rows[0][k] for k in ("job_id", "condition", "scenario", "repeat", "sender_id", "receiver_id", "deployment", "job_status")}
        row.update(expected_blocks=len(rows), sent_blocks=len(sent), received_blocks=len(received), decoded_blocks=len(decoded),
                   missing_blocks=len(rows) - len(received), decode_failures=len(received) - len(decoded),
                   missing_rate=(len(rows) - len(received)) / len(rows), deadline_expired=status_counts["deadline_expired"],
                   receiver_queue_drops=status_counts["receiver_queue_drop"], router_queue_drops=status_counts["router_queue_drop"],
                   late_decoded=status_counts["late_decoded"], source_audio_seconds=duration,
                   payload_bytes=sum(r["payload_bytes"] or 0 for r in rows),
                   application_bytes=sum(r["application_bytes"] or 0 for r in rows), **recognition)
        row["deadline_failure_blocks"] = sum(r["status"] in ("deadline_expired", "late_decoded", "not_received_by_session_deadline") for r in rows)
        row["deadline_failure_rate"] = row["deadline_failure_blocks"] / len(rows)
        row["payload_kbps_per_source_audio"] = row["payload_bytes"] * 8 / duration / 1000 if row["deployment"] != "local" else None
        row["application_kbps_per_source_audio"] = row["application_bytes"] * 8 / duration / 1000 if row["deployment"] != "local" else None
        for metric in ("encode_ms", "decode_ms", "queue_wait_ms", "transport_ms", "capture_to_decode_ms", "capture_to_callback_ms"):
            row.update(prefix_stats(metric, [r[metric] for r in rows]))
        row.update(prefix_stats("inference_ms", [p["inference_ms"] for p in outputs]))
        for metric in ("stoi", "pesq_wb", "mel_l1"):
            row.update(prefix_stats(metric, [s.get(metric) for s in scores]))
            row[metric + "_failures"] = sum(s.get(metric) is None for s in scores)
        result.append(row)
    return result


def response_rows(plan, outputs, root):
    rows = []
    for job in plan["jobs"]:
        receipt_path = root / "jobs" / job["job_id"] / "receipt.json"
        origin = read_json(receipt_path).get("origin_ns") if receipt_path.is_file() else None
        if not job["recognition"]:
            continue
        for sender, receiver in expected_links(job):
            start = 0
            previous = None
            for index, segment in enumerate(job["streams"][sender]):
                samples = segment["samples"]
                changed = previous is not None and previous != segment["speaker_id"]
                if not job["continuous"] or previous is None or changed:
                    dwell_end = start + samples
                    if job["continuous"]:
                        for following in job["streams"][sender][index + 1:]:
                            if following["speaker_id"] != segment["speaker_id"]:
                                break
                            dwell_end += following["samples"]
                    candidates = [p for p in outputs if p["job_id"] == job["job_id"] and p["sender_id"] == sender
                                  and p["receiver_id"] == receiver and start < p["window_end_sample"] <= dwell_end
                                  and p["expected_speaker"] == segment["speaker_id"] and p["correct"]]
                    first = min((p["output_ns"] for p in candidates if p["output_ns"] is not None), default=None)
                    begin = origin + round(start / SR * 1e9) if origin else None
                    end_ns = origin + round(dwell_end / SR * 1e9) if origin else None
                    after_dwell = bool(job["continuous"] and first is not None and end_ns is not None and first > end_ns)
                    rows.append({"job_id": job["job_id"], "condition": job["condition"], "scenario": job["scenario"],
                                 "repeat": job["repeat"], "sender_id": sender, "receiver_id": receiver,
                                 "segment_index": index, "speaker_id": segment["speaker_id"],
                                 "kind": "switch" if changed and job["continuous"] else "initial",
                                 "source_start_ns": begin, "first_correct_output_ns": first,
                                 "first_correct_ms": milliseconds(first, begin),
                                 "identity_dwell_end_ns": end_ns, "output_after_identity_left": after_dwell,
                                 "first_correct_within_dwell_ms": milliseconds(first, begin) if first and not after_dwell else None,
                                 "status": "correct_after_identity_left" if after_dwell else "correct_observed" if first else "no_correct_output"})
                previous = segment["speaker_id"]
                start += samples
    return rows


def run_summaries(plan, blocks, outputs, scores, resources, events, root):
    grouped = defaultdict(list)
    for job in plan["jobs"]:
        grouped[(job["condition"], job["scenario"], job["repeat"])].append(job)
    summaries, confusion, per_speaker, traffic = [], [], [], []
    for (condition, scenario, repeat), jobs in grouped.items():
        ids = {j["job_id"] for j in jobs}
        bs = [b for b in blocks if b["job_id"] in ids]
        predictions = [p for p in outputs if p["job_id"] in ids and not p["transition"]]
        metric = recognition_metrics(predictions)
        confusion_counts = metric.pop("confusion")
        for label in USERS.values():
            for column in [*USERS.values(), "other_trained_identity", "no_output"]:
                confusion.append({"condition": condition, "scenario": scenario, "repeat": repeat,
                                  "true_speaker": label, "predicted_column": column,
                                  "count": confusion_counts[(label, column)]})
            personal = recognition_metrics([p for p in predictions if p["expected_speaker"] == label], [label])
            personal.pop("confusion")
            personal.pop("macro_f1")
            tp = sum(p["correct"] for p in predictions if p["expected_speaker"] == label)
            fp = sum(p["status"] == "predicted" and p["predicted_speaker"] == label and p["expected_speaker"] != label for p in predictions)
            fn = sum(p["expected_speaker"] == label for p in predictions) - tp
            personal.update(class_precision=tp / (tp + fp) if tp + fp else None,
                            class_recall=tp / (tp + fn) if tp + fn else None,
                            class_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
            per_speaker.append({"condition": condition, "scenario": scenario, "repeat": repeat, "speaker_id": label, **personal})
        rs = [r for r in resources if r["job_id"] in ids]
        ev = [r for r in events if r["job_id"] in ids]
        failed_jobs = sum(read_json(root / "jobs" / j["job_id"] / "receipt.json")["status"] != "complete"
                          if (root / "jobs" / j["job_id"] / "receipt.json").exists() else True for j in jobs)
        summary = {"condition": condition, "scenario": scenario, "repeat": repeat, "jobs_planned": len(jobs),
                   "failed_or_unstarted_jobs": failed_jobs, "connected_users_planned": max(len(j["users"]) for j in jobs),
                   "active_senders_planned": max(len(j["active_senders"]) for j in jobs),
                   "recognition_enabled": condition != "C", **metric,
                   "transition_windows": sum(p["transition"] for p in outputs if p["job_id"] in ids),
                   "expected_blocks": len(bs), "received_blocks": sum(b["receive_ns"] is not None for b in bs),
                   "decoded_blocks": sum(b["decode_end_ns"] is not None for b in bs),
                   "deadline_expired_blocks": sum(b["status"] == "deadline_expired" for b in bs),
                   "late_decoded_blocks": sum(b["status"] == "late_decoded" for b in bs),
                   "receiver_queue_drops": sum(b["status"] == "receiver_queue_drop" for b in bs),
                   "router_queue_drops": sum(b["status"] == "router_queue_drop" for b in bs),
                   "measured_online_users_max": max((r.get("online_users", 0) for r in ev), default=0) if condition != "A" else None,
                   "process_rss_max_bytes": max((r.get("rss_bytes", 0) for r in rs), default=None),
                   "system_cpu_percent_max": max((r["system_cpu_percent"] for r in rs if "system_cpu_percent" in r), default=None),
                   "system_used_memory_max_bytes": max((r["system_used_memory_bytes"] for r in rs if "system_used_memory_bytes" in r), default=None),
                   "gpu_utilization_max_percent": max((r["gpu_percent"] for r in rs if r.get("gpu_percent") not in (None, "")), default=None),
                   "gpu_memory_max_bytes": max((r["vram_bytes"] for r in rs if r.get("vram_bytes") not in (None, "")), default=None),
                   "gpu_allocated_max_bytes": max((r["gpu_allocated_bytes"] for r in rs if r.get("gpu_allocated_bytes") not in (None, "")), default=None),
                   "gpu_reserved_max_bytes": max((r["gpu_reserved_bytes"] for r in rs if r.get("gpu_reserved_bytes") not in (None, "")), default=None),
                   "decode_queue_blocks_max": max((r.get("decode_queue_blocks", 0) for r in rs), default=None),
                   "router_queue_blocks_max": max((r.get("router_queue_blocks", 0) for r in rs), default=None),
                   "timeout_events": sum(r.get("event") == "timeout" for r in ev),
                   "disconnect_events": sum(r.get("event") == "disconnect" for r in ev),
                   "playback_underruns": sum(r.get("playback_underruns", 0) for r in ev if r.get("event") == "client_exit"),
                   "acoustic_output_latency_ms": None, "traffic_boundary": "application_layer",
                   "classifier_output_space": classifier_output_space(jobs[0]),
                   "unique_source_utterances": len({p["utterance_group"] for p in predictions})}
        summary["missing_rate"] = (len(bs) - summary["received_blocks"]) / len(bs) if bs else None
        summary["deadline_failure_blocks"] = sum(b["status"] in ("deadline_expired", "late_decoded", "not_received_by_session_deadline") for b in bs)
        summary["deadline_failure_rate"] = summary["deadline_failure_blocks"] / len(bs) if bs else None
        first_last = defaultdict(list)
        for event in ev:
            if event.get("event") == "send":
                first_last[(event["job_id"], event["user_id"])].append(event["send_start_ns"])
        summary["observed_source_stream_instances"] = len(first_last)
        overlaps = []
        for job in jobs:
            changes = []
            for (jid, _), times in first_last.items():
                if jid == job["job_id"]:
                    changes.extend([(min(times), 1), (max(times) + 1, -1)])
            active, peak = 0, 0
            for _, change in sorted(changes):
                active += change
                peak = max(peak, active)
            overlaps.append(peak)
        summary["observed_source_send_span_overlap_max"] = max(overlaps, default=0)
        receipts = [read_json(root / "jobs" / j["job_id"] / "receipt.json") for j in jobs
                    if (root / "jobs" / j["job_id"] / "receipt.json").is_file()]
        summary["job_elapsed_seconds_sum"] = sum(r.get("elapsed_seconds", 0) for r in receipts) if receipts else None
        summary["source_replay_seconds_sum"] = sum(r.get("source_duration_seconds", 0) for r in receipts) if receipts else None
        summary["client_reconnect_attempts"] = 0
        summary["reconnect_policy"] = "disabled_disconnect_recorded"
        for direction, event_name in (("router_ingress", "router_receive"), ("router_egress", "router_forward"),
                                      ("client_tx", "send"), ("client_rx", "receive")):
            chosen = [e for e in ev if e.get("event") == event_name]
            for field in ("payload_bytes", "application_bytes"):
                summary[direction + "_" + field] = sum(e.get(field, 0) for e in chosen) if condition != "A" else None
        for key in ("capture_to_decode_ms", "capture_to_callback_ms", "encode_ms", "decode_ms", "transport_ms"):
            summary.update(prefix_stats(key, [b[key] for b in bs]))
        summary.update(prefix_stats("inference_ms", [p["inference_ms"] for p in predictions]))
        # Equal weight per source utterance, after averaging receiver copies and replay occurrences.
        quality = [q for q in scores if q["job_id"] in ids]
        for key in ("stoi", "pesq_wb", "mel_l1"):
            source_groups = defaultdict(list)
            for q in quality:
                if q.get(key) is not None:
                    source_groups[q["utterance_group"]].append(q[key])
            summary[key + "_source_mean"] = statistics.mean(statistics.mean(v) for v in source_groups.values()) if source_groups else None
            summary[key + "_observations"] = sum(q.get(key) is not None for q in quality)
            summary[key + "_failures"] = sum(q.get(key) is None for q in quality)
        summaries.append(summary)
        for job in jobs:
            job_events = [r for r in ev if r["job_id"] == job["job_id"]]
            source_seconds = {s: sum(p["samples"] for p in parts) / SR for s, parts in job["streams"].items()}
            for user in [*job["users"], *(["router"] if job["transport"] != "local" else [])]:
                tx = [r for r in job_events if r.get("event") == ("router_forward" if user == "router" else "send")
                      and (user == "router" or r.get("user_id") == user)]
                rx = [r for r in job_events if r.get("event") == ("router_receive" if user == "router" else "receive")
                      and (user == "router" or r.get("user_id") == user)]
                traffic.append({"job_id": job["job_id"], "condition": condition, "scenario": scenario, "repeat": repeat,
                                "user_id": user, "tx_payload_bytes": sum(r.get("payload_bytes", 0) for r in tx),
                                "rx_payload_bytes": sum(r.get("payload_bytes", 0) for r in rx),
                                "tx_application_bytes": sum(r.get("application_bytes", 0) for r in tx),
                                "rx_application_bytes": sum(r.get("application_bytes", 0) for r in rx),
                                "sent_source_audio_seconds": source_seconds.get(user),
                                "measurement_boundary": "codes_v1_application_messages",
                                "transport_link_bytes": None,
                                "status": "observed" if job_events else "not_started"})
    return summaries, confusion, per_speaker, traffic


def figures(root, confusion, summaries, blocks, resources, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(directory / (name + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)
    conditions = sorted({(r["condition"], r["scenario"]) for r in summaries})
    for condition, scenario in conditions:
        if condition == "C":
            continue
        counts = [c for c in confusion if (c["condition"], c["scenario"]) == (condition, scenario)]
        matrix = np.zeros((6, 8), dtype=int)
        columns = [*USERS.values(), "other_trained_identity", "no_output"]
        for row in counts:
            matrix[list(USERS.values()).index(row["true_speaker"]), columns.index(row["predicted_column"])] += row["count"]
        fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(8), [*USERS.values(), "Other", "No output"], rotation=35, ha="right")
        ax.set_yticks(range(6), USERS.values())
        ax.set(xlabel="Prediction", ylabel="True identity", title=f"{condition} / {scenario} - dependent window records")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Records")
        save(fig, f"confusion_{condition}_{scenario}")
    for condition, scenario in conditions:
        matrix = np.zeros((6, 6), dtype=int)
        for row in blocks:
            if (row["condition"], row["scenario"]) == (condition, scenario) and row["receive_ns"] is not None:
                matrix[list(USERS).index(row["sender_id"]), list(USERS).index(row["receiver_id"])] += 1
        fig, ax = plt.subplots(figsize=(5, 4.5), layout="constrained")
        ax.imshow(matrix, cmap="Greens")
        ax.set_xticks(range(6), USERS)
        ax.set_yticks(range(6), USERS)
        ax.set(xlabel="Receiver", ylabel="Sender", title=f"{condition}/{scenario} block coverage")
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(j, i, str(count), ha="center", va="center", fontsize=8)
        save(fig, f"coverage_{condition}_{scenario}")
    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    any_latency = False
    for condition, scenario in conditions:
        values = sorted(b["capture_to_decode_ms"] for b in blocks if (b["condition"], b["scenario"]) == (condition, scenario) and b["capture_to_decode_ms"] is not None)
        if values:
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=f"{condition}/{scenario}")
            any_latency = True
    ax.set(xlabel="WAV block first sample to decoder completion (ms)", ylabel="Observed CDF")
    if any_latency:
        ax.legend(fontsize=7)
    save(fig, "decode_latency_cdf")
    write_json(directory / "figure_sources.json", {"confusion": "confusion_matrix.csv", "latency": "blocks.csv",
               "resource": "resources.csv", "response": "predictions.csv", "purpose": "system measurement diagnostics"})
    # A representative job keeps the timeline legible; all raw runs remain in CSV.
    for metric, label in (("rss_bytes", "Process RSS (bytes)"), ("decode_queue_blocks", "Decode queue (blocks)")):
        selected = next((r["job_id"] for r in resources if r.get(metric) is not None), None)
        if selected:
            fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
            rs = [r for r in resources if r["job_id"] == selected and r.get(metric) is not None]
            zero = min(r["event_ns"] for r in rs)
            for user in sorted({r["user_id"] for r in rs}):
                series = sorted((r for r in rs if r["user_id"] == user), key=lambda r: r["event_ns"])
                ax.plot([(r["event_ns"] - zero) / 1e9 for r in series], [r[metric] for r in series], label=user)
            ax.set(xlabel="Elapsed sampling time (s)", ylabel=label, title=selected)
            ax.legend(fontsize=7)
            save(fig, metric + "_timeline")
    switches = [p for p in predictions if p["scenario"] == "switch" and p.get("output_ns")]
    if switches:
        key = (switches[0]["job_id"], switches[0]["sender_id"], switches[0]["receiver_id"])
        rows = [p for p in switches if (p["job_id"], p["sender_id"], p["receiver_id"]) == key]
        job = read_json(root / "jobs" / key[0] / "job.json")
        parts = job["streams"][key[1]]
        # Missing prediction outputs must not remove identities from the source timeline.
        labels = sorted({part["speaker_id"] for part in parts}
                        | {p["expected_speaker"] for p in rows} | {p["predicted_speaker"] for p in rows})
        fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * .25)), layout="constrained")
        source_times, source_labels, cursor = [], [], 0
        for part in parts:
            source_times.append(cursor / SR)
            source_labels.append(labels.index(part["speaker_id"]))
            cursor += part["samples"]
        source_times.append(cursor / SR)
        source_labels.append(source_labels[-1])
        ax.step(source_times, source_labels, where="post", color="black", label="Actual source identity")
        zero = rows[0]["source_window_end_ns"] - round(rows[0]["window_end_sample"] / SR * 1e9)
        points = ax.scatter([(p["output_ns"] - zero) / 1e9 for p in rows], [labels.index(p["predicted_speaker"]) for p in rows],
                            c=[p["score"] for p in rows], vmin=0, vmax=1, cmap="viridis", label="Prediction output", s=20)
        fig.colorbar(points, ax=ax, label="Calibrated confidence")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set(xlabel="Source replay time (s)", ylabel="Identity", title=" / ".join(key))
        ax.legend(fontsize=7)
        save(fig, "speaker_switch_timeline")


def summarize(root, quality=True, plot=True):
    root = Path(root).resolve()
    plan = read_json(root / "experiment_plan.json")
    manifest = read_json(root / "run_manifest.json")
    blocks, predictions, audio, resources, events = [], [], [], [], []
    for job in plan["jobs"]:
        data = normalize_job(root, job)
        for target, items in zip((blocks, predictions, audio, resources, events), data):
            target.extend(items)
    scorer = None
    score_failures = []
    if quality and not manifest.get("fixture"):
        try:
            from quality_metrics import AudioScorer
            scorer = AudioScorer(plan["policies"]["mel_config"], manifest["quality_python"], root / "logs" / "pesq_worker.log")
            cache = {}
            for row in audio:
                if row["job_status"] == "not_started":
                    row.update(stoi=None, pesq_wb=None, mel_l1=None, evaluation_status="not_started")
                    continue
                try:
                    reconstructed = Path(row["reconstructed_path"])
                    key = (sha256(row["reference_path"]), sha256(reconstructed) if reconstructed.is_file() else "missing_zero_fill")
                    reused = key in cache
                    result = cache[key] if reused else scorer.score(row["reference_path"], row["reconstructed_path"])
                    if all(result.get(status) == "measured" for status in ("stoi_status", "pesq_status", "mel_l1_status")):
                        cache[key] = result
                    row.update(result)
                    row["reference_sha256"], row["reconstructed_sha256"] = key
                    row["exact_audio_score_cache_reused"] = reused
                    row["evaluation_status"] = "measured_with_zero_fill" if row["missing_blocks"] else "measured"
                except Exception as error:
                    row.update(stoi=None, pesq_wb=None, mel_l1=None, evaluation_status="failed", evaluation_reason=repr(error))
                    score_failures.append(row["job_id"] + ":" + row["utterance_group"] + ":" + repr(error))
        except Exception as error:
            score_failures.append(repr(error))
        finally:
            if scorer:
                scorer.close()
    for row in audio:
        if "evaluation_status" not in row:
            row.update(stoi=None, pesq_wb=None, mel_l1=None,
                       evaluation_status="fixture_not_scored" if manifest.get("fixture") else "not_measured",
                       evaluation_reason=";".join(score_failures) or "quality scoring disabled")
    streams = stream_summary(blocks, predictions, audio)
    summaries, confusion, personal, traffic = run_summaries(plan, blocks, predictions, audio, resources, events, root)
    if manifest.get("fixture"):
        for row in [*summaries, *streams, *personal]:
            for key in ("top1", "macro_f1", "coverage", "end_to_end_correct_rate", "class_precision", "class_recall", "class_f1"):
                if key in row:
                    row[key] = None
            row["recognition_status"] = "fixture_not_evaluated"
    responses = response_rows(plan, predictions, root)
    for summary in summaries:
        rs = [r for r in responses if (r["condition"], r["scenario"], r["repeat"]) == (summary["condition"], summary["scenario"], summary["repeat"])]
        for kind in ("initial", "switch"):
            selected = [r for r in rs if r["kind"] == kind]
            summary.update(prefix_stats(kind + "_correct_ms", [r["first_correct_within_dwell_ms"] for r in selected]))
            summary[kind + "_no_correct_output"] = sum(r["status"] == "no_correct_output" for r in selected)
            summary[kind + "_response_attempts"] = len(selected)
            summary[kind + "_response_failures"] = sum(r["status"] != "correct_observed" for r in selected)
            summary[kind + "_correct_after_identity_left"] = sum(r["status"] == "correct_after_identity_left" for r in selected)
        grouped = defaultdict(list)
        for p in predictions:
            if (p["condition"], p["scenario"], p["repeat"]) == (summary["condition"], summary["scenario"], summary["repeat"]) and p["output_ns"]:
                grouped[(p["job_id"], p["sender_id"], p["receiver_id"])].append(p["output_ns"])
        intervals = [(b - a) / 1e6 for v in grouped.values() for a, b in zip(sorted(v), sorted(v)[1:])]
        summary.update(prefix_stats("actual_output_interval_ms", intervals))
        grouped_resources = defaultdict(list)
        for row in resources:
            if (row["condition"], row["scenario"], row["repeat"]) == (summary["condition"], summary["scenario"], summary["repeat"]) and row.get("phase") == "replay" and "decode_queue_blocks" in row:
                grouped_resources[(row["job_id"], row["user_id"])].append(row)
        slopes = []
        for series in grouped_resources.values():
            zero = min(r["event_ns"] for r in series)
            xs = [(r["event_ns"] - zero) / 1e9 for r in series]
            ys = [r["decode_queue_blocks"] for r in series]
            if len(xs) > 1:
                xbar, ybar = statistics.mean(xs), statistics.mean(ys)
                denom = sum((x - xbar) ** 2 for x in xs)
                if denom:
                    slopes.append(sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom)
        summary["decode_queue_replay_slope_max_blocks_per_second"] = max(slopes, default=None)
    files = {"blocks.csv": blocks, "predictions.csv": predictions, "audio_metrics.csv": audio,
             "resources.csv": resources, "events.csv": events, "per_stream_metrics.csv": streams,
             "summary.csv": summaries, "confusion_matrix.csv": confusion, "per_speaker_metrics.csv": personal,
             "traffic.csv": traffic, "identity_responses.csv": responses}
    for name, rows in files.items():
        write_csv(root / name, rows, fields=None if rows else ["job_id", "status"])
    grouped = defaultdict(list)
    for row in summaries:
        grouped[(row["condition"], row["scenario"])].append(row)
    repeat_summary = []
    for (condition, scenario), rows in grouped.items():
        for metric in ("top1", "macro_f1", "coverage", "end_to_end_correct_rate", "capture_to_decode_ms_p95", "inference_ms_p95", "stoi_source_mean", "pesq_wb_source_mean", "mel_l1_source_mean"):
            vals = [r[metric] for r in rows if r.get(metric) is not None]
            repeat_summary.append({"condition": condition, "scenario": scenario, "metric": metric,
                                   "runs_planned": len(rows), "runs_observed": len(vals),
                                   "system_run_mean": statistics.mean(vals) if vals else None,
                                   "system_run_sd": statistics.stdev(vals) if len(vals) > 1 else None,
                                   "aggregation": "per_run_then_mean; system_replay_repeats_not_training_seeds"})
    write_csv(root / "repeat_summary.csv", repeat_summary)
    analysis_status = {"quality_failures": score_failures, "fixture": manifest.get("fixture", False),
               "analysis_source_hashes": {name: sha256(Path(__file__).parent / name) for name in
                                          ("summarize_metrics_experiment.py", "metrics_protocol.py", "quality_metrics.py")},
               "expected_block_rows": len(blocks), "expected_prediction_rows": len(predictions), "plots": False,
               "plot_status": "running" if plot else "skipped",
               "steady_prediction_rows": sum(not p["transition"] for p in predictions),
               "steady_no_output": sum(p["status"] != "predicted" and not p["transition"] for p in predictions)}
    write_json(root / "analysis_status.json", analysis_status)
    if plot:
        try:
            figures(root, confusion, summaries, blocks, resources, predictions)
        except Exception as error:
            analysis_status.update(plot_status="failed", plot_error=repr(error))
            write_json(root / "analysis_status.json", analysis_status)
            raise
        analysis_status.update(plots=True, plot_status="complete")
        write_json(root / "analysis_status.json", analysis_status)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summarize(args.run_dir, not args.no_quality, not args.no_plots)
