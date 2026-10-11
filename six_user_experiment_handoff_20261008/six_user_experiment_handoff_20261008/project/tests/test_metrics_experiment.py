import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = next(ROOT.glob("3*demo/speechtokenizer_now/speechtokenizer/*/run_metrics_experiment.py")).parent
sys.path.insert(0, str(DEMO))
from metrics_protocol import (BLOCK, USERS, expected_links, pairing_window, prediction_windows,
                              read_json, recognition_metrics, timeline_blocks, trial_jobs, write_json)
from summarize_metrics_experiment import (figures, milliseconds, normalize_job, response_rows,
                                         stream_summary, summarize)


def segment(label="p225", group="p225_001"):
    return {"speaker_id": label, "utterance_group": group, "samples": 48000, "wav_path": "fixture.wav"}


class MetricsExperimentTest(unittest.TestCase):
    def test_six_concurrent_senders_produce_thirty_links(self):
        job = trial_jobs("D", "concurrent")[0]
        self.assertEqual(len(job["users"]), 6)
        self.assertEqual(len(job["active_senders"]), 6)
        self.assertEqual(len(expected_links(job)), 30)
        self.assertTrue(all(a != b for a, b in expected_links(job)))

    def test_sequential_and_three_user_control(self):
        sequential = trial_jobs("D", "sequential")
        self.assertEqual(len(sequential), 6)
        self.assertTrue(all(len(j["users"]) == 6 and len(j["active_senders"]) == 1 for j in sequential))
        groups = trial_jobs("B", "concurrent")
        self.assertEqual([g["users"] for g in groups], [list("ABC"), list("DEF")])
        self.assertEqual(sum(len(expected_links(j)) for j in groups), 12)

    def test_missing_predictions_remain_in_denominator(self):
        rows = [{"expected_speaker": "p225", "predicted_speaker": "p225", "status": "predicted"},
                {"expected_speaker": "p226", "predicted_speaker": "p300", "status": "predicted"},
                {"expected_speaker": "p227", "predicted_speaker": "p227", "status": "late_prediction"}]
        m = recognition_metrics(rows)
        self.assertEqual(m["n_expected"], 3)
        self.assertEqual(m["n_predicted"], 2)
        self.assertAlmostEqual(m["top1"], .5)
        self.assertAlmostEqual(m["coverage"], 2 / 3)
        self.assertAlmostEqual(m["end_to_end_correct_rate"], 1 / 3)
        self.assertEqual(m["confusion"][("p226", "other_trained_identity")], 1)
        self.assertEqual(m["confusion"][("p227", "no_output")], 1)
        self.assertIsNone(recognition_metrics([])["top1"])

    def test_switch_changes_truth_within_same_routing_stream(self):
        segments = [segment(), segment(), segment("p226", "p226_001"), segment("p226", "p226_001")]
        windows = list(prediction_windows(segments, continuous=True))
        self.assertEqual(len(windows), 10)
        self.assertEqual(windows[0]["window_end_sample"], 48000)
        seven = next(w for w in windows if w["window_end_sample"] == 7 * 16000)
        self.assertTrue(seven["transition"])
        self.assertEqual(seven["expected_speaker"], "p226")
        nine = next(w for w in windows if w["window_end_sample"] == 9 * 16000)
        self.assertFalse(nine["transition"])

    def test_switch_plot_keeps_source_identities_without_prediction_outputs(self):
        from matplotlib.figure import Figure

        job = {"streams": {"A": [segment(label) for label in USERS.values()]}}
        output = {"job_id": "switch_plot", "scenario": "switch", "sender_id": "A",
                  "receiver_id": "B", "expected_speaker": "p225", "predicted_speaker": "p300",
                  "source_window_end_ns": 4_000_000_000, "window_end_sample": 48000,
                  "output_ns": 4_500_000_000, "score": .8}
        saved = {}

        def capture(figure, path, **kwargs):
            saved[Path(path).stem] = [t.get_text() for t in figure.axes[0].get_yticklabels()]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "jobs/switch_plot/job.json", job)
            with patch.object(Figure, "savefig", capture):
                figures(root, [], [], [], [], [output])
        self.assertEqual(saved["speaker_switch_timeline"], [*USERS.values(), "p300"])

    def test_summary_records_plot_failure_separately_from_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "experiment_plan.json", {"jobs": []})
            write_json(root / "run_manifest.json", {"status": "complete", "fixture": True})
            with patch("summarize_metrics_experiment.figures", side_effect=ValueError("plot failure")):
                with self.assertRaisesRegex(ValueError, "plot failure"):
                    summarize(root)
            status = read_json(root / "analysis_status.json")
            self.assertEqual(status["plot_status"], "failed")
            self.assertFalse(status["plots"])
            self.assertIn("plot failure", status["plot_error"])
            self.assertEqual(read_json(root / "run_manifest.json")["status"], "complete")

    def test_pairing_keeps_actual_frames_and_detects_gaps(self):
        import numpy as np
        parts = {i: (np.full(BLOCK, i, dtype="float32"), np.full((3, 26), i)) for i in range(6)}
        pcm, codes = pairing_window(parts, 0, 48000)
        self.assertEqual(len(pcm), 48000)
        self.assertEqual(codes.shape, (3, 156))
        self.assertEqual(codes[0, 26], 1)
        del parts[2]
        self.assertIsNone(pairing_window(parts, 0, 48000))

    def test_source_plan_creates_missing_block_and_prediction_rows(self):
        job = {"job_id": "D_fixture", "condition": "D", "scenario": "sequential", "repeat": 1,
               "users": list("ABCDEF"), "active_senders": ["A"], "transport": "tcp_loopback",
               "streams": {"A": [segment()]}, "recognition": True, "continuous": False, "playback": "off"}
        with tempfile.TemporaryDirectory() as temporary:
            blocks, outputs, audio, resources, _ = normalize_job(Path(temporary), job)
        self.assertEqual(len(blocks), 30)
        self.assertEqual(len(outputs), 5)
        self.assertTrue(all(p["status"] == "not_started" for p in outputs))
        self.assertTrue(all(b["capture_to_decode_ms"] is None for b in blocks))
        self.assertEqual(len(audio), 5)

    def test_latency_is_a_paired_measurement(self):
        self.assertEqual(milliseconds(700_000_000, 0), 700)
        self.assertIsNone(milliseconds(None, 0))

    def test_fixed_source_block_sequence(self):
        rows = list(timeline_blocks([segment(), segment()]))
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[6]["seq_id"], 6)
        self.assertEqual(rows[6]["source_start_sample"], 48000)
        self.assertEqual(rows[6]["segment_offset"], 0)

    def test_continuous_six_stream_plan_retains_all_positions(self):
        job = {**trial_jobs("D", "concurrent")[0], "job_id": "long_fixture", "repeat": 1,
               "streams": {s: [segment(USERS[s], USERS[s] + "_001") for _ in range(12)] for s in USERS},
               "continuous": True, "playback": "off"}
        with tempfile.TemporaryDirectory() as temporary:
            blocks, outputs, audio, _, _ = normalize_job(Path(temporary), job)
        self.assertEqual(len(blocks), 30 * 72)
        self.assertEqual(len(outputs), 30 * 34)
        self.assertEqual(len(audio), 30 * 12)

    def test_first_response_uses_whole_identity_dwell(self):
        job = {**trial_jobs("A", "stability")[0], "job_id": "response", "repeat": 1,
               "streams": {"A": [segment(), segment()]}, "continuous": True}
        output = {"job_id": "response", "sender_id": "A", "receiver_id": "A", "window_end_sample": 64000,
                  "expected_speaker": "p225", "correct": 1, "output_ns": 5_500_000_000}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "jobs/response/receipt.json", {"origin_ns": 1_000_000_000})
            rows = response_rows({"jobs": [job]}, [output], root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "correct_observed")
        self.assertEqual(rows[0]["first_correct_ms"], 4500)

    def test_switch_response_after_identity_left_remains_failure(self):
        job = {**trial_jobs("A", "switch")[0], "job_id": "response", "repeat": 1,
               "streams": {"A": [segment(), segment(), segment("p226"), segment("p226")]}, "continuous": True}
        output = {"job_id": "response", "sender_id": "A", "receiver_id": "A", "window_end_sample": 96000,
                  "expected_speaker": "p225", "correct": 1, "output_ns": 8_000_000_000}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "jobs/response/receipt.json", {"origin_ns": 1_000_000_000})
            rows = response_rows({"jobs": [job]}, [output], root)
        self.assertEqual(rows[0]["status"], "correct_after_identity_left")
        self.assertIsNone(rows[0]["first_correct_within_dwell_ms"])

    def test_interrupted_client_records_failed_exit_and_joins_workers(self):
        from metrics_client import MeasuredClient
        job = {"streams": {"A": [segment()]}, "continuous": False, "resource_interval_seconds": 1,
               "queue_blocks": 4, "playback": "off"}
        with tempfile.TemporaryDirectory() as temporary:
            client = MeasuredClient(job, "A", Path(temporary), fake_codec=True)
            with patch.object(client, "load"), patch.object(client, "connect"), patch.object(client, "start_playback"), patch.object(client, "gate", side_effect=KeyboardInterrupt):
                code = client.run()
            done = read_json(Path(temporary) / "A.done.json")
            self.assertEqual(code, 1)
            self.assertEqual(done["exit_code"], 1)
            self.assertIn("interrupted", done["errors"])
            self.assertFalse(client.resources_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
