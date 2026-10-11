import csv
import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "3用户demo" / "speechtokenizer_now" / "speechtokenizer" / "三用户中心路由通信demo"
sys.path.insert(0, str(DEMO))

from stage1_protocol import CHUNKS_PER_TRIAL, TrialLedger, USERS, load_stage1_config
from run_stage1_six_user import aggregate, prepare_wav


class Stage1ProtocolTest(unittest.TestCase):
    def make_config(self, root, count=2):
        trials = []
        for index in range(count):
            wav = root / f"trial{index}.wav"
            wav.write_bytes(b"RIFF")
            trials.append({"utterance_group": f"p225_{index:03d}", "wav_path": str(wav)})
        return {"users": USERS, "sender": "A", "chunks_per_trial": CHUNKS_PER_TRIAL, "trials": trials}

    def test_local_truth_and_trial_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.make_config(Path(temp))
            ledger = TrialLedger(config, "B")
            prediction = SimpleNamespace(predicted_speaker="p225", score=0.9, margin=0.8, verified=True)
            for seq in range(6):
                ledger.record_received(seq)
                ledger.record_decoded(seq, prediction if seq == 5 else None, 10.0)
            # The second utterance misses one packet; its prediction is never
            # reused from the first utterance.
            for seq in range(6, 11):
                ledger.record_received(seq)
                ledger.record_decoded(seq, None, 12.0)
            rows = ledger.rows()
            self.assertEqual(rows[0]["expected_speaker_label"], "p225")
            self.assertEqual(rows[0]["end_to_end_correct"], 1)
            self.assertEqual(rows[1]["predicted_speaker"], "")
            self.assertEqual(rows[1]["missing_decoded_packets"], 1)
            self.assertFalse(ledger.complete())
            summary = aggregate(rows)[0]
            self.assertEqual(summary["utterances"], 2)
            self.assertEqual(summary["recognition_coverage"], 0.5)
            self.assertEqual(summary["end_to_end_accuracy"], 0.5)

    def test_split_config_rejects_duplicate_utterance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.make_config(root, count=15)
            config["trials"][1]["utterance_group"] = config["trials"][0]["utterance_group"]
            path = root / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_stage1_config(path)

    def test_short_audio_is_padded_to_one_three_second_trial(self):
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.wav"
            target = root / "target.wav"
            sf.write(source, np.ones(16000, dtype=np.float32), 16000)
            prepare_wav(source, target)
            audio, sr = sf.read(target, dtype="float32")
            self.assertEqual((sr, len(audio)), (16000, 48000))
            self.assertTrue(np.allclose(audio[16000:], 0.0))

    def test_long_audio_uses_center_three_seconds(self):
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.wav"
            target = root / "target.wav"
            original = np.arange(80000, dtype=np.float32) / 80000
            sf.write(source, original, 16000, subtype="FLOAT")
            prepare_wav(source, target)
            audio, sr = sf.read(target, dtype="float32")
            self.assertEqual((sr, len(audio)), (16000, 48000))
            self.assertTrue(np.allclose(audio, original[16000:64000]))

    def test_formal_summary_has_30_directions_and_450_trials(self):
        rows = []
        for sender in USERS:
            for receiver in USERS:
                if receiver == sender:
                    continue
                for index in range(15):
                    rows.append({"sender_id": sender, "receiver_id": receiver, "evaluated": 1, "correct": 1, "end_to_end_correct": 1, "verified": 1, "received_packets": 6, "decoded_packets": 6, "missing_decoded_packets": 0})
        summary = aggregate(rows)
        self.assertEqual((len(rows), len(summary)), (450, 30))
        self.assertTrue(all(row["end_to_end_accuracy"] == 1 for row in summary))

    def test_vctk_truth_is_not_transmitted_in_packet_header(self):
        import numpy as np
        import group_client

        args = SimpleNamespace(user_id="A", room_id="test", channel="clean", device="cpu", key_path="unused")
        client = group_client.GroupClient(args)
        client.sock = object()
        captured = {}
        with patch.object(group_client, "xor_bytes", side_effect=lambda body, key: body), patch.object(
            group_client, "send_message", side_effect=lambda sock, header, body: captured.update(header=header)
        ):
            client.send_codes(np.zeros((3, 1, 25), dtype=np.int64))
        self.assertEqual(captured["header"]["sender_id"], "A")
        self.assertNotIn("speaker_label", captured["header"])

    def test_router_fans_one_sender_out_to_five_receivers(self):
        from router_server import RouterState

        class Writer:
            def close(self):
                pass

        async def exercise():
            state = RouterState("stage1", 8)
            sessions = {}
            for user in USERS:
                sessions[user] = await state.register(user, "stage1", Writer())
            header = {"kind": "codes_v1", "sender_id": "A", "seq_id": 0}
            forwarded = await state.forward(sessions["A"], header, b"abc")
            self.assertEqual(forwarded, 5)
            self.assertTrue(sessions["A"].send_queue.empty())
            for user in "BCDEF":
                self.assertEqual(sessions[user].send_queue.get_nowait(), (header, b"abc"))

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
