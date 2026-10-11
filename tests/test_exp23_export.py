import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


class ConstantClassifier(torch.nn.Module):
    def forward(self, waveform):
        batch = waveform.shape[0]
        logits = torch.zeros(batch, 2)
        logits[:, 0] = 3.0
        return logits


class CodesWaveformClassifier(torch.nn.Module):
    def forward(self, codes, waveform):
        batch = waveform.shape[0]
        logits = torch.zeros(batch, 2)
        logits[:, 1] = codes[:, 0, 0].to(torch.float32)
        return logits


class DummyTokenEncoder(torch.nn.Module):
    def forward(self, codes, frame_mask, layer_mask):
        return torch.stack([codes[:, 0, 0].float(), frame_mask[:, 0].float()], dim=-1)


class DummyAudioEncoder(torch.nn.Module):
    def forward(self, waveform):
        return torch.stack([waveform[:, 0], waveform[:, -1]], dim=-1)


class DummyFusion(torch.nn.Module):
    def forward(self, token_embedding, audio_embedding):
        fused = token_embedding + audio_embedding
        return fused, torch.ones_like(fused)


class DummyHead(torch.nn.Module):
    def forward(self, embedding, labels=None):
        return embedding


class DummySpeakerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_encoder = DummyTokenEncoder()
        self.audio_encoder = DummyAudioEncoder()
        self.fusion = DummyFusion()
        self.fusion_head = DummyHead()


class ExportBundleTest(unittest.TestCase):
    def test_bundle_validation_and_prediction(self):
        from speechtokenizer.speaker_identity.inference import SpeakerClassifierBundle, write_export_bundle

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scripted = torch.jit.trace(ConstantClassifier(), torch.zeros(1, 1600))
            artifact = root / 'source.pt'
            scripted.save(str(artifact))
            bundle = root / 'bundle'
            write_export_bundle(bundle, artifact, ['p225', 'p226'], {'sample_rate': 16000, 'window_samples': 1600}, {'temperature': 1.0}, {'speaker_count': 2})
            classifier = SpeakerClassifierBundle(bundle, threshold=0.5)
            prediction = classifier.predict(torch.zeros(1600))
            self.assertEqual(prediction.predicted_speaker, 'p225')
            self.assertTrue(prediction.verified)

    def test_bundle_requires_all_files_and_matching_hashes(self):
        from speechtokenizer.speaker_identity.inference import validate_export_bundle
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, 'missing'):
                validate_export_bundle(root)

    def test_codes_waveform_bundle_requires_and_uses_codes(self):
        from speechtokenizer.speaker_identity.inference import SpeakerClassifierBundle, write_export_bundle

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scripted = torch.jit.trace(CodesWaveformClassifier(), (torch.ones(1, 3, 5, dtype=torch.long), torch.zeros(1, 1600)))
            artifact = root / 'source.pt'
            scripted.save(str(artifact))
            bundle = root / 'bundle'
            write_export_bundle(
                bundle,
                artifact,
                ['p225', 'p226'],
                {'sample_rate': 16000, 'window_samples': 1600, 'rvq_layers': 3},
                {'temperature': 1.0},
                {'speaker_count': 2, 'input_mode': 'codes_waveform'},
            )
            classifier = SpeakerClassifierBundle(bundle, threshold=0.5)
            with self.assertRaisesRegex(ValueError, 'codes'):
                classifier.predict(torch.zeros(1600))
            prediction = classifier.predict(torch.zeros(1600), codes=torch.ones(3, 5, dtype=torch.long))
            self.assertEqual(prediction.predicted_speaker, 'p226')

    def test_checkpoint_wrapper_traces_codes_and_waveform_inputs(self):
        from speechtokenizer.speaker_identity.inference import CodesWaveformLogits

        wrapper = CodesWaveformLogits(DummySpeakerModel()).eval()
        codes = torch.ones(1, 3, 5, dtype=torch.long)
        waveform = torch.tensor([[2.0, 0.0, 3.0]])
        scripted = torch.jit.trace(wrapper, (codes, waveform))
        logits = scripted(codes, waveform)
        self.assertEqual(tuple(logits.shape), (1, 2))
        torch.testing.assert_close(logits, torch.tensor([[3.0, 4.0]]))

    def test_export_cli_builds_bundle_from_checkpoint_and_manifest(self):
        from scripts.export_exp23_speaker_classifier import main as export_main
        from speechtokenizer.speaker_identity.inference import SpeakerClassifierBundle

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = DummySpeakerModel()
            checkpoint = root / 'checkpoint.pt'
            torch.save({'state_dict': model.state_dict(), 'config': {'task': {'speaker_count': 2}}, 'ecapa_source': 'fake'}, checkpoint)
            manifest = root / 'records.json'
            manifest.write_text(json.dumps([
                {'speaker_id': 'p226', 'label_index': 1},
                {'speaker_id': 'p225', 'label_index': 0},
            ]), encoding='utf-8')
            calibration = root / 'temperature.json'
            calibration.write_text(json.dumps({'temperature': 1.0}), encoding='utf-8')
            bundle = root / 'bundle'
            with mock.patch('scripts.export_exp23_speaker_classifier.build_model', return_value=model):
                result = export_main([
                    '--checkpoint', str(checkpoint),
                    '--cache-manifest', str(manifest),
                    '--calibration', str(calibration),
                    '--output-dir', str(bundle),
                    '--window-samples', '3',
                    '--code-frames', '5',
                ])
            self.assertEqual(result, 0)
            classifier = SpeakerClassifierBundle(bundle, threshold=0.5)
            self.assertEqual(classifier.labels, ['p225', 'p226'])
            prediction = classifier.predict(torch.tensor([2.0, 0.0, 3.0]), codes=torch.ones(3, 5, dtype=torch.long))
            self.assertEqual(prediction.predicted_speaker, 'p226')
