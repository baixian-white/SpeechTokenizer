import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class FakeTokenizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.encode_grad_enabled = None
        self.decode_grad_enabled = None
        self.last_n_q = None

    def encode(self, audio, n_q, st=0):
        self.encode_grad_enabled = torch.is_grad_enabled()
        self.last_n_q = n_q
        frames = max(1, audio.shape[-1] // 320)
        return torch.zeros((n_q, 1, frames), dtype=torch.long, device=audio.device)

    def decode(self, codes, st=0):
        self.decode_grad_enabled = torch.is_grad_enabled()
        return torch.zeros((1, 1, codes.shape[-1] * 320), device=codes.device) + self.weight * 0


class FrozenTokenizerTest(unittest.TestCase):
    def test_lca_nas_config_operations_are_registered(self):
        from nas.model_components import get_nas_ops
        operations = get_nas_ops('weight_norm', 'reflect', False)
        self.assertIn('std_k7', operations)
        self.assertIn('sep_k9', operations)
        self.assertIn('dil_k5', operations)

    def test_bidirectional_slstm_skip_matches_output_channels(self):
        from nas.model_components import SLSTM
        module = SLSTM(4, num_layers=1, skip=True, bidirectional=True)
        output = module(torch.zeros(2, 4, 6))
        self.assertEqual(output.shape, (2, 8, 6))

    def test_encode_decode_freezes_model_disables_grad_and_preserves_state(self):
        from speechtokenizer.speaker_identity.cache import encode_decode_frozen, state_dict_sha256

        model = FakeTokenizer().train()
        before = state_dict_sha256(model.state_dict())
        codes, reconstruction, audit = encode_decode_frozen(
            model,
            torch.zeros((1, 1, 1600)),
            rvq_layers=3,
        )
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.encode_grad_enabled)
        self.assertFalse(model.decode_grad_enabled)
        self.assertFalse(codes.requires_grad)
        self.assertFalse(reconstruction.requires_grad)
        self.assertEqual(codes.shape, (3, 5))
        self.assertEqual(model.last_n_q, 3)
        self.assertEqual(audit['before_sha256'], before)
        self.assertEqual(audit['before_sha256'], audit['after_sha256'])


class CacheCliTest(unittest.TestCase):
    def test_source_audio_is_resampled_to_16k(self):
        import numpy as np
        import soundfile as sf
        module = importlib.import_module('scripts.build_exp23_speaker_cache')
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'source.flac'
            sf.write(path, np.zeros(4800, dtype=np.float32), 48000)
            audio = module.load_source_audio(path)
            self.assertEqual(audio.ndim, 1)
            self.assertEqual(len(audio), 1600)

    def test_model_builder_uses_nas_aware_repository_loader(self):
        module = importlib.import_module('scripts.build_exp23_speaker_cache')
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.json'
            config_path.write_text(json.dumps({'nas_encoder_config': 'best.json'}), encoding='utf-8')
            fake_model = torch.nn.Linear(1, 1)
            with patch.object(module, 'build_tokenizer_model', return_value=fake_model) as builder:
                result = module.build_model(config_path, Path(temp_dir) / 'unused.pt', torch.device('cpu'))
            self.assertIs(result, fake_model)
            builder.assert_called_once_with({'nas_encoder_config': 'best.json'})

    def test_parser_supports_required_matrix_arguments(self):
        module = importlib.import_module('scripts.build_exp23_speaker_cache')
        args = module.build_parser().parse_args(
            [
                '--config', 'config.json',
                '--run-dir', 'run',
                '--cache-root', 'cache',
                '--models', 'base', 'lca',
                '--layers', '1', '3',
                '--device', 'cpu',
                '--max-speakers', '2',
                '--max-items-per-split', '1',
                '--split-kind', 'text-controlled',
                '--dry-run-estimate',
            ]
        )
        self.assertEqual(args.models, ['base', 'lca'])
        self.assertEqual(args.layers, [1, 3])
        self.assertEqual(args.split_kind, 'text-controlled')
        self.assertTrue(args.dry_run_estimate)

    def test_dry_run_estimate_never_builds_or_loads_model(self):
        module = importlib.import_module('scripts.build_exp23_speaker_cache')
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / 'config.json'
            config_path.write_text(
                json.dumps(
                    {
                        'frozen_upstream': {
                            'lca': {'config': 'unused.json', 'checkpoint': 'unused.pt'}
                        },
                        'cache': {'budget_gb': 30, 'free_space_multiplier': 1.2},
                    }
                ),
                encoding='utf-8',
            )
            split_manifest = {
                'speakers': ['p225'],
                'splits': {
                    'train': [{'speaker_id': 'p225', 'utterance_group': 'u1', 'audio_path': 'a.flac', 'transcript_hash': None}],
                    'validation': [],
                    'test': [],
                },
            }
            with patch.object(module, 'build_split_manifest', return_value=split_manifest), patch.object(
                module, 'estimate_manifest_storage', return_value={'estimated_bytes': 1, 'item_count': 1}
            ), patch.object(module, 'preflight_cache_storage', return_value={'estimated_bytes': 1}), patch.object(
                module, 'build_model'
            ) as build_model, patch.object(module, 'load_state') as load_state:
                result = module.main(
                    [
                        '--config', str(config_path),
                        '--run-dir', str(root / 'run'),
                        '--cache-root', str(root / 'cache'),
                        '--models', 'lca',
                        '--layers', '3',
                        '--device', 'cpu',
                        '--dry-run-estimate',
                    ]
                )
            self.assertEqual(result, 0)
            build_model.assert_not_called()
            load_state.assert_not_called()
            self.assertTrue((root / 'run' / 'cache_manifest' / 'cache_storage_estimate.json').is_file())


if __name__ == '__main__':
    unittest.main()
