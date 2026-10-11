import unittest

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


class TrainingPolicyTest(unittest.TestCase):
    def test_replication_gate_is_validation_only(self):
        from speechtokenizer.speaker_identity.training import replication_decision
        self.assertEqual(replication_decision(0.89), 'freeze_and_replicate')
        self.assertEqual(replication_decision(0.86), 'bounded_revision')
        self.assertEqual(replication_decision(0.84), 'stop')

    def test_validation_dataset_uses_deterministic_center_crop(self):
        from speechtokenizer.speaker_identity.cache import CacheRecord, atomic_write_npz
        from speechtokenizer.speaker_identity.training import CachedSpeakerDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'source.flac'
            reconstruction = root / 'reconstruction.flac'
            codes = root / 'codes.npz'
            audio = np.linspace(-0.1, 0.1, 64000, dtype=np.float32)
            sf.write(source, audio, 16000)
            sf.write(reconstruction, audio, 16000)
            atomic_write_npz(codes, np.arange(600, dtype=np.int64).reshape(3, 200) % 1024)
            record = CacheRecord('p225', 0, 'u1', 'validation', source, 'lca', 3, codes, reconstruction, 64000, 200, 64000, None, 'ckpt', 'cfg')
            dataset = CachedSpeakerDataset([record], split='validation', crop_samples=48000)
            first = dataset[0]
            second = dataset[0]
            self.assertTrue(np.array_equal(first['codes'], second['codes']))
            self.assertTrue(np.array_equal(first['waveform'], second['waveform']))

    def test_training_rejects_test_manifest(self):
        from speechtokenizer.speaker_identity.training import CachedSpeakerDataset
        with self.assertRaisesRegex(ValueError, 'test'):
            CachedSpeakerDataset([], split='test', for_training=True)

    def test_collation_respects_active_rvq_layers(self):
        from speechtokenizer.speaker_identity.training import collate_cache_batch
        item = {
            'codes': np.zeros((3, 4), dtype=np.int64),
            'waveform': np.zeros(8, dtype=np.float32),
            'label': 0,
            'active_layers': 2,
        }
        batch = collate_cache_batch([item])
        self.assertEqual(batch['layer_mask'].tolist(), [[True, True, False]])


if __name__ == '__main__':
    unittest.main()
