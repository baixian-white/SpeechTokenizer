import unittest

import torch


class FakeEcapa(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(5)])

    def forward(self, features):
        value = features
        for block in self.blocks:
            value = torch.relu(block(value))
        return value.mean(dim=1)


class EcapaGradientScopeTest(unittest.TestCase):
    def test_stage_b_unfreezes_only_configured_patterns(self):
        from speechtokenizer.speaker_identity.audio_encoder import CodecAwareEcapaEncoder

        encoder = CodecAwareEcapaEncoder(FakeEcapa(), input_dim=8, ecapa_dim=8, output_dim=16)
        encoder.configure_stage('B', ['embedding_model.blocks.4'])
        names = encoder.trainable_parameter_names()
        self.assertTrue(any('blocks.4' in name for name in names))
        self.assertFalse(any('blocks.0' in name for name in names))
        self.assertTrue(any('projection' in name for name in names))

    def test_stage_a_freezes_ecapa_but_trains_projection(self):
        from speechtokenizer.speaker_identity.audio_encoder import CodecAwareEcapaEncoder

        encoder = CodecAwareEcapaEncoder(FakeEcapa(), input_dim=8, ecapa_dim=8, output_dim=16)
        encoder.configure_stage('A', [])
        names = encoder.trainable_parameter_names()
        self.assertFalse(any('embedding_model' in name for name in names))
        self.assertTrue(any('projection' in name for name in names))
        encoder.train()
        self.assertFalse(encoder.embedding_model.training)


if __name__ == '__main__':
    unittest.main()
