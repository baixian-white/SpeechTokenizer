import unittest

import torch


class TokenEncoderShapeTest(unittest.TestCase):
    def test_token_encoder_returns_normalized_embedding(self):
        from speechtokenizer.speaker_identity.token_encoder import RVQTokenEncoder

        encoder = RVQTokenEncoder(rvq_layers=3, codebook_size=1024, embedding_dim=32, model_dim=64, output_dim=256, block_count=2)
        codes = torch.randint(0, 1024, (2, 3, 20))
        frame_mask = torch.ones(2, 20, dtype=torch.bool)
        layer_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
        embedding = encoder(codes, frame_mask, layer_mask)
        self.assertEqual(embedding.shape, (2, 256))
        self.assertTrue(torch.allclose(embedding.norm(dim=-1), torch.ones(2), atol=1e-5))

    def test_padding_and_masked_layers_do_not_change_embedding(self):
        from speechtokenizer.speaker_identity.token_encoder import RVQTokenEncoder

        torch.manual_seed(3)
        encoder = RVQTokenEncoder(rvq_layers=3, codebook_size=1024, embedding_dim=16, model_dim=32, output_dim=64, block_count=1).eval()
        codes = torch.randint(0, 1024, (1, 3, 8))
        frame_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        layer_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        changed = codes.clone()
        changed[:, 2] = torch.randint(0, 1024, (1, 8))
        changed[:, :, 5:] = torch.randint(0, 1024, (1, 3, 3))
        self.assertTrue(torch.allclose(encoder(codes, frame_mask, layer_mask), encoder(changed, frame_mask, layer_mask), atol=1e-5))


class FullModelShapeTest(unittest.TestCase):
    def test_full_model_returns_three_heads_and_gate(self):
        from speechtokenizer.speaker_identity.model import SpeakerIdentityModel

        class Encoder(torch.nn.Module):
            def forward(self, *args):
                return torch.nn.functional.normalize(torch.ones(args[0].shape[0], 256), dim=-1)

        model = SpeakerIdentityModel(Encoder(), Encoder(), speaker_count=110, embedding_dim=256)
        output = model({
            'codes': torch.zeros(2, 3, 5, dtype=torch.long),
            'frame_mask': torch.ones(2, 5, dtype=torch.bool),
            'layer_mask': torch.ones(2, 3, dtype=torch.bool),
            'audio_features': torch.ones(2, 5, 256),
        })
        self.assertEqual(output.fusion_logits.shape, (2, 110))
        self.assertEqual(output.fusion_embedding.shape, (2, 256))
        self.assertEqual(output.gate.shape, (2, 256))

    def test_eval_mode_does_not_apply_aam_target_margin(self):
        from speechtokenizer.speaker_identity.model import SpeakerIdentityModel

        class Encoder(torch.nn.Module):
            def forward(self, *args):
                return torch.nn.functional.normalize(torch.ones(args[0].shape[0], 256), dim=-1)

        model = SpeakerIdentityModel(Encoder(), Encoder(), speaker_count=2, embedding_dim=256)
        batch = {'codes': torch.zeros(1, 3, 2, dtype=torch.long), 'frame_mask': torch.ones(1, 2, dtype=torch.bool), 'layer_mask': torch.ones(1, 3, dtype=torch.bool), 'audio_features': torch.ones(1, 2, 256), 'labels': torch.tensor([0])}
        model.train()
        train_logits = model(batch).fusion_logits
        model.eval()
        eval_logits = model(batch).fusion_logits
        self.assertFalse(torch.allclose(train_logits, eval_logits))


if __name__ == '__main__':
    unittest.main()
