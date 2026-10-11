import unittest

import torch
from torch.nn import functional as F


class AamSoftmaxTest(unittest.TestCase):
    def test_target_logit_receives_angular_margin(self):
        from speechtokenizer.speaker_identity.losses import AAMSoftmaxHead

        head = AAMSoftmaxHead(2, 2, scale=30, margin=0.2)
        with torch.no_grad():
            head.weight.copy_(torch.eye(2))
        logits = head(F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1), torch.tensor([0]))
        self.assertLess(logits[0, 0].item(), 30.0)
        self.assertAlmostEqual(logits[0, 1].item(), 0.0, places=5)

    def test_joint_loss_combines_three_heads(self):
        from speechtokenizer.speaker_identity.losses import joint_classification_loss

        logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
        loss = joint_classification_loss(logits, logits, logits, torch.tensor([0, 1]), auxiliary_weight=0.3)
        self.assertGreater(loss.item(), 0.0)


if __name__ == '__main__':
    unittest.main()
