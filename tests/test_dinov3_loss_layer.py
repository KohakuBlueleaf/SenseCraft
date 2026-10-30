"""
ViTDinoV3PerceptualLoss must read the block that loss_layer names in the full model,
also after the layers past the target have been truncated.

Uses a small randomly initialized DINOv3 ViT, so no weights are downloaded.
"""

import unittest
from unittest import mock

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from sensecraft.loss import gram_dinov3
from sensecraft.loss.gram_dinov3 import ModelType, ViTDinoV3PerceptualLoss

NUM_LAYERS = 12


def tiny_vit(*args, **kwargs):
    torch.manual_seed(0)
    config = DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        patch_size=16,
        num_register_tokens=4,
    )
    return DINOv3ViTModel(config).eval()


def make_loss(loss_layer):
    with mock.patch.object(gram_dinov3.DINOv3ViTModel, "from_pretrained", side_effect=tiny_vit):
        return ViTDinoV3PerceptualLoss(
            model_type=ModelType.BASE, use_gram=False, loss_layer=loss_layer
        ).eval()


class TestDinoV3LossLayer(unittest.TestCase):
    def test_reads_the_named_block(self):
        x = torch.rand(1, 3, 64, 64)
        for loss_layer in (-1, -2, -4, -6, 0, 5, NUM_LAYERS - 1):
            expected = loss_layer % NUM_LAYERS
            with self.subTest(loss_layer=loss_layer):
                loss = make_loss(loss_layer)
                self.assertEqual(len(loss.model.model.layer), expected + 1)
                ran = []
                hooks = [
                    block.register_forward_hook(lambda m, i, o, k=k: ran.append((k, o)))
                    for k, block in enumerate(loss.model.model.layer)
                ]
                with torch.no_grad():
                    out = loss.dinov3_fwd(loss.normalize_input(x))
                for h in hooks:
                    h.remove()
                last_block, last_out = ran[-1]
                self.assertEqual(last_block, expected)
                self.assertTrue(torch.equal(out, last_out[0] if isinstance(last_out, tuple) else last_out))

    def test_out_of_range_layer_raises(self):
        for loss_layer in (NUM_LAYERS, -NUM_LAYERS - 1):
            with self.subTest(loss_layer=loss_layer):
                with self.assertRaises(ValueError):
                    make_loss(loss_layer)


if __name__ == "__main__":
    unittest.main()
