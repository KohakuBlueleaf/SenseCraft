"""
ViT DINOv3 Perceptual Loss using Vision Transformer models.

This module provides perceptual loss based on Vision Transformer (ViT) features
from Facebook's DINOv3 self-supervised learning framework. ViT models process
images as sequences of patches, providing a different perspective compared to
convolutional models.

Key features:
- Self-supervised features (DINOv3 ViT) for semantic understanding
- Single-layer feature extraction with configurable layer selection
- L2 normalization for feature comparison
- Gram matrix loss for texture/style matching
- Direct MSE loss for content matching

Note:
    When using L2 normalization without Gram matrix (use_norm=True, use_gram=False),
    this loss is equivalent to the DINO perceptual loss described in:
    https://na-vae.github.io/dino_perceptual/

Requires:
    transformers >= 4.56.0

Example:
    >>> from sensecraft.loss import ViTDinoV3PerceptualLoss
    >>> from sensecraft.loss.gram_dinov3 import ModelType
    >>>
    >>> loss_fn = ViTDinoV3PerceptualLoss(
    ...     model_type=ModelType.SMALL_PLUS,
    ...     use_norm=True,
    ...     use_gram=True,
    ...     loss_layer=-4,  # 4th layer from end (default)
    ... )
    >>> loss = loss_fn(predicted_image, target_image)
"""

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

# Need 4.56.0 ^
from transformers import DINOv3ViTModel

from .config import register_loss, ValueRange


class ModelType(Enum):
    """Available DINOv3 ViT model types from HuggingFace.

    All models are pretrained on LVD-1689M dataset using self-supervised learning.
    The number after 'vit' indicates the patch size (16x16 pixels).

    Attributes:
        SMALL: ViT-Small with 16x16 patches (~22M params)
        SMALL_PLUS: ViT-Small-Plus with 16x16 patches (larger hidden dim)
        BASE: ViT-Base with 16x16 patches (~86M params)
        LARGE: ViT-Large with 16x16 patches (~307M params)
    """

    SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
    SMALL_PLUS = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    BASE = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    LARGE = "facebook/dinov3-vitl16-pretrain-lvd1689m"


@register_loss("dino_vit", ValueRange.SYMMETRIC, is_2d_only=True)
class ViTDinoV3PerceptualLoss(nn.Module):
    """ViT-based perceptual loss using DINOv3 self-supervised features.

    This loss extracts features from a Vision Transformer pretrained with DINOv3
    self-supervised learning. Unlike CNN-based losses, ViT processes images as
    sequences of patches, capturing global context from early layers.

    The loss supports:
    - L2 normalization of features before comparison
    - Gram matrix computation for texture/style matching
    - Layer selection for feature extraction

    Attributes:
        mean: ImageNet mean for normalization
        std: ImageNet std for normalization
        model: DINOv3 ViT model
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        model_type: ModelType = ModelType.SMALL_PLUS,
        use_gram: bool = True,
        use_norm: bool = True,
        input_range: tuple[float, float] = (-1, 1),
        loss_layer: int = -4,
    ):
        """Initialize perceptual loss module.

        Args:
            model_type: Which DINOv3 ViT variant to use
            use_gram: If True, compute Gram matrix loss (L1). If False, MSE on features.
            use_norm: If True, L2 normalize features before loss computation
            input_range: Expected (min, max) range of input values
            loss_layer: Layer index to extract features from. Supports Python-style
                negative indexing (e.g., -1 for last layer, -4 for 4th from last).
                Default is -4 which typically gives good perceptual results.
        """
        super().__init__()

        self.input_range = input_range
        self.use_gram = use_gram
        self.use_norm = use_norm
        self.loss_layer = loss_layer

        # Load pretrained ViT model
        model_name = model_type.value
        self.model: DINOv3ViTModel = DINOv3ViTModel.from_pretrained(model_name)

        # Store number of layers for negative indexing
        self.num_layers = len(self.model.layer)

        # Register normalization parameters
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _resolve_layer_index(self, layer_idx: int) -> int:
        """Convert potentially negative layer index to positive index."""
        if layer_idx < 0:
            return self.num_layers + layer_idx
        return layer_idx

    def dinov3_fwd(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(
            self.model.embeddings.patch_embeddings.weight.dtype
        )
        hidden_states = self.model.embeddings(pixel_values, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(pixel_values)

        # Resolve negative index to actual layer index
        target_layer = self._resolve_layer_index(self.loss_layer)

        for i, layer_module in enumerate(self.model.layer):
            layer_head_mask = None
            hidden_states = layer_module(
                hidden_states,
                attention_mask=layer_head_mask,
                position_embeddings=position_embeddings,
            )
            if i == target_layer:
                return hidden_states

        return hidden_states

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor"""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        min_val, max_val = self.input_range
        x = (x - min_val) / (max_val - min_val)
        x = (x - self.mean) / self.std

        return x

    def gram_matrix(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Compute Gram matrix of feature maps"""
        b, seq, c = x.size()
        features = x
        gram = torch.bmm(features.transpose(1, 2), features)  # b, c, c
        if normalize:
            gram = gram / seq
        return gram

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Forward pass to compute loss"""
        input = self.normalize_input(input)
        target = self.normalize_input(target)
        with torch.enable_grad() if input.requires_grad else torch.no_grad():
            x_feat = self.dinov3_fwd(input)
        with torch.enable_grad() if target.requires_grad else torch.no_grad():
            y_feat = self.dinov3_fwd(target)

        if self.use_norm:
            x_feat = F.normalize(x_feat, dim=-1)
            y_feat = F.normalize(y_feat, dim=-1)

        if self.use_gram:
            x_gram = self.gram_matrix(x_feat)
            y_gram = self.gram_matrix(y_feat)
            loss = nn.functional.l1_loss(x_gram, y_gram, **kwargs)
        else:
            loss = nn.functional.mse_loss(x_feat, y_feat, **kwargs)

        return loss


def main():
    class TestNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    """Main function for testing"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_net = TestNet().to(device).train()
    test_opt = torch.optim.Adam(test_net.parameters(), lr=1e-2)
    loss_fn = ViTDinoV3PerceptualLoss(
        model_type=ModelType.SMALL_PLUS,
        use_gram=True,
    ).to(device)

    test_input_image = torch.rand(1, 3, 256, 256, device=device)
    output = test_net(test_input_image + torch.randn_like(test_input_image) * 0.05)
    print(f"Loss before: {loss_fn(test_input_image, output).item()}")

    test_datasets = torch.rand(128, 3, 256, 256, device=device)
    for data in test_datasets:
        data = data.unsqueeze(0)
        output = test_net(data + torch.randn_like(data) * 0.05)
        loss = loss_fn(data, output)
        loss.backward()
        print(
            f"Loss: {loss.item()} - Grad norm: {test_net.conv.weight.grad.norm().item()}",
            flush=True,
            end="\r",
        )
        test_opt.step()
        test_opt.zero_grad()
    print("\nFinished small test.")

    output = test_net(test_input_image + torch.randn_like(test_input_image) * 0.05)
    print(f"Loss after: {loss_fn(test_input_image, output).item()}")


if __name__ == "__main__":
    main()
