"""
ConvNext Perceptual Loss using DINOv3 self-supervised pretrained models.

This module provides perceptual loss based on ConvNext features from Facebook's
DINOv3 self-supervised learning framework. Unlike ImageNet-supervised ConvNext,
DINOv3 models learn more general visual representations without label supervision.

Key features:
- Self-supervised features (DINOv3) for better generalization
- Multi-layer or single-layer feature extraction
- Optional L2 normalization before loss computation
- Gram matrix loss for texture/style matching
- Direct MSE loss for content matching

Inspired by https://github.com/sypsyp97/convnext_perceptual_loss by sypsyp97
This implementation uses DINOv3 ConvNext models instead of ImageNet-supervised
ConvNext, and adds gradient flow fixes and new features.

Requires:
    transformers >= 4.56.0

Example:
    >>> from sensecraft.loss import ConvNextDinoV3PerceptualLoss
    >>> from sensecraft.loss.convnext_dinov3 import ConvNextType
    >>>
    >>> # Multi-layer mode (traditional)
    >>> loss_fn = ConvNextDinoV3PerceptualLoss(
    ...     model_type=ConvNextType.SMALL,
    ...     feature_layers=[2, 4, 8, 14],
    ...     use_gram=False,
    ... )
    >>>
    >>> # Single-layer mode with normalization (recommended)
    >>> loss_fn = ConvNextDinoV3PerceptualLoss(
    ...     model_type=ConvNextType.SMALL,
    ...     loss_layer=-1,  # last layer
    ...     use_norm=True,
    ...     use_gram=False,
    ... )
    >>> loss = loss_fn(predicted_image, target_image)
"""

from enum import Enum
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Need 4.56.0 ^
from transformers import DINOv3ConvNextModel

from .config import register_loss, ValueRange


class ConvNextType(Enum):
    """Available DINOv3 ConvNext model types from HuggingFace.

    All models are pretrained on LVD-1689M dataset using self-supervised learning.

    Attributes:
        TINY: ConvNext-Tiny DINOv3 (~28M params)
        SMALL: ConvNext-Small DINOv3 (~50M params)
        BASE: ConvNext-Base DINOv3 (~89M params)
        LARGE: ConvNext-Large DINOv3 (~198M params)
    """

    TINY = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    SMALL = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
    BASE = "facebook/dinov3-convnext-base-pretrain-lvd1689m"
    LARGE = "facebook/dinov3-convnext-large-pretrain-lvd1689m"


@register_loss("dino_convnext", ValueRange.SYMMETRIC, is_2d_only=True)
class ConvNextDinoV3PerceptualLoss(nn.Module):
    """ConvNext Perceptual Loss Module with DinoV3 ConvNext model

    Supports two modes:
    1. Multi-layer mode (legacy): Use feature_layers to specify multiple layers
    2. Single-layer mode: Use loss_layer to specify a single layer with norm/mse loss

    When loss_layer is specified (not None), it takes precedence over feature_layers.
    """

    mean: torch.Tensor
    std: torch.Tensor
    feature_weights: torch.Tensor

    def __init__(
        self,
        model_type: ConvNextType = ConvNextType.SMALL,
        feature_layers: List[int] = [0, 2, 4, 8, 14],
        feature_weights: Optional[List[float]] = None,
        use_gram: bool = False,
        use_norm: bool = True,
        input_range: Tuple[float, float] = (-1, 1),
        layer_weight_decay: float = 1.0,
        loss_layer: Optional[int] = 14,
    ):
        """Initialize perceptual loss module

        Args:
            model_type: ConvNext model variant to use
            feature_layers: List of layer indices for multi-layer mode
            feature_weights: Optional weights for each feature layer
            use_gram: If True, compute Gram matrix loss (L1), else MSE on features
            use_norm: If True, L2-normalize features before computing loss
            input_range: Expected input value range (min, max)
            layer_weight_decay: Exponential decay for layer weights if not specified
            loss_layer: If specified, use single-layer mode with this layer index
        """
        super().__init__()

        self.input_range = input_range
        self.use_gram = use_gram
        self.use_norm = use_norm
        self.loss_layer = loss_layer

        # If loss_layer is specified, use single-layer mode
        if loss_layer is not None:
            self.feature_layers = [loss_layer] if loss_layer >= 0 else None
            weights = torch.tensor([1.0], dtype=torch.float32)
        else:
            self.feature_layers = feature_layers
            # Calculate weights with decay if not specified
            if feature_weights is None:
                decay_values = [
                    layer_weight_decay**i for i in range(len(feature_layers))
                ]
                weights = torch.tensor(decay_values, dtype=torch.float32)
                weights = weights / weights.sum()
            else:
                weights = torch.tensor(feature_weights, dtype=torch.float32)

            assert len(feature_layers) == len(
                weights
            ), "Number of feature layers must match number of weights"

        self.register_buffer("feature_weights", weights)

        # Load pretrained ConvNext model
        model_name = model_type.value
        model = DINOv3ConvNextModel.from_pretrained(model_name)

        # Extract blocks and ensure they're in eval mode
        self.blocks = nn.ModuleList()
        for stage in model.stages:
            self.blocks.extend(stage.downsample_layers)
            self.blocks.extend(stage.layers)

        self.blocks = self.blocks.eval().requires_grad_(False)

        # Register normalization parameters
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

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
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        if normalize:
            gram = gram / (c * h * w)
        return gram

    def compute_feature_loss(
        self,
        input_features: List[torch.Tensor],
        target_features: List[torch.Tensor],
        layers_indices: List[int],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature loss ensuring scalar output"""
        losses = []

        for idx, weight in zip(layers_indices, weights):
            input_feat = input_features[idx]
            target_feat = target_features[idx]

            # Apply L2 normalization if enabled
            if self.use_norm:
                # Normalize along channel dimension for 4D tensors (B, C, H, W)
                input_feat = F.normalize(input_feat, dim=1)
                target_feat = F.normalize(target_feat, dim=1)

            if self.use_gram:
                input_gram = self.gram_matrix(input_feat)
                target_gram = self.gram_matrix(target_feat)
                layer_loss = F.l1_loss(input_gram, target_gram)
            else:
                layer_loss = F.mse_loss(input_feat, target_feat)

            losses.append(weight * layer_loss)

        # Sum all losses and ensure scalar output
        return torch.stack(losses).sum()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute loss"""
        input = self.normalize_input(input)
        target = self.normalize_input(target)

        # Extract features
        input_features = []
        target_features = []

        x_input = input
        x_target = target
        for block in self.blocks:
            with torch.enable_grad() if x_input.requires_grad else torch.no_grad():
                x_input = block(x_input)
            with torch.enable_grad() if x_target.requires_grad else torch.no_grad():
                x_target = block(x_target)
            input_features.append(x_input)
            target_features.append(x_target)

        # Handle single-layer mode with negative indexing
        if self.loss_layer is not None and self.loss_layer < 0:
            # Use negative index to get layer from end
            feature_layers = [len(input_features) + self.loss_layer]
        else:
            feature_layers = self.feature_layers

        loss = self.compute_feature_loss(
            input_features, target_features, feature_layers, self.feature_weights
        )

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
    test_opt = torch.optim.Adam(test_net.parameters(), lr=1e-3)
    loss_fn = ConvNextDinoV3PerceptualLoss(
        feature_layers=[2, 4, 8, 14, 20, 26, 32, 38],
        feature_weights=[1.0] * 8,
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
