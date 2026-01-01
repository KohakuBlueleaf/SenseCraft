"""
ConvNext Perceptual Loss using torchvision pretrained models.

This module provides perceptual loss based on ConvNext features trained on ImageNet.
It extracts multi-scale features from ConvNext blocks and computes either:
- Direct MSE loss on feature maps
- Gram matrix loss (L1) for texture/style matching

Modified from https://github.com/sypsyp97/convnext_perceptual_loss by sypsyp97
Original code is licensed under the Apache License 2.0.
This modified code fixes gradient flow issues in the original implementation.

Example:
    >>> from sensecraft.loss import ConvNextPerceptualLoss
    >>> loss_fn = ConvNextPerceptualLoss(
    ...     model_type=ConvNextType.SMALL,
    ...     feature_layers=[2, 4, 8, 14],
    ...     use_gram=False,
    ...     input_range=(0, 1),
    ... )
    >>> loss = loss_fn(predicted_image, target_image)
"""

from enum import Enum
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from .config import register_loss, ValueRange


class ConvNextType(Enum):
    """Available ConvNext model types from torchvision.

    Attributes:
        TINY: ConvNext-Tiny (~28M params)
        SMALL: ConvNext-Small (~50M params)
        BASE: ConvNext-Base (~89M params)
        LARGE: ConvNext-Large (~198M params)
    """

    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"


@register_loss("convnext", ValueRange.SYMMETRIC, is_2d_only=True)
class ConvNextPerceptualLoss(nn.Module):
    """ConvNext-based perceptual loss using ImageNet-pretrained features.

    This loss extracts features from multiple layers of a pretrained ConvNext
    model and computes the difference between predicted and target features.
    It supports both direct feature matching (MSE) and Gram matrix matching (L1).

    The model weights are frozen and the forward pass handles gradient flow
    correctly to allow backpropagation through the input image.

    Attributes:
        mean: ImageNet mean for normalization
        std: ImageNet std for normalization
        feature_weights: Per-layer weights for loss computation
        blocks: ConvNext blocks used for feature extraction
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
        input_range: Tuple[float, float] = (-1, 1),
        layer_weight_decay: float = 1.0,
    ):
        """Initialize the ConvNext perceptual loss module.

        Args:
            model_type: Which ConvNext variant to use (TINY, SMALL, BASE, LARGE)
            feature_layers: List of layer indices to extract features from.
                ConvNext-Small has ~40 blocks, so valid indices are 0-39.
            feature_weights: Optional explicit weights for each layer. If None,
                weights are computed using exponential decay.
            use_gram: If True, compute Gram matrix loss (good for style/texture).
                If False, compute direct MSE loss on features.
            input_range: Expected (min, max) range of input values. Common values:
                - (-1, 1) for normalized inputs
                - (0, 1) for standard image tensors
            layer_weight_decay: Decay factor for automatic weight computation.
                Higher values weight later layers more heavily.
        """
        super().__init__()

        self.input_range = input_range
        self.use_gram = use_gram
        self.feature_layers = feature_layers

        # Calculate weights with decay if not specified
        if feature_weights is None:
            decay_values = [layer_weight_decay**i for i in range(len(feature_layers))]
            weights = torch.tensor(decay_values, dtype=torch.float32)
            weights = weights / weights.sum()
        else:
            weights = torch.tensor(feature_weights, dtype=torch.float32)

        assert len(feature_layers) == len(
            weights
        ), "Number of feature layers must match number of weights"
        self.register_buffer("feature_weights", weights)

        # Load pretrained ConvNext model
        model_name = f"convnext_{model_type.value}"
        try:
            weights_enum = getattr(
                torchvision.models, f"ConvNeXt_{model_type.value.capitalize()}_Weights"
            )
            weights = weights_enum.DEFAULT
            model = getattr(torchvision.models, model_name)(weights=weights)
        except (AttributeError, ImportError):
            model = getattr(torchvision.models, model_name)(pretrained=True)

        # Extract blocks and ensure they're in eval mode
        self.blocks = nn.ModuleList()
        for stage in model.features:
            if isinstance(stage, nn.Sequential):
                self.blocks.extend(stage)
            else:
                self.blocks.append(stage)

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

            if self.use_gram:
                input_gram = self.gram_matrix(input_feat)
                target_gram = self.gram_matrix(target_feat)
                layer_loss = nn.functional.l1_loss(input_gram, target_gram)
            else:
                layer_loss = nn.functional.mse_loss(input_feat, target_feat)

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

        loss = self.compute_feature_loss(
            input_features, target_features, self.feature_layers, self.feature_weights
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
    loss_fn = ConvNextPerceptualLoss().to(device)

    test_input_image = torch.rand(1, 3, 256, 256, device=device)
    output = test_net(test_input_image + torch.randn_like(test_input_image) * 0.05)
    print(f"Loss before: {loss_fn(test_input_image, output).item()}")

    test_datasets = torch.rand(1024, 3, 256, 256, device=device)
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
