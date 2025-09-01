"""
Modified from https://github.com/sypsyp97/convnext_perceptual_loss
by sypsyp97
Original code is licensed under the Apache License 2.0.

This modified code utilize ViT model instead of ConvNext
"""
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
# Need 4.56.0 ^
from transformers import DINOv3ViTModel


class ModelType(Enum):
    """Available ConvNext model types"""

    SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
    SMALL_PLUS = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    BASE = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    LARGE = "facebook/dinov3-vitl16-pretrain-lvd1689m"


class ViTDinoV3PerceptualLoss(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        model_type: ModelType = ModelType.SMALL_PLUS,
        use_gram: bool = True,
        input_range: tuple[float, float] = (-1, 1),
    ):
        """Initialize perceptual loss module"""
        super().__init__()

        self.input_range = input_range
        self.use_gram = use_gram

        # Load pretrained ConvNext model
        model_name = model_type.value
        self.model = DINOv3ViTModel.from_pretrained(model_name)

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
        b, seq, c = x.size()
        features = x
        gram = torch.bmm(features.transpose(1, 2), features) # b, c, c
        if normalize:
            gram = gram / seq
        return gram

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass to compute loss"""
        input = self.normalize_input(input)
        target = self.normalize_input(target)
        with torch.enable_grad() if input.requires_grad else torch.no_grad():
            x_feat = self.model(input).last_hidden_state
        with torch.enable_grad() if target.requires_grad else torch.no_grad():
            y_feat = self.model(target).last_hidden_state

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
