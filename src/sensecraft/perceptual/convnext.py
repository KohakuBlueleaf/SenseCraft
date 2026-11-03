"""ConvNeXt features at chosen layers, ImageNet-supervised or DINOv3, by MSE or gram matrices.

Layers count the flat net of `sensecraft.nets.convnext` as the 0.3.13 losses did: the stem conv
is 0 and its norm 1, every block and every downsample norm and conv is one layer.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from sensecraft.distance import feature_distance
from sensecraft.kernels import wide
from sensecraft.nets.convnext import TORCHVISION, ConvNeXtConfig, convnext
from sensecraft.registry import Range, register
from sensecraft.weights import DINOV3_CONVNEXT, config, weights

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ConvNeXtTerm(nn.Module):
    """Weighted sum over `feature_layers` of a distance between ConvNeXt feature maps.

    Layers count from 0, and from the end when negative. Without `feature_weights` the weights
    are `layer_weight_decay ** i`, normalized to sum to one.
    """

    def __init__(
        self,
        net: ConvNeXtConfig,
        source: str,
        feature_layers: Sequence[int],
        feature_weights: Sequence[float] | None,
        layer_weight_decay: float,
        use_gram: bool,
        use_norm: bool,
        pretrained: bool,
    ):
        super().__init__()
        depth = net.layer_count
        layers = [layer + depth if layer < 0 else layer for layer in feature_layers]
        if not layers or not all(0 <= layer < depth for layer in layers):
            raise ValueError(
                f"feature_layers {feature_layers} outside a {depth}-layer net"
            )

        # Weights rounded to fp32 as upstream held them, then kept as floats so no cast moves them
        if feature_weights is None:
            decay = torch.tensor(
                [layer_weight_decay**i for i in range(len(layers))], dtype=torch.float32
            )
            scale = decay / decay.sum()
        else:
            scale = torch.tensor(feature_weights, dtype=torch.float32)
        if len(scale) != len(layers):
            raise ValueError(
                f"{len(scale)} feature_weights for {len(layers)} feature_layers"
            )
        self.layer_weights = scale.tolist()

        # The net returns each tapped layer once, shallow first; `slots` finds each layer there
        taps = sorted(set(layers))
        self.slots = [taps.index(layer) for layer in layers]
        self.net = convnext(net, taps)
        if pretrained:
            state = weights(source)
            self.net.load_state_dict({key: state[key] for key in self.net.state_dict()})
        self.net.freeze()
        self.use_gram, self.use_norm = use_gram, use_norm
        self.register_buffer("mean", Tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", Tensor(STD).view(1, 3, 1, 1))

    def features(self, x: Tensor) -> list[Tensor]:
        """Feature maps of an image in [-1, 1]; a gray image is repeated to three channels.

        The normalized image stays in at least fp32; the stem keeps its low bits.
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = wide(x)
        return self.net(((x + 1) / 2 - self.mean.to(x.dtype)) / self.std.to(x.dtype))

    def distance(self, fake: list[Tensor], real: list[Tensor]) -> Tensor:
        """The weighted sum of the configured distance over the feature layers."""
        parts = []
        for slot, weight in zip(self.slots, self.layer_weights):
            fake_map, real_map = fake[slot], real[slot]
            if self.use_gram:
                # Upstream divides its gram matrices by C * H * W, `feature_distance` by H * W
                weight = weight / fake_map.shape[1]
            fake_tokens = fake_map.flatten(2).transpose(1, 2)
            real_tokens = real_map.flatten(2).transpose(1, 2)
            value = feature_distance(
                fake_tokens, real_tokens, self.use_gram, self.use_norm, dim=-1
            )
            parts.append(weight * value)
        return torch.stack(parts).sum()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.distance(self.features(input), self.features(target))


@register("convnext", Range.SYMMETRIC, dims="2d")
class ConvNeXtLoss(ConvNeXtTerm):
    """Torchvision's ImageNet ConvNeXt (`tiny`, `small`, `base`, `large`) as a perceptual term."""

    def __init__(
        self,
        model: str = "small",
        feature_layers: Sequence[int] = (0, 2, 4, 8, 14),
        feature_weights: Sequence[float] | None = None,
        layer_weight_decay: float = 1.0,
        use_gram: bool = False,
        pretrained: bool = True,
    ):
        if model not in TORCHVISION:
            raise ValueError(f"ConvNeXt model {model!r} not in {list(TORCHVISION)}")
        super().__init__(
            TORCHVISION[model],
            f"convnext_{model}",
            feature_layers,
            feature_weights,
            layer_weight_decay,
            use_gram,
            use_norm=False,
            pretrained=pretrained,
        )


@register("dino_convnext", Range.SYMMETRIC, dims="2d")
class DinoV3ConvNeXtLoss(ConvNeXtTerm):
    """DINOv3 ConvNeXt (`tiny`, `small`, `base`, `large`) as a perceptual term.

    `use_norm` unit-scales each feature vector over channels before the distance.
    """

    def __init__(
        self,
        model: str = "small",
        feature_layers: Sequence[int] = (14,),
        feature_weights: Sequence[float] | None = None,
        layer_weight_decay: float = 1.0,
        use_gram: bool = False,
        use_norm: bool = True,
        pretrained: bool = True,
    ):
        if model not in DINOV3_CONVNEXT:
            raise ValueError(
                f"DINOv3 ConvNeXt model {model!r} not in {list(DINOV3_CONVNEXT)}"
            )
        source = f"dinov3_convnext_{model}"
        super().__init__(
            ConvNeXtConfig.of(config(source)),
            source,
            feature_layers,
            feature_weights,
            layer_weight_decay,
            use_gram,
            use_norm,
            pretrained,
        )
