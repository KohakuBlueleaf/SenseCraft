"""DINOv3 ViT features at one layer, compared by MSE or by gram matrices."""

import re

from torch import Tensor, nn

from sensecraft.distance import feature_distance
from sensecraft.nets.dinov3_vit import DinoV3ViT, ViTConfig
from sensecraft.registry import Range, register
from sensecraft.weights import DINOV3_VIT, config, weights

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def runs_up_to(key: str, layer: int) -> bool:
    """Whether a ViT weight key is used by a net cut after `layer`."""
    if key.startswith("embeddings."):
        return True
    found = re.match(r"layer\.(\d+)\.", key)
    return found is not None and int(found.group(1)) <= layer


@register("dino_vit", Range.SYMMETRIC, dims="2d")
class DinoV3ViTLoss(nn.Module):
    """Distance between DINOv3 ViT tokens of `input` and `target` at `loss_layer`.

    `loss_layer` counts from 0, and from the end when negative. Tokens include the class and
    register tokens.
    """

    def __init__(
        self,
        model: str = "small_plus",
        loss_layer: int = -4,
        use_gram: bool = True,
        use_norm: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        if model not in DINOV3_VIT:
            raise ValueError(f"DINOv3 ViT model {model!r} not in {list(DINOV3_VIT)}")
        source = f"dinov3_vit_{model}"
        vit = ViTConfig.of(config(source))
        depth = vit.num_hidden_layers
        layer = loss_layer + depth if loss_layer < 0 else loss_layer
        if not 0 <= layer < depth:
            raise ValueError(f"loss_layer {loss_layer} outside a {depth}-layer model")

        self.net = DinoV3ViT(vit, [layer])
        if pretrained:
            # The final norm and the layers past the tap are never run
            state = {k: v for k, v in weights(source).items() if runs_up_to(k, layer)}
            self.net.load_state_dict(state)
        self.net.freeze()
        self.use_gram, self.use_norm = use_gram, use_norm
        self.register_buffer("mean", Tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", Tensor(STD).view(1, 3, 1, 1))

    def features(self, x: Tensor) -> list[Tensor]:
        """Tokens of an image in [-1, 1]; a gray image is repeated to three channels."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.net((((x + 1) / 2 - self.mean) / self.std).to(self.net.dtype))

    def distance(self, fake: list[Tensor], real: list[Tensor]) -> Tensor:
        """The configured distance between two token lists of this net."""
        return feature_distance(fake[0], real[0], self.use_gram, self.use_norm, dim=-1)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.distance(self.features(input), self.features(target))
