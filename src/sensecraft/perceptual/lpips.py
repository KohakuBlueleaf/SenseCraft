"""LPIPS: a VGG16, AlexNet or SqueezeNet with the learned v0.1 heads."""

from torch import Tensor, nn

from sensecraft.distance import lpips_distance
from sensecraft.nets import alexnet, squeezenet, vgg16
from sensecraft.registry import Range, register
from sensecraft.weights import packaged, weights

# net name: (builder, weight source)
NETS = {
    "vgg": (vgg16, "vgg16"),
    "alex": (alexnet, "alexnet"),
    "squeeze": (squeezenet, "squeezenet"),
}
SHIFT = (-0.030, -0.088, -0.188)
SCALE = (0.458, 0.448, 0.450)


@register("lpips", Range.SYMMETRIC, dims="2d")
class LPIPS(nn.Module):
    """LPIPS between `input` and `target`, both in [-1, 1]."""

    def __init__(self, net: str = "vgg", pretrained: bool = True):
        super().__init__()
        if net not in NETS:
            raise ValueError(f"LPIPS net {net!r} not in {list(NETS)}")
        build, source = NETS[net]
        self.backbone = build()
        heads = packaged(f"lpips_{net}")
        if pretrained:
            self.backbone.load_state_dict(weights(source))
        self.backbone.freeze()

        # Learned per-tap heads and the input scaling, as buffers so they follow `.to`
        for index in range(len(heads)):
            self.register_buffer(f"head{index}", heads[f"lins.{index}"])
        self.register_buffer("shift", Tensor(SHIFT).view(1, 3, 1, 1))
        self.register_buffer("scale", Tensor(SCALE).view(1, 3, 1, 1))
        self.taps = len(heads)

    def heads(self) -> list[Tensor]:
        return [getattr(self, f"head{index}") for index in range(self.taps)]

    def features(self, x: Tensor) -> list[Tensor]:
        """Backbone features of an image in [-1, 1]."""
        return self.backbone(((x - self.shift) / self.scale).to(self.backbone.dtype))

    def distance(self, fake: list[Tensor], real: list[Tensor]) -> Tensor:
        """LPIPS between two feature lists of this backbone."""
        return lpips_distance(fake, real, self.heads())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.distance(self.features(input), self.features(target))
