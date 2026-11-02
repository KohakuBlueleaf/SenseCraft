"""DINOv3 ViT: patch tokens with a class token and registers, rope attention, layer scale.

Built from the checkpoint's own `config.json`, and only up to the deepest tapped layer.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sensecraft.nets.base import FeatureNet


@dataclass(frozen=True)
class ViTConfig:
    """The fields of a DINOv3 ViT `config.json` the net needs."""

    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    use_gated_mlp: bool
    hidden_act: str
    num_register_tokens: int
    patch_size: int
    layer_norm_eps: float
    rope_theta: float
    query_bias: bool
    key_bias: bool
    value_bias: bool
    proj_bias: bool
    mlp_bias: bool

    @classmethod
    def of(cls, config: dict) -> "ViTConfig":
        """The config from a parsed `config.json`."""
        return cls(**{name: config[name] for name in cls.__dataclass_fields__})


ACTIVATIONS = {"gelu": F.gelu, "silu": F.silu}


def rotate_half(x: Tensor) -> Tensor:
    """The second half of the last dim, negated, before the first half."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class LayerScale(nn.Module):
    """A learned per-channel scale."""

    def __init__(self, size: int):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(size))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.lambda1


class Attention(nn.Module):
    """Multi-head self-attention with rope on the patch tokens only."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        size = config.hidden_size
        self.heads = config.num_attention_heads
        self.q_proj = nn.Linear(size, size, bias=config.query_bias)
        self.k_proj = nn.Linear(size, size, bias=config.key_bias)
        self.v_proj = nn.Linear(size, size, bias=config.value_bias)
        self.o_proj = nn.Linear(size, size, bias=config.proj_bias)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, tokens, size = x.shape

        def split(projected: Tensor) -> Tensor:
            return projected.view(batch, tokens, self.heads, -1).transpose(1, 2)

        query, key, value = (
            split(self.q_proj(x)),
            split(self.k_proj(x)),
            split(self.v_proj(x)),
        )

        # Rope on the patch tokens; the class and register tokens come first and stay as they are
        prefix = tokens - cos.shape[-2]
        query = torch.cat(
            (query[..., :prefix, :], self._rotate(query[..., prefix:, :], cos, sin)),
            dim=-2,
        )
        key = torch.cat(
            (key[..., :prefix, :], self._rotate(key[..., prefix:, :], cos, sin)), dim=-2
        )

        out = F.scaled_dot_product_attention(query, key, value)
        return self.o_proj(out.transpose(1, 2).reshape(batch, tokens, size))

    @staticmethod
    def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        return x * cos + rotate_half(x) * sin


class MLP(nn.Module):
    """Up, activation, down; gated when the config says so."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        size, inner, bias = (
            config.hidden_size,
            config.intermediate_size,
            config.mlp_bias,
        )
        self.gated = config.use_gated_mlp
        if self.gated:
            self.gate_proj = nn.Linear(size, inner, bias=bias)
        self.up_proj = nn.Linear(size, inner, bias=bias)
        self.down_proj = nn.Linear(inner, size, bias=bias)
        self.act = ACTIVATIONS[config.hidden_act]

    def forward(self, x: Tensor) -> Tensor:
        if self.gated:
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act(self.up_proj(x)))


class Layer(nn.Module):
    """Pre-norm attention and MLP, each scaled before its residual add."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        size, eps = config.hidden_size, config.layer_norm_eps
        self.norm1 = nn.LayerNorm(size, eps=eps)
        self.attention = Attention(config)
        self.layer_scale1 = LayerScale(size)
        self.norm2 = nn.LayerNorm(size, eps=eps)
        self.mlp = MLP(config)
        self.layer_scale2 = LayerScale(size)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.layer_scale1(self.attention(self.norm1(x), cos, sin))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class Embeddings(nn.Module):
    """Patch embedding with the class and register tokens in front."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        size = config.hidden_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, size))
        self.register_tokens = nn.Parameter(
            torch.zeros(1, config.num_register_tokens, size)
        )
        self.patch_embeddings = nn.Conv2d(3, size, config.patch_size, config.patch_size)

    def forward(self, pixels: Tensor) -> Tensor:
        patches = self.patch_embeddings(pixels).flatten(2).transpose(1, 2)
        batch = patches.shape[0]
        prefix = [
            self.cls_token.expand(batch, -1, -1),
            self.register_tokens.expand(batch, -1, -1),
        ]
        return torch.cat([*prefix, patches], dim=1)


class DinoV3ViT(FeatureNet):
    """Token features after each layer in `taps` (0 is the first layer), shallow first."""

    def __init__(self, config: ViTConfig, taps: list[int]):
        super().__init__()
        self.config = config
        self.taps = set(taps)
        self.embeddings = Embeddings(config)
        self.layer = nn.ModuleList(Layer(config) for _ in range(max(taps) + 1))
        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1 / config.rope_theta ** torch.arange(
            0, 1, 4 / head_dim, dtype=torch.float32
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def positions(
        self, height: int, width: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """Rope cosine and sine per patch, from patch centers in [-1, 1]."""
        device = self.inv_freq.device
        rows = torch.arange(0.5, height, dtype=torch.float32, device=device) / height
        cols = torch.arange(0.5, width, dtype=torch.float32, device=device) / width
        centers = torch.stack(
            torch.meshgrid(rows, cols, indexing="ij"), dim=-1
        ).flatten(0, 1)
        angles = (
            2 * math.pi * (2 * centers - 1)[:, :, None] * self.inv_freq[None, None, :]
        )
        angles = angles.flatten(1, 2).tile(2)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def forward(self, pixels: Tensor) -> list[Tensor]:
        patch = self.config.patch_size
        cos, sin = self.positions(
            pixels.shape[-2] // patch, pixels.shape[-1] // patch, pixels.dtype
        )
        x = self.embeddings(pixels)
        features = []
        for index, layer in enumerate(self.layer):
            x = layer(x, cos, sin)
            if index in self.taps:
                features.append(x)
        return features
