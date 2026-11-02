"""Pretrained weights: each net's upstream source, converted once to a local safetensors file.

The upstream file is read only on the first use, with `torch.hub` or `huggingface_hub`; later
loads read the converted file. `SENSECRAFT_CACHE` moves the cache.
"""

import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from sensecraft.nets.convnext import TORCHVISION

PACKAGED = Path(__file__).parent


@dataclass(frozen=True)
class Source:
    """Where a net's weights come from, and how upstream keys map to ours."""

    url: str = ""  # a torch.hub URL
    repo: str = ""  # a Hugging Face repo holding `model.safetensors`
    rename: Callable[[str], str | None] = lambda key: key  # None drops the key
    # Reshapes a tensor, given its new key, to our layout
    reshape: Callable[[str, Tensor], Tensor] = lambda key, value: value


def _prefix(old: str, new: str) -> Callable[[str], str | None]:
    """Keep keys under `old`, moved under `new`."""
    return lambda key: new + key[len(old) :] if key.startswith(old) else None


SOURCES: dict[str, Source] = {
    "vgg16": Source(
        url="https://download.pytorch.org/models/vgg16-397923af.pth",
        rename=_prefix("features.", "layers."),
    ),
    "alexnet": Source(
        url="https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        rename=_prefix("features.", "layers."),
    ),
    "squeezenet": Source(
        url="https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
        rename=_prefix("features.", "layers."),
    ),
}

# DINOv3 ViTs on the Hugging Face hub; their repos are gated, so the first use needs a token
DINOV3_VIT = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "small_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
for _size, _repo in DINOV3_VIT.items():
    SOURCES[f"dinov3_vit_{_size}"] = Source(
        repo=_repo, rename=lambda key: None if key.endswith("mask_token") else key
    )


# Torchvision ConvNeXt block layers under the DINOv3 names our block uses
TORCHVISION_BLOCK = {
    "block.0": "depthwise_conv",
    "block.2": "layer_norm",
    "block.3": "pointwise_conv1",
    "block.5": "pointwise_conv2",
    "layer_scale": "gamma",
}


def _torchvision_convnext(depths: list[int]) -> Callable[[str], str | None]:
    """Torchvision `features.<group>.<index>` keys onto our flat layers; the head is dropped.

    Groups alternate between a stem or downsample (two layers) and a stage's blocks.
    """
    counts = [count for depth in depths for count in (2, depth)]
    group_starts = [sum(counts[:group]) for group in range(len(counts))]

    def rename(key: str) -> str | None:
        found = re.fullmatch(r"features\.(\d+)\.(\d+)\.(.+)", key)
        if found is None:
            return None
        group, index, rest = found.groups()
        for old, new in TORCHVISION_BLOCK.items():
            if rest == old or rest.startswith(old + "."):
                rest = new + rest[len(old) :]
        return f"layers.{group_starts[int(group)] + int(index)}.{rest}"

    return rename


# Torchvision ImageNet ConvNeXts: the hash part of each checkpoint's file name
TORCHVISION_CONVNEXT = {
    "tiny": "983f1562",
    "small": "0c510722",
    "base": "6075fbad",
    "large": "ea097f82",
}
for _size, _hash in TORCHVISION_CONVNEXT.items():
    SOURCES[f"convnext_{_size}"] = Source(
        url=f"https://download.pytorch.org/models/convnext_{_size}-{_hash}.pth",
        rename=_torchvision_convnext(TORCHVISION[_size].depths),
        # Torchvision holds the layer scale as `(C, 1, 1)`, our block as `(C,)`
        reshape=lambda key, value: value.flatten() if key.endswith("gamma") else value,
    )


def _dinov3_convnext(name: str) -> Callable[[str], str | None]:
    """DINOv3 `stages.<stage>.{downsample_layers,layers}.<index>` keys onto our flat layers.

    Stage depths come from the checkpoint's `config.json`; the final norm is dropped.
    """

    def rename(key: str) -> str | None:
        found = re.fullmatch(
            r"stages\.(\d+)\.(downsample_layers|layers)\.(\d+)\.(.+)", key
        )
        if found is None:
            return None
        stage, part, index, rest = found.groups()
        depths = config(name)["depths"]
        start = sum(depths[: int(stage)]) + 2 * int(stage)
        offset = 0 if part == "downsample_layers" else 2
        return f"layers.{start + offset + int(index)}.{rest}"

    return rename


# DINOv3 ConvNeXts on the Hugging Face hub, gated like the ViTs
DINOV3_CONVNEXT = {
    "tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}
for _size, _repo in DINOV3_CONVNEXT.items():
    SOURCES[f"dinov3_convnext_{_size}"] = Source(
        repo=_repo, rename=_dinov3_convnext(f"dinov3_convnext_{_size}")
    )


def cache_dir() -> Path:
    """Root of converted weights: `SENSECRAFT_CACHE`, else under the user cache."""
    root = os.environ.get("SENSECRAFT_CACHE")
    if root:
        return Path(root)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "sensecraft"


def upstream(name: str) -> dict[str, Tensor]:
    """The upstream state dict of `name`, downloaded if needed."""
    source = SOURCES[name]
    if source.url:
        return torch.hub.load_state_dict_from_url(
            source.url, map_location="cpu", weights_only=True
        )
    from huggingface_hub import hf_hub_download  # optional: only DINOv3 needs it

    return load_file(hf_hub_download(source.repo, "model.safetensors"))


def convert(name: str) -> dict[str, Tensor]:
    """Upstream weights of `name` under our keys."""
    source = SOURCES[name]
    renamed = {source.rename(key): value for key, value in upstream(name).items()}
    return {
        key: source.reshape(key, value).contiguous()
        for key, value in renamed.items()
        if key is not None
    }


def weights(name: str) -> dict[str, Tensor]:
    """Converted weights of `name`, from the cache or converted now."""
    path = cache_dir() / f"{name}.safetensors"
    if not path.exists():
        # Each writer has its own temporary file; the last rename wins with identical bytes
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, partial = tempfile.mkstemp(dir=path.parent, suffix=".partial")
        os.close(handle)
        save_file(convert(name), partial)
        os.replace(partial, path)
    return load_file(str(path))


def config(name: str) -> dict:
    """The `config.json` of a hub-sourced net, cached beside its weights."""
    path = cache_dir() / f"{name}.json"
    if not path.exists():
        from huggingface_hub import hf_hub_download  # optional: only DINOv3 needs it

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            Path(hf_hub_download(SOURCES[name].repo, "config.json")).read_text()
        )
    return json.loads(path.read_text())


def packaged(name: str) -> dict[str, Tensor]:
    """Weights shipped inside the package, such as the LPIPS heads."""
    return load_file(str(PACKAGED / f"{name}.safetensors"))
