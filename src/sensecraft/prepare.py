"""Put a `Loss` on a device: feature nets in the compute dtype and layout, compiled by scope.

| scope | compiled |
| --- | --- |
| `off` | nothing |
| `term` | each term's forward |
| `loss` | the whole loss forward |
"""

import torch
from torch import nn

from sensecraft.nets import FeatureNet

SCOPES = ("off", "term", "loss")


def prepare(
    loss: nn.Module,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    memory_format: torch.memory_format = torch.preserve_format,
    compile: str = "off",
    **compile_kwargs,
) -> nn.Module:
    """`loss` on `device`, its feature nets cast to `dtype` and `memory_format`, compiled by `compile`.

    Distances keep their own precision, so only the frozen nets change dtype.
    """
    if compile not in SCOPES:
        raise ValueError(f"compile scope {compile!r} not in {SCOPES}")
    loss.to(device)
    for module in loss.modules():
        if isinstance(module, FeatureNet):
            module.to(
                dtype=dtype or next(module.parameters()).dtype,
                memory_format=memory_format,
            )

    if compile == "term":
        for part in loss.parts.values():
            part.compile(**compile_kwargs)
    elif compile == "loss":
        loss.compile(**compile_kwargs)
    return loss
