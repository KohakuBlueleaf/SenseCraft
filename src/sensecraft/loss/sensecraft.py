"""
SenseCraftLoss: Configurable compound loss for 2D images and 3D video.

This module provides a unified wrapper for combining multiple loss functions
with automatic value range handling and 2D/3D mode support.

Example:
    >>> from sensecraft.loss import SenseCraftLoss
    >>> from sensecraft.loss.config import DinoV3LossConfig, LPIPSConfig
    >>>
    >>> # Simple dict shorthand: {name: weight}
    >>> loss_fn = SenseCraftLoss(
    ...     loss_config=[
    ...         {"charbonnier": 1.0},
    ...         {"sobel": 0.1},
    ...     ],
    ...     input_range=(-1, 1),
    ...     mode="2d",
    ... )
    >>>
    >>> # Typed configs for complex losses
    >>> loss_fn = SenseCraftLoss(
    ...     loss_config=[
    ...         {"charbonnier": 1.0},
    ...         DinoV3LossConfig(weight=0.1, loss_layer=-6, use_gram=False),
    ...         LPIPSConfig(weight=0.05, net="alex"),
    ...     ],
    ...     input_range=(-1, 1),
    ... )
    >>>
    >>> # Returns dict with individual losses and total
    >>> losses = loss_fn(pred, target)
    >>> total_loss = losses["loss"]
    >>> lpips_value = losses["lpips"]
"""

from typing import Any, Dict, List, Literal, Tuple, Union

import torch
import torch.nn as nn

from .config import (
    LOSS_REGISTRY,
    ValueRange,
    LossConfig,
    LossConfigInput,
    parse_loss_config,
    list_registered_losses,
)


class SenseCraftLoss(nn.Module):
    """Configurable compound loss for 2D images and 3D video.

    Supports automatic value range adjustment and 2D/3D mode handling.
    In 3D mode, 2D-only losses are applied frame-by-frame.

    Args:
        loss_config: List of loss configurations. Each item can be:
            - Dict {name: weight}: e.g., {"charbonnier": 1.0}
            - LossConfig subclass: e.g., DinoV3LossConfig(weight=0.1, ...)
        input_range: Tuple of (min, max) for input values (default: (-1, 1))
        mode: "2d" for images (B, C, H, W), "3d" for video (B, T, C, H, W)
        return_dict: If True, return dict with all losses. If False, return only total.

    Example:
        >>> loss_fn = SenseCraftLoss(
        ...     loss_config=[
        ...         {"charbonnier": 1.0},
        ...         {"lpips": 0.1},
        ...         {"ssim": 0.0},  # weight=0 -> computed under no_grad for monitoring
        ...     ],
        ...     input_range=(-1, 1),
        ...     mode="2d",
        ... )
        >>> losses = loss_fn(pred, target)
        >>> losses["loss"]  # Total weighted loss
        >>> losses["charbonnier"]  # Individual loss value
    """

    def __init__(
        self,
        loss_config: List[LossConfigInput],
        input_range: Tuple[float, float] = (-1, 1),
        mode: Literal["2d", "3d"] = "2d",
        return_dict: bool = True,
    ):
        super().__init__()
        self.input_range = input_range
        self.mode = mode
        self.return_dict = return_dict

        # Parse and validate loss config
        self.loss_names: List[str] = []
        self.loss_weights: Dict[str, float] = {}
        self.loss_modules: nn.ModuleDict = nn.ModuleDict()

        # Group losses by required value range
        self.unit_range_losses: List[str] = []  # Need [0, 1]
        self.symmetric_range_losses: List[str] = []  # Need [-1, 1]
        self.any_range_losses: List[str] = []  # Don't care

        # Track 2D-only losses for 3D mode handling
        self.is_2d_only: Dict[str, bool] = {}

        for config in loss_config:
            name, weight, kwargs = parse_loss_config(config)

            if name not in LOSS_REGISTRY:
                raise ValueError(
                    f"Unknown loss: {name}. Available: {list_registered_losses()}"
                )

            info = LOSS_REGISTRY[name]

            # Check mode compatibility
            if mode == "2d" and info.is_3d_only:
                raise ValueError(f"Loss '{name}' is 3D-only but mode is '2d'")

            # Create loss instance
            try:
                loss_module = info.loss_class(**kwargs)
            except TypeError as e:
                raise ValueError(f"Error creating loss '{name}': {e}")

            self.loss_names.append(name)
            self.loss_weights[name] = weight
            self.loss_modules[name] = loss_module
            self.is_2d_only[name] = info.is_2d_only

            # Categorize by value range
            if info.required_range == ValueRange.UNIT:
                self.unit_range_losses.append(name)
            elif info.required_range == ValueRange.SYMMETRIC:
                self.symmetric_range_losses.append(name)
            else:
                self.any_range_losses.append(name)

    def _convert_to_unit_range(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from input_range to [0, 1]."""
        min_val, max_val = self.input_range
        return (x - min_val) / (max_val - min_val)

    def _convert_to_symmetric_range(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from input_range to [-1, 1]."""
        min_val, max_val = self.input_range
        # First to [0, 1], then to [-1, 1]
        x_unit = (x - min_val) / (max_val - min_val)
        return x_unit * 2 - 1

    def _compute_single_loss(
        self,
        name: str,
        input: torch.Tensor,
        target: torch.Tensor,
        input_unit: torch.Tensor,
        target_unit: torch.Tensor,
        input_symmetric: torch.Tensor,
        target_symmetric: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a single loss with appropriate value range."""
        loss_fn = self.loss_modules[name]

        # Select appropriate input based on required range
        if name in self.unit_range_losses:
            inp, tgt = input_unit, target_unit
        elif name in self.symmetric_range_losses:
            inp, tgt = input_symmetric, target_symmetric
        else:
            inp, tgt = input, target

        # Handle 2D-only losses in 3D mode
        if self.mode == "3d" and self.is_2d_only[name]:
            B, T, C, H, W = inp.shape
            inp_2d = inp.reshape(B * T, C, H, W)
            tgt_2d = tgt.reshape(B * T, C, H, W)
            loss_value = loss_fn(inp_2d, tgt_2d)
        else:
            loss_value = loss_fn(inp, tgt)

        return loss_value

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute compound loss.

        Args:
            input: Predicted tensor
                - 2D mode: (B, C, H, W)
                - 3D mode: (B, T, C, H, W)
            target: Target tensor (same shape as input)

        Returns:
            If return_dict=True: Dict with "loss" (total) and individual losses
            If return_dict=False: Total weighted loss tensor
        """
        device = input.device
        dtype = input.dtype

        # Precompute converted ranges if needed
        input_unit = None
        target_unit = None
        input_symmetric = None
        target_symmetric = None

        if self.unit_range_losses:
            input_unit = self._convert_to_unit_range(input)
            target_unit = self._convert_to_unit_range(target)

        if self.symmetric_range_losses:
            input_symmetric = self._convert_to_symmetric_range(input)
            target_symmetric = self._convert_to_symmetric_range(target)

        # Compute all losses
        loss_dict: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for name in self.loss_names:
            weight = self.loss_weights[name]

            if weight == 0:
                # Compute without gradients for monitoring
                with torch.no_grad():
                    loss_value = self._compute_single_loss(
                        name,
                        input,
                        target,
                        input_unit,
                        target_unit,
                        input_symmetric,
                        target_symmetric,
                    )
            else:
                loss_value = self._compute_single_loss(
                    name,
                    input,
                    target,
                    input_unit,
                    target_unit,
                    input_symmetric,
                    target_symmetric,
                )
                total_loss = total_loss + weight * loss_value

            loss_dict[name] = loss_value

        loss_dict["loss"] = total_loss

        if self.return_dict:
            return loss_dict
        else:
            return total_loss

    @staticmethod
    def available_losses() -> List[str]:
        """Return list of available loss names."""
        return list_registered_losses()

    @staticmethod
    def loss_info_str() -> str:
        """Return formatted string describing available losses."""
        lines = ["Available losses:", ""]
        for name, info in LOSS_REGISTRY.items():
            range_str = info.required_range.value
            mode_str = ""
            if info.is_2d_only:
                mode_str = " (2D only)"
            elif info.is_3d_only:
                mode_str = " (3D only)"
            lines.append(f"  {name}: range={range_str}{mode_str}")
        return "\n".join(lines)
