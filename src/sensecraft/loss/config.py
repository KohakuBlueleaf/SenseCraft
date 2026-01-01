"""
Loss configuration classes and registry system.

This module provides:
1. Decorator-based loss registration
2. Typed config classes for complex losses
3. Simple {name: weight} dict shorthand support

Example:
    >>> from sensecraft.loss.config import (
    ...     GeneralConfig,
    ...     DinoV3LossConfig,
    ...     LPIPSConfig,
    ... )
    >>>
    >>> # Simple dict shorthand
    >>> config = [
    ...     {"charbonnier": 1.0},
    ...     {"sobel": 0.1},
    ... ]
    >>>
    >>> # Typed configs for complex losses
    >>> config = [
    ...     {"charbonnier": 1.0},
    ...     DinoV3LossConfig(weight=0.1, loss_layer=-6, use_gram=False),
    ...     LPIPSConfig(weight=0.05, net="alex"),
    ... ]
    >>>
    >>> # GeneralConfig for losses with extra kwargs
    >>> config = [
    ...     GeneralConfig("patch_fft", weight=0.1, patch_size=16),
    ... ]
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union


class ValueRange(Enum):
    """Required value range for different loss functions."""

    UNIT = "unit"  # [0, 1] - SSIM, MS-SSIM, edge losses
    SYMMETRIC = "symmetric"  # [-1, 1] - VGG/DINO based
    ANY = "any"  # No specific requirement


@dataclass
class LossInfo:
    """Metadata about a registered loss function."""

    loss_class: Type
    required_range: ValueRange
    is_2d_only: bool = False
    is_3d_only: bool = False


# Global registry
LOSS_REGISTRY: Dict[str, LossInfo] = {}


def register_loss(
    name: str,
    required_range: ValueRange = ValueRange.ANY,
    is_2d_only: bool = False,
    is_3d_only: bool = False,
) -> Callable[[Type], Type]:
    """Decorator to register a loss class.

    Args:
        name: Name to register the loss under
        required_range: Required input value range
        is_2d_only: If True, only works with 2D input (applied per-frame in 3D)
        is_3d_only: If True, only works with 3D input

    Example:
        >>> @register_loss("my_loss", ValueRange.UNIT, is_2d_only=True)
        ... class MyLoss(nn.Module):
        ...     def __init__(self, param=1.0):
        ...         ...
    """

    def decorator(cls: Type) -> Type:
        LOSS_REGISTRY[name] = LossInfo(
            loss_class=cls,
            required_range=required_range,
            is_2d_only=is_2d_only,
            is_3d_only=is_3d_only,
        )
        # Store the registered name on the class for reference
        cls._registered_name = name
        return cls

    return decorator


def get_registered_loss(name: str) -> Optional[LossInfo]:
    """Get loss info by name."""
    return LOSS_REGISTRY.get(name)


def list_registered_losses() -> List[str]:
    """List all registered loss names."""
    return list(LOSS_REGISTRY.keys())


# =============================================================================
# Config Classes
# =============================================================================


@dataclass
class LossConfig:
    """Base class for loss configurations."""

    weight: float = 1.0

    def get_name(self) -> str:
        """Get the loss name for registry lookup."""
        raise NotImplementedError

    def get_kwargs(self) -> Dict[str, Any]:
        """Get kwargs to pass to loss constructor."""
        raise NotImplementedError


@dataclass
class GeneralConfig(LossConfig):
    """General config for simple losses with optional kwargs.

    Use this for losses that just need a name and optional parameters.

    Example:
        >>> GeneralConfig("charbonnier", weight=1.0)
        >>> GeneralConfig("patch_fft", weight=0.1, patch_size=16)
    """

    name: str = ""
    weight: float = 1.0
    # Extra kwargs stored in a dict
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str = "", weight: float = 1.0, **kwargs):
        self.name = name
        self.weight = weight
        self.kwargs = kwargs

    def get_name(self) -> str:
        return self.name

    def get_kwargs(self) -> Dict[str, Any]:
        return self.kwargs


@dataclass
class DinoV3LossConfig(LossConfig):
    """Config for ViT DINOv3 perceptual loss.

    Args:
        weight: Loss weight
        model_type: Model variant ("small", "small_plus", "base", "large")
        loss_layer: Layer index for feature extraction (negative = from end)
        use_gram: If True, use Gram matrix loss. If False, use MSE.
        use_norm: If True, L2 normalize features before loss
        input_range: Expected input value range
    """

    weight: float = 1.0
    model_type: str = "small_plus"
    loss_layer: int = -4
    use_gram: bool = True
    use_norm: bool = True
    input_range: tuple = (-1, 1)

    def get_name(self) -> str:
        return "dino_vit"

    def get_kwargs(self) -> Dict[str, Any]:
        from .gram_dinov3 import ModelType

        # Map string to enum
        model_map = {
            "small": ModelType.SMALL,
            "small_plus": ModelType.SMALL_PLUS,
            "base": ModelType.BASE,
            "large": ModelType.LARGE,
        }
        return {
            "model_type": model_map.get(self.model_type, ModelType.SMALL_PLUS),
            "loss_layer": self.loss_layer,
            "use_gram": self.use_gram,
            "use_norm": self.use_norm,
            "input_range": self.input_range,
        }


@dataclass
class ConvNextDinoV3LossConfig(LossConfig):
    """Config for ConvNext DINOv3 perceptual loss.

    Args:
        weight: Loss weight
        model_type: Model variant ("atto", "femto", "pico", "nano", "tiny",
                    "small", "base", "large", "huge")
        use_gram: If True, use Gram matrix loss
        use_norm: If True, L2 normalize features
        loss_layer: Layer index for single-layer mode (None = multi-layer)
        layer_weights: Weights for multi-layer mode
        input_range: Expected input value range
    """

    weight: float = 1.0
    model_type: str = "pico"
    use_gram: bool = True
    use_norm: bool = True
    loss_layer: Optional[int] = None
    layer_weights: Optional[List[float]] = None
    input_range: tuple = (-1, 1)

    def get_name(self) -> str:
        return "dino_convnext"

    def get_kwargs(self) -> Dict[str, Any]:
        from .convnext_dinov3 import ModelType

        model_map = {
            "atto": ModelType.ATTO,
            "femto": ModelType.FEMTO,
            "pico": ModelType.PICO,
            "nano": ModelType.NANO,
            "tiny": ModelType.TINY,
            "small": ModelType.SMALL,
            "base": ModelType.BASE,
            "large": ModelType.LARGE,
            "huge": ModelType.HUGE,
        }
        kwargs = {
            "model_type": model_map.get(self.model_type, ModelType.PICO),
            "use_gram": self.use_gram,
            "use_norm": self.use_norm,
            "input_range": self.input_range,
        }
        if self.loss_layer is not None:
            kwargs["loss_layer"] = self.loss_layer
        if self.layer_weights is not None:
            kwargs["layer_weights"] = self.layer_weights
        return kwargs


@dataclass
class LPIPSConfig(LossConfig):
    """Config for LPIPS perceptual loss.

    Args:
        weight: Loss weight
        net: Network backbone ("vgg", "alex", "squeeze")
    """

    weight: float = 1.0
    net: str = "vgg"

    def get_name(self) -> str:
        return "lpips"

    def get_kwargs(self) -> Dict[str, Any]:
        return {"net": self.net}


@dataclass
class SSIMConfig(LossConfig):
    """Config for SSIM loss.

    Args:
        weight: Loss weight
        win_size: Window size
        win_sigma: Gaussian sigma
        data_range: Data range (set automatically by SenseCraftLoss)
    """

    weight: float = 1.0
    win_size: int = 11
    win_sigma: float = 1.5

    def get_name(self) -> str:
        return "ssim"

    def get_kwargs(self) -> Dict[str, Any]:
        return {
            "win_size": self.win_size,
            "win_sigma": self.win_sigma,
        }


@dataclass
class MSSSIMConfig(LossConfig):
    """Config for MS-SSIM loss.

    Args:
        weight: Loss weight
        win_size: Window size
        win_sigma: Gaussian sigma
        weights: Scale weights (default: 5 scales)
    """

    weight: float = 1.0
    win_size: int = 11
    win_sigma: float = 1.5
    weights: Optional[List[float]] = None

    def get_name(self) -> str:
        return "ms_ssim"

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "win_size": self.win_size,
            "win_sigma": self.win_sigma,
        }
        if self.weights is not None:
            kwargs["weights"] = self.weights
        return kwargs


@dataclass
class PatchFFTConfig(LossConfig):
    """Config for Patch FFT loss.

    Args:
        weight: Loss weight
        patch_size: Size of patches (e.g., 8 for 8x8)
        loss_type: "mse", "l1", or "charbonnier"
        norm_type: Normalization ("none", "l2", "log", "log1p")
        use_phase: Whether to include phase loss
        phase_weight: Weight for phase component
    """

    weight: float = 1.0
    patch_size: int = 8
    loss_type: str = "mse"
    norm_type: str = "log1p"
    use_phase: bool = False
    phase_weight: float = 0.1

    def get_name(self) -> str:
        return "patch_fft"

    def get_kwargs(self) -> Dict[str, Any]:
        from .general import NormType

        norm_map = {
            "none": NormType.NONE,
            "l2": NormType.L2,
            "log": NormType.LOG,
            "log1p": NormType.LOG1P,
        }
        return {
            "patch_size": self.patch_size,
            "loss_type": self.loss_type,
            "norm_type": norm_map.get(self.norm_type, NormType.LOG1P),
            "use_phase": self.use_phase,
            "phase_weight": self.phase_weight,
        }


# Type alias for loss config input
LossConfigInput = Union[Dict[str, float], LossConfig]


def parse_loss_config(config: LossConfigInput) -> tuple[str, float, Dict[str, Any]]:
    """Parse a loss config into (name, weight, kwargs).

    Args:
        config: Either a {name: weight} dict or a LossConfig instance

    Returns:
        Tuple of (loss_name, weight, kwargs)
    """
    if isinstance(config, LossConfig):
        return config.get_name(), config.weight, config.get_kwargs()
    elif isinstance(config, dict):
        # Simple {name: weight} format
        if len(config) != 1:
            raise ValueError(
                f"Dict config must have exactly one key (name), got: {config}"
            )
        name, weight = next(iter(config.items()))
        return name, weight, {}
    else:
        raise TypeError(f"Invalid config type: {type(config)}")
