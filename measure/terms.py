"""The perceptual terms the net stages cover: our build and the upstream build of each."""

# key: (kind, keyword arguments shared by both builds)
TERMS: dict[str, tuple[str, dict]] = {
    "lpips_vgg": ("lpips", {"net": "vgg"}),
    "lpips_alex": ("lpips", {"net": "alex"}),
    "lpips_squeeze": ("lpips", {"net": "squeeze"}),
    "dino_base_mse": (
        "dino",
        {"model": "base", "loss_layer": -4, "use_gram": False, "use_norm": True},
    ),
    "dino_base_gram": (
        "dino",
        {"model": "base", "loss_layer": -4, "use_gram": True, "use_norm": True},
    ),
    "convnext_small": (
        "convnext",
        {"model": "small", "feature_layers": [0, 2, 4, 8, 14], "use_gram": False},
    ),
    "convnext_small_gram": (
        "convnext",
        {"model": "small", "feature_layers": [0, 2, 4, 8, 14], "use_gram": True},
    ),
    # DINOv3 ConvNeXt repos are gated: recorded on the random-weight stand-in in refs/standin_hub
    "dino_convnext_tiny": (
        "dino_convnext",
        {"model": "tiny", "feature_layers": [14], "use_gram": False, "use_norm": True},
    ),
    "dino_convnext_tiny_gram": (
        "dino_convnext",
        {
            "model": "tiny",
            "feature_layers": [14, 2, 8, -1],
            "use_gram": True,
            "use_norm": True,
        },
    ),
}


def ours(kind: str, kwargs: dict):
    """Our term, and the function from an image to its features."""
    if kind == "lpips":
        from sensecraft.perceptual import LPIPS

        term = LPIPS(**kwargs)
    elif kind == "convnext":
        from sensecraft.perceptual import ConvNeXtLoss

        term = ConvNeXtLoss(**kwargs)
    elif kind == "dino_convnext":
        from sensecraft.perceptual import DinoV3ConvNeXtLoss

        term = DinoV3ConvNeXtLoss(**kwargs)
    else:
        from sensecraft.perceptual import DinoV3ViTLoss

        term = DinoV3ViTLoss(**kwargs)
    return term, term.features


def upstream(kind: str, kwargs: dict):
    """The 0.3.13 term, its feature function, and its net's state dict."""
    if kind == "lpips":
        from sensecraft.loss.lpips import LPIPS

        term = LPIPS(net=kwargs["net"]).eval().requires_grad_(False)
        return term, lambda x: list(term.net(term.scaling_layer(x))), term.net
    if kind in ("convnext", "dino_convnext"):
        return upstream_convnext(kind, kwargs)
    from sensecraft.loss.gram_dinov3 import ModelType, ViTDinoV3PerceptualLoss

    model_type = {
        "small": "SMALL",
        "small_plus": "SMALL_PLUS",
        "base": "BASE",
        "large": "LARGE",
    }
    term = (
        ViTDinoV3PerceptualLoss(
            model_type=ModelType[model_type[kwargs["model"]]],
            loss_layer=kwargs["loss_layer"],
            use_gram=kwargs["use_gram"],
            use_norm=kwargs["use_norm"],
        )
        .eval()
        .requires_grad_(False)
    )
    return term, lambda x: [term.dinov3_fwd(term.normalize_input(x))], term.model


def upstream_convnext(kind: str, kwargs: dict):
    """The 0.3.13 ConvNeXt term, its blocks run up to the deepest tap, and its blocks."""
    if kind == "convnext":
        from sensecraft.loss.convnext import ConvNextPerceptualLoss, ConvNextType

        term = ConvNextPerceptualLoss(
            model_type=ConvNextType[kwargs["model"].upper()],
            feature_layers=kwargs["feature_layers"],
            use_gram=kwargs["use_gram"],
        )
    else:
        from sensecraft.loss.convnext_dinov3 import (
            ConvNextDinoV3PerceptualLoss,
            ConvNextType,
        )

        # Multi-layer mode: `loss_layer` must be None or it overrides `feature_layers`
        term = ConvNextDinoV3PerceptualLoss(
            model_type=ConvNextType[kwargs["model"].upper()],
            feature_layers=kwargs["feature_layers"],
            use_gram=kwargs["use_gram"],
            use_norm=kwargs["use_norm"],
            loss_layer=None,
        )
    term = term.eval().requires_grad_(False)
    taps = sorted({layer % len(term.blocks) for layer in kwargs["feature_layers"]})

    def features_of(x):
        x = term.normalize_input(x)
        features = []
        for index, block in enumerate(term.blocks[: taps[-1] + 1]):
            x = block(x)
            if index in taps:
                features.append(x)
        return features

    return term, features_of, term.blocks
