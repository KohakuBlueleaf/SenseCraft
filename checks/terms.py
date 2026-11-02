"""Check: every registered term builds and gives a finite value and gradient on a tiny pair.

Run after adding or changing a term, the registry or the compose layer. CPU, seconds; terms whose
weights are gated are built without weights.

    python checks/terms.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from huggingface_hub import get_token

from sensecraft import TERMS, Loss, Term

GATED = {"dino_vit", "dino_convnext"}  # need a Hugging Face token even without weights


def main() -> None:
    """One forward and backward per term."""
    torch.manual_seed(0)
    token = get_token()
    for name, info in sorted(TERMS.items()):
        if name in GATED and token is None:
            print(f"skip  {name}: gated, no Hugging Face token")
            continue
        kwargs = {"pretrained": False} if name in GATED else {}
        dims = 3 if info.dims == "3d" else 2
        shape = (2, 9, 3, 64, 64) if dims == 3 else (2, 3, 224, 224)
        fake, real = torch.rand(shape) * 2 - 1, torch.rand(shape) * 2 - 1
        out = Loss([Term(name, **kwargs)], dims=dims).value_and_seed(fake, real)
        finite = bool(torch.isfinite(out.total)) and bool(
            torch.isfinite(out.seed).all()
        )
        print(("ok  " if finite else "FAIL") + f"  {name}")
        if not finite:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
