"""T5 memory: what each term holds, from the input to its gradient.

Per term and compile scope: memory held after the forward (what backward will read), and the
peak over one `value_and_seed`, both above the inputs. Real compute.

    python measure/t5_memory.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import torch
from terms import TERMS

from sensecraft.compose import Loss, Term

DEVICE: str = "cuda:0"
KEYS: list[str] = ["lpips_vgg", "dino_base_mse"]
BATCH: int = 32
SIZE: int = 256
DTYPE: str = "bfloat16"
SCOPES: list[str] = ["off", "term"]
MIB = 2**20


def main() -> None:
    """Print held and peak MiB per term and scope."""
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    dtype = getattr(torch, DTYPE)
    for key in KEYS:
        kind, kwargs = TERMS[key]
        name = "lpips" if kind == "lpips" else "dino_vit"
        for scope in SCOPES:
            loss = Loss([Term(name, **kwargs)]).prepare(
                DEVICE, dtype, torch.channels_last, scope
            )
            real = (torch.rand(BATCH, 3, SIZE, SIZE, device=DEVICE) * 2 - 1).to(
                memory_format=torch.channels_last
            )
            fake = (real + 0.1 * torch.randn_like(real)).clamp(-1, 1)
            for _ in range(3):
                with torch.autocast("cuda", dtype=dtype):
                    loss.value_and_seed(fake, real)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Held after forward: the graph backward will read
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            leaf = fake.detach().requires_grad_()
            with torch.autocast("cuda", dtype=dtype):
                total = loss(leaf, real).total
            held = torch.cuda.memory_allocated() - base
            torch.autograd.grad(total, leaf)
            peak = torch.cuda.max_memory_allocated() - base
            del total, leaf
            print(
                f"{key:15s} compile {scope:4s}  held after forward {held / MIB:8.0f} MiB  peak {peak / MIB:8.0f} MiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
