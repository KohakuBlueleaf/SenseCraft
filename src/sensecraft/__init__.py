"""SenseCraft: perceptual and signal losses for image and video models.

from sensecraft import Loss, Term

loss = Loss([Term("mse"), Term("lpips", 0.5, net="vgg")]).prepare("cuda", torch.bfloat16)
out = loss.value_and_seed(fake, real)
"""

# Importing the term packages registers every term
from sensecraft import perceptual, pixel, signal
from sensecraft.compose import Loss, Out, Term
from sensecraft.prepare import prepare
from sensecraft.registry import TERMS, Range, lookup, register

__all__ = [
    "TERMS",
    "Loss",
    "Out",
    "Range",
    "Term",
    "lookup",
    "perceptual",
    "pixel",
    "prepare",
    "register",
    "signal",
]
