"""Feature nets: built empty, filled by `sensecraft.weights`."""

from sensecraft.nets.alex import alexnet
from sensecraft.nets.base import FeatureNet, TapSequence
from sensecraft.nets.squeeze import squeezenet
from sensecraft.nets.vgg import vgg16

__all__ = ["FeatureNet", "TapSequence", "alexnet", "squeezenet", "vgg16"]
