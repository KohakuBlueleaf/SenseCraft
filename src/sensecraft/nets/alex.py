"""AlexNet features as torchvision numbers them, tapped after each ReLU."""

from torch import nn

from sensecraft.nets.base import TapSequence, conv_relu

TAPS = [1, 4, 7, 9, 11]
CHANNELS = [64, 192, 384, 256, 256]


def alexnet() -> TapSequence:
    """AlexNet up to its last ReLU; weights come from `sensecraft.weights`."""
    layers = [
        *conv_relu(3, 64, kernel_size=11, stride=4, padding=2),
        nn.MaxPool2d(3, 2),
        *conv_relu(64, 192, kernel_size=5, padding=2),
        nn.MaxPool2d(3, 2),
        *conv_relu(192, 384, kernel_size=3, padding=1),
        *conv_relu(384, 256, kernel_size=3, padding=1),
        *conv_relu(256, 256, kernel_size=3, padding=1),
    ]
    return TapSequence(layers, TAPS)
