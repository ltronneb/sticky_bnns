"""Network architectures, a fully connected net (toy, UCI, MNIST), LeNet-5
(MNIST) and ResNet-20 (CIFAR-10)."""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ACTIVATIONS = {"tanh": torch.tanh, "relu": torch.relu}


class FFN(nn.Module):
    def __init__(self, layer_sizes: Sequence[int], activation: str = "tanh"):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(layer_sizes[:-1], layer_sizes[1:]))
        self.activation = ACTIVATIONS[activation]

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class LeNet5(nn.Module):
    """LeCun et al. (1998) LeNet-5 with average pooling. 28x28 inputs are
    padded to 32x32."""

    def __init__(self, activation: str = "tanh"):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.activation = ACTIVATIONS[activation]

    def forward(self, x: Tensor) -> Tensor:
        a = self.activation
        x = F.avg_pool2d(a(self.conv1(F.pad(x, [2, 2, 2, 2]))), 2)
        x = F.avg_pool2d(a(self.conv2(x)), 2)
        x = a(self.fc1(torch.flatten(x, 1)))
        return self.fc3(a(self.fc2(x)))


class _Block(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.activation = activation
        self.shortcut_conv = self.shortcut_bn = None
        if stride != 1 or c_in != c_out:
            self.shortcut_conv = nn.Conv2d(c_in, c_out, 1, stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(c_out)

    def forward(self, x: Tensor) -> Tensor:
        h = self.bn2(self.conv2(self.activation(self.bn1(self.conv1(x)))))
        skip = x if self.shortcut_conv is None else self.shortcut_bn(self.shortcut_conv(x))
        return self.activation(h + skip)


class ResNet20(nn.Module):
    """CIFAR-10 ResNet-20 (He et al., 2016). BatchNorm running statistics are
    buffers, not parameters. They come from an SGD pretrain and the module is
    kept in eval() mode afterwards, so the target is a fixed function."""

    def __init__(self, activation: str = "relu"):
        super().__init__()
        a = self.activation = ACTIVATIONS[activation]
        self.stem_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.stage1, self.stage2, self.stage3 = (
            nn.ModuleList(_Block(c_in if k == 0 else c_out, c_out, stride if k == 0 else 1, a)
                          for k in range(3))
            for c_in, c_out, stride in [(16, 16, 1), (16, 32, 2), (32, 64, 2)])
        self.fc = nn.Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        h = self.activation(self.stem_bn(self.stem_conv(x)))
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                h = block(h)
        return self.fc(F.adaptive_avg_pool2d(h, 1).flatten(1))
