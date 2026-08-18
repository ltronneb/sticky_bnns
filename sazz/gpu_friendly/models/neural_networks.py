import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Sequence, Union

_ACTIVATIONS = {
    "relu":       torch.relu,
    "tanh":       torch.tanh,
    "sigmoid":    torch.sigmoid,
    "leaky_relu": F.leaky_relu,
    "elu":        F.elu,
}

def get_activation(name: str):
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]


class FFN(nn.Module):
    """
    Plain fully-connected network. Stored as nn.ModuleList so that
    named_parameters() yields layers.0.weight, layers.0.bias, layers.1.weight,
    ..., matching ParamSpec.from_layer_sizes ordering.
    """
    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: Union[str, callable] = "tanh",
    ):
        super().__init__()

        if isinstance(activation, str):
            activation = get_activation(activation)

        self.layers = nn.ModuleList([
            nn.Linear(n_in, n_out)
            for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ])

        self.activation = activation
        self._n_layers = len(self.layers)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < self._n_layers - 1:
                h = self.activation(h)
        return h


class CNN(nn.Module):
    """LeNet-style CNN for ~28x28 grayscale images. Adaptive pooling means
    input size is somewhat flexible, but the channel/class count is fixed
    at construction.

    pool: "avg" (default) uses F.avg_pool2d, "max" uses F.max_pool2d (the
    original hardcoded behavior). Exposed as a constructor param rather
    than a forward()-only choice so switching pooling never requires
    editing this class again -- a grid-bound PDMP target wants "avg"
    (max_pool2d is non-smooth, a second kink source independent of
    activation choice), while "max" stays available for a relu+max_pool2d
    ablation against the original architecture. Changes no shapes/D either
    way, but the target's rate function does depend on it -- a checkpoint
    trained against one choice is not a valid reference for the other.
    """
    def __init__(
        self,
        activation: Union[str, callable] = "relu",
        pool: str = "avg",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # after global average pooling we have 64 features
        self.fc = nn.Linear(64, 10)

        if isinstance(activation, str):
                activation = get_activation(activation)
        self.activation = activation

        if pool not in ("avg", "max"):
            raise ValueError(f"Unknown pool '{pool}'. Choose from ('avg', 'max')")
        self.pool = pool

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, 2) if self.pool == "avg" else F.max_pool2d(x, 2)
        # (batch, 64, 12, 12) -> (batch, 64, 1, 1)
        x = F.adaptive_avg_pool2d(x, 1)
        # (batch, 64)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LeNet5(nn.Module):
    """Classic LeCun et al. (1998) LeNet-5, ~61.7k params. Input is padded
    2px on each side (28x28 -> 32x32) inside forward() so this drops
    straight into this repo's 28x28 MNIST pipeline while preserving the
    original architecture's exact layer shapes and parameter count.

    pool: "avg" (default, matches the original subsampling layers) or
    "max" -- same rationale as CNN.pool (max_pool2d is non-smooth, a
    second kink source independent of activation choice).
    """
    def __init__(
        self,
        activation: Union[str, callable] = "tanh",
        pool: str = "avg",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        if isinstance(activation, str):
            activation = get_activation(activation)
        self.activation = activation

        if pool not in ("avg", "max"):
            raise ValueError(f"Unknown pool '{pool}'. Choose from ('avg', 'max')")
        self.pool = pool

    def forward(self, x):
        x = F.pad(x, [2, 2, 2, 2])  # 28x28 -> 32x32, LeCun's original convention
        x = self.conv1(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, 2) if self.pool == "avg" else F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, 2) if self.pool == "avg" else F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

