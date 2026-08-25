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


class _BasicBlock(nn.Module):
    """One ResNet-v1 basic block: conv-bn-act, conv-bn, residual add, act.
    Bias-free convs throughout (BatchNorm's own bias makes a conv bias
    redundant -- standard ResNet convention, also keeps every 1-D parameter
    in this module a genuine BatchNorm weight/bias, never a stray conv/
    linear bias). stride=2 on the first conv downsamples space and requires
    a matching 1x1-conv+BN shortcut (channel/stride change); stride=1 with
    matching in/out channels uses a plain identity shortcut."""
    def __init__(self, in_channels: int, out_channels: int, stride: int, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = activation

        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                            stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.shortcut_conv is not None:
            identity = self.shortcut_bn(self.shortcut_conv(identity))
        h = self.activation(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.activation(h + identity)


class ResNet20(nn.Module):
    """Standard Keras-style CIFAR-10 ResNet-20 ("v1", He et al. 2015) --
    the architecture Wenzel et al. 2020 (and, per Goan et al. 2023, the
    "modified ResNet20" they build on) use for their CIFAR-10 SG-MCMC
    experiments. Input: [N, 3, 32, 32].

    Structure: a bias-free 3x3 stem conv + BN + activation (16 channels),
    then 3 stages of 3 _BasicBlocks each (16/32/64 filters) -- the first
    block of stages 2 and 3 downsamples (stride=2: 32x32 -> 16x16 -> 8x8)
    via a 1x1-conv+BN shortcut, every other block uses stride=1 with an
    identity shortcut. Head: global average pool -> Linear(64, 10) (the
    one bias-ful layer in the whole network, matching CNN/LeNet5's `fc`
    convention). D = 272,474 trainable parameters, 21 BatchNorm2d layers.

    Every BatchNorm2d layer's running_mean/running_var/num_batches_tracked
    are ordinary nn.Module buffers -- BayesianModule.build only ever
    flattens named_parameters() into beta (see model.py), so these buffers
    are NEVER part of the sampled state. They must be populated by an
    ordinary (non-Bayesian) pretrain pass and the module switched to
    eval() PERMANENTLY before being handed to BayesianModule.build --
    train-mode BatchNorm recomputes per-call batch statistics and mutates
    these buffers as a side effect, which breaks the sticky PDMP samplers'
    requirement that grad_target be a fixed function of x for an entire
    _grid_bound episode. See diagnose_batchnorm_eval.py (Phase 0 of the
    CIFAR-10/ResNet-20 plan) for the empirical validation of this
    eval-mode-only usage pattern, and priors.py's build_*_resnet functions
    for the corresponding BatchNorm-aware prior/kappa/freeze-mask builders
    (a plain nn.BatchNorm2d's `weight`/`bias` are both 1-D, so the base
    priors.py builders' dim()==1 "is this a bias" heuristic would otherwise
    misclassify BatchNorm's weight/gamma as a bias).

    No `pool` constructor parameter, unlike CNN/LeNet5 -- deliberate, not
    an oversight: all downsampling here is via smooth strided convs, never
    max-pooling, so there is no non-smooth-pooling-vs-smooth-pooling choice
    to expose as a knob the way there is for CNN/LeNet5's avg/max toggle.
    """
    def __init__(self, activation: Union[str, callable] = "relu"):
        super().__init__()
        if isinstance(activation, str):
            activation = get_activation(activation)
        self.activation = activation

        self.stem_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)

        self.stage1 = self._make_stage(16, 16, n_blocks=3, first_stride=1)
        self.stage2 = self._make_stage(16, 32, n_blocks=3, first_stride=2)
        self.stage3 = self._make_stage(32, 64, n_blocks=3, first_stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def _make_stage(self, in_channels: int, out_channels: int, n_blocks: int, first_stride: int) -> nn.ModuleList:
        blocks = [_BasicBlock(in_channels, out_channels, first_stride, self.activation)]
        for _ in range(n_blocks - 1):
            blocks.append(_BasicBlock(out_channels, out_channels, 1, self.activation))
        return nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        h = self.activation(self.stem_bn(self.stem_conv(x)))
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                h = block(h)
        h = self.pool(h).flatten(1)
        return self.fc(h)

