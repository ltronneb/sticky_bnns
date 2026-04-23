from typing import List, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, Bernoulli, Categorical
from .models_torch import Likelihood
from .priors_torch import Prior

from ..utils.bnn_utils import unflatten_params, _build_prior_precision


class BNNLikelihood(Likelihood):
    """
    Base class for BNN likelihoods. Handles the forward pass from a
    flat parameter vector through the layer structure. Subclasses
    implement log_prob by wrapping the output in an appropriate
    distribution.
    """
    def __init__(
        self,
        X: Tensor,
        y: Tensor,
        layer_sizes: List[int],
        activation: Callable,
    ):
        super().__init__()
        self.register_buffer("X", X)
        self.register_buffer("y", y)
        self.layer_sizes = layer_sizes
        self.activation = activation

    def forward_net(self, beta: Tensor) -> Tensor:
        """Run the BNN forward pass from a flat parameter vector."""
        params = unflatten_params(beta, self.layer_sizes)
        h = self.X
        for i, (W, b) in enumerate(params):
            h = h @ W.T + b
            if i < len(params) - 1:
                h = self.activation(h)
        return h


class BNNGaussianPrior(Prior):
    def __init__(self, layer_sizes, prior_std_weight, prior_std_bias,
                 fan_in_scaling=True, dtype=torch.float64, device="cpu"):
        super().__init__()
        prec = _build_prior_precision(
            layer_sizes, prior_std_weight, prior_std_bias,
            fan_in_scaling, dtype, device,
        )
        self.register_buffer("_precision", prec)

    def log_prob(self, beta: Tensor) -> Tensor:
        return -0.5 * (self._precision * beta ** 2).sum()

    def precision_diag(self) -> Tensor:
        return self._precision


class BNNGaussianLikelihood(BNNLikelihood):
    """Regression: y | x, beta ~ N(f(x; beta), noise_std^2)."""
    def __init__(self, X, y, layer_sizes, activation, noise_std: float = 0.1):
        super().__init__(X, y, layer_sizes, activation)
        self.noise_std = noise_std

    def log_prob(self, beta: Tensor) -> Tensor:
        preds = self.forward_net(beta).squeeze(-1)
        dist = Normal(preds, self.noise_std)
        return dist.log_prob(self.y).sum()


class BNNBernoulliLikelihood(BNNLikelihood):
    """Binary classification: y | x, beta ~ Bernoulli(sigmoid(f(x; beta)))."""
    def log_prob(self, beta: Tensor) -> Tensor:
        logits = self.forward_net(beta).squeeze(-1)
        dist = Bernoulli(logits=logits)
        return dist.log_prob(self.y.to(logits.dtype)).sum()


class BNNCategoricalLikelihood(BNNLikelihood):
    """Multiclass: y | x, beta ~ Categorical(softmax(f(x; beta)))."""
    def log_prob(self, beta: Tensor) -> Tensor:
        logits = self.forward_net(beta)
        dist = Categorical(logits=logits)
        return dist.log_prob(self.y.long()).sum()