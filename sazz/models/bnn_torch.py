from typing import List, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, Bernoulli, Categorical
from .smoke_test import TorchTarget
from .priors_torch import Prior
from .likelihoods_torch import Likelihood
from .models_torch import BayesianModel

from ..utils.bnn_utils import unflatten_params, _build_prior_precision, get_activation, count_params
from ..utils.warmup import find_reference_bnn


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
        return self.predict(beta, self.X)

    def predict(self, beta: Tensor, X_new: Tensor) -> Tensor:
        params = unflatten_params(beta, self.layer_sizes)
        h = X_new
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
    
    
def make_bnn_regression(X, y, layer_sizes, activation="tanh",
                       prior_std_weight=1.0, prior_std_bias=1.0,
                       fan_in_scaling=True, noise_std=0.1,
                       covariance_reference="prior",
                       adam_steps=1000, adam_lr=1e-2,
                       dtype=torch.float64, device="cpu"):
    X = X.to(dtype=dtype, device=device)
    y = y.to(dtype=dtype, device=device)
    D = count_params(layer_sizes)
    act = get_activation(activation)

    prior = BNNGaussianPrior(
        layer_sizes, prior_std_weight, prior_std_bias,
        fan_in_scaling, dtype, device,
    )
    likelihood = BNNGaussianLikelihood(X, y, layer_sizes, act, noise_std)
    model = BayesianModel(prior, likelihood)

    x_ref, Sigma_inv = find_reference_bnn(model.energy, D, 
                                          model=model, dtype=dtype,
                                          device=device,
                                          reference=covariance_reference,
                                          n_steps=adam_steps, lr=adam_lr)

    return TorchTarget(
        name=f"bnn_regression_{'x'.join(map(str, layer_sizes))}_{activation}",
        D=D,
        grad_target=model.grad_energy,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={"model": model, "layer_sizes": layer_sizes},
    )
    


def make_bnn_classification(X, y, layer_sizes, activation="tanh",
                            prior_std_weight=1.0, prior_std_bias=1.0,
                            fan_in_scaling=True,
                            adam_steps=1000, adam_lr=1e-2,
                            dtype=torch.float64, device="cpu"):
    X = X.to(dtype=dtype, device=device)
    y = y.to(device=device)

    d_in = X.shape[1]
    n_classes = int(y.max().item()) + 1
    d_out = 1 if n_classes == 2 else n_classes

    assert layer_sizes[0] == d_in, \
        f"layer_sizes[0]={layer_sizes[0]} must match X.shape[1]={d_in}"
    assert layer_sizes[-1] == d_out, (
        f"layer_sizes[-1]={layer_sizes[-1]} must be 1 (binary) "
        f"or n_classes={n_classes} (multiclass)"
    )

    D = count_params(layer_sizes)
    act = get_activation(activation)

    prior = BNNGaussianPrior(
        layer_sizes, prior_std_weight, prior_std_bias,
        fan_in_scaling, dtype, device,
    )

    if n_classes == 2:
        likelihood = BNNBernoulliLikelihood(X, y, layer_sizes, act)
    else:
        likelihood = BNNCategoricalLikelihood(X, y, layer_sizes, act)

    model = BayesianModel(prior, likelihood)

    x_ref, Sigma_inv = find_reference_bnn(
        model.energy, D,
        model=model, dtype=dtype, device=device,
        n_steps=adam_steps, lr=adam_lr,
    )

    return TorchTarget(
        name=(f"bnn_classification_{'x'.join(map(str, layer_sizes))}"
              f"_{activation}_{n_classes}cls"),
        D=D,
        grad_target=model.grad_energy,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={"model": model, "layer_sizes": layer_sizes, "n_classes": n_classes},
    )
    
    
# ===========================================================================
# Prediction utilities
# ===========================================================================

@torch.no_grad()
def predict_regression(
    samples: Tensor,
    X_test: Tensor,
    target: TorchTarget,
) -> tuple[Tensor, Tensor]:
    """Posterior predictive mean and std for regression."""
    likelihood = target.meta["model"].likelihood
    X_test = X_test.to(dtype=likelihood.X.dtype, device=likelihood.X.device)
    preds = torch.stack([
        likelihood.predict(beta, X_test).squeeze(-1) for beta in samples
    ])
    return preds.mean(0), preds.std(0)


@torch.no_grad()
def predict_classification(
    samples: Tensor,
    X_test: Tensor,
    target: TorchTarget,
) -> tuple[Tensor, Tensor]:
    """Posterior predictive class probabilities and entropy."""
    likelihood = target.meta["model"].likelihood
    X_test = X_test.to(dtype=likelihood.X.dtype, device=likelihood.X.device)
    probs_list = []
    for beta in samples:
        logits = likelihood.predict(beta, X_test)
        if logits.shape[-1] == 1 or logits.ndim == 1:
            p1 = torch.sigmoid(logits.squeeze(-1))
            p = torch.stack([1 - p1, p1], dim=-1)
        else:
            p = torch.softmax(logits, dim=-1)
        probs_list.append(p)
    probs = torch.stack(probs_list).mean(0)
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum(-1)
    return probs, entropy


