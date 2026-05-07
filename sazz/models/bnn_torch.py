from typing import List, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, Bernoulli, Categorical
from .make_models import TorchTarget
from .priors_torch import Prior
from .likelihoods_torch import Likelihood
from .models_torch import BayesianModel

from ..utils.OLD.bnn_utils import unflatten_params, _build_prior_precision, get_activation, count_params
from ..utils.warmup import find_reference_bnn
from ..utils.bnn_modular_utils import ParamSpec



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

    def log_prob_single(self, beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        """Log-likelihood on an arbitrary (X_i, y_i) — used by the
        empirical Fisher estimator in find_reference_bnn."""
        preds = self.predict(beta, X_i).squeeze(-1)
        dist = Normal(preds, self.noise_std)
        return dist.log_prob(y_i).sum()

    def log_prob(self, beta: Tensor) -> Tensor:
        return self.log_prob_single(beta, self.X, self.y)


class BNNBernoulliLikelihood(BNNLikelihood):
    """Binary classification: y | x, beta ~ Bernoulli(sigmoid(f(x; beta)))."""

    def log_prob_single(self, beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        """Log-likelihood on an arbitrary (X_i, y_i) — used by the
        empirical Fisher estimator in find_reference_bnn."""
        logits = self.predict(beta, X_i).squeeze(-1)
        dist = Bernoulli(logits=logits)
        return dist.log_prob(y_i.to(logits.dtype)).sum()

    def log_prob(self, beta: Tensor) -> Tensor:
        return self.log_prob_single(beta, self.X, self.y)


class BNNCategoricalLikelihood(BNNLikelihood):
    """Multiclass: y | x, beta ~ Categorical(softmax(f(x; beta)))."""

    def log_prob_single(self, beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        """Log-likelihood on an arbitrary (X_i, y_i) — used by the
        empirical Fisher estimator in find_reference_bnn."""
        logits = self.predict(beta, X_i)
        dist = Categorical(logits=logits)
        return dist.log_prob(y_i.long()).sum()

    def log_prob(self, beta: Tensor) -> Tensor:
        return self.log_prob_single(beta, self.X, self.y)
 
 
 
 

 
class ModuleLikelihood(Likelihood):
    def __init__(self, module: nn.Module, spec: ParamSpec, 
                 X: Tensor, y: Tensor):
        super().__init__()
        self.module = module
        self.module.eval()              # turn off dropout/BN updates
        self.spec = spec
        self.register_buffer("X", X)
        self.register_buffer("y", y)
    
    def predict(self, beta: Tensor, X_new: Tensor) -> Tensor:
        return torch.func.functional_call(
            self.module, self.spec.to_dict(beta), (X_new,)
        )
  

class ModuleGaussianPrior(Prior):
    """Diagonal Gaussian prior, indexed by a precision vector aligned with
    a ParamSpec's flatten order."""

    def __init__(self, prec: Tensor):
        super().__init__()
        self.register_buffer("_precision", prec)

    def log_prob(self, beta: Tensor) -> Tensor:
        return -0.5 * (self._precision * beta ** 2).sum()

    def precision_diag(self) -> Tensor:
        return self._precision

        
class ModuleGaussianLikelihood(ModuleLikelihood):
    """BNN regression likelihood evaluated via torch.func.functional_call.

    Wraps an nn.Module + ParamSpec. The forward pass uses functional_call so
    the module's parameters are taken from the flat beta vector instead of
    its own .parameters().
    """

    def __init__(self, module: nn.Module, spec: ParamSpec,
                 X: Tensor, y: Tensor, noise_std: float):
        super().__init__(module, spec, X, y) 
        self.noise_std = noise_std

    def log_prob_single(self, beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        preds = self.predict(beta, X_i).squeeze(-1)
        return Normal(preds, self.noise_std).log_prob(y_i).sum()

    def log_prob(self, beta: Tensor) -> Tensor:
        return self.log_prob_single(beta, self.X, self.y)
 

class ModuleGaussianLikelihoodLearnedNoise(ModuleLikelihood):
    """Gaussian regression with a learned noise scale.

    Sampled coordinate is log_sigma (last entry of beta). The implied prior
    on sigma = exp(log_sigma) is HalfNormal(prior_sigma_scale). The
    half-normal density on sigma, transformed to log_sigma via the standard
    change-of-variables, contributes:

        log p(log_sigma) = -sigma**2 / (2 * tau**2) + log_sigma + const

    The corresponding entry in ModuleGaussianPrior's precision vector
    should be 0, since the prior on log_sigma lives entirely here.
    """
    def __init__(self, module, base_spec, X, y, prior_sigma_scale=1.0):
        extended_spec = base_spec.with_extra_scalar(
            "log_sigma", can_freeze=False,
        )
        super().__init__(module, extended_spec, X, y)
        self.base_spec = base_spec
        self.prior_sigma_scale = prior_sigma_scale

    def _split(self, beta):
        return beta[:-1], beta[-1]

    def predict(self, beta, X_new):
        weights, _ = self._split(beta)
        params = self.base_spec.to_dict(weights)
        return torch.func.functional_call(self.module, params, (X_new,))

    def log_prob_single(self, beta, X_i, y_i):
        _, log_sigma = self._split(beta)
        sigma = log_sigma.exp()
        preds = self.predict(beta, X_i).squeeze(-1)
        log_lik = Normal(preds, sigma).log_prob(y_i).sum()
        # HalfNormal(tau) on sigma, with Jacobian to log_sigma
        log_prior = -0.5 * (sigma / self.prior_sigma_scale) ** 2 + log_sigma
        return log_lik + log_prior

    def log_prob(self, beta):
        return self.log_prob_single(beta, self.X, self.y)
    
 
class ModuleCategoricalLikelihood(ModuleLikelihood):
    """Multiclass classification likelihood evaluated via functional_call."""

    def __init__(self, module: nn.Module, spec: ParamSpec,
                 X: Tensor, y: Tensor):
        super().__init__(module, spec, X, y.long())

    def log_prob_single(self, beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        logits = self.predict(beta, X_i)
        return Categorical(logits=logits).log_prob(y_i.long()).sum()

    def log_prob(self, beta: Tensor) -> Tensor:
        return self.log_prob_single(beta, self.X, self.y)

    
    
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


