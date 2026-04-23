"""
models/bnn.py

Functional BNN targets for the Boomerang and Sticky Boomerang samplers.

Prior structure
---------------
Weights and biases get separate Gaussian priors:
    W ~ N(0, sigma_w^2)     with optional fan-in scaling
    b ~ N(0, sigma_b^2)

Fan-in scaling sets sigma_w^(l) = prior_std_weight / sqrt(n_in^(l)),
which is the standard Glorot/He-style scaling that keeps pre-activations
O(1). Set fan_in_scaling=False to get a single global sigma_w = prior_std_weight.
"""

from typing import List, Optional
import torch
from torch import Tensor
import torch.nn.functional as F

from ..smoke_test import TorchTarget
from ...utils.bnn_utils import (
    get_activation,
    count_params,
    unflatten_params,
    layer_shapes,
    find_reference_bnn,
    layer_slices_from_sizes,
)


# ===========================================================================
# Prior precision vector builder
# ===========================================================================

def _build_prior_precision(
    layer_sizes: List[int],
    prior_std_weight: float,
    prior_std_bias: float,
    fan_in_scaling: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Build a per-coordinate prior precision vector matching the flatten
    convention in unflatten_params / layer_slices_from_sizes.

    Layout (per layer): weights first, biases second.
    """
    D = count_params(layer_sizes)
    prec = torch.empty(D, dtype=dtype, device=device)
    offset = 0

    for layer_idx, (W_shape, b_shape) in enumerate(layer_shapes(layer_sizes)):
        n_W = W_shape[0] * W_shape[1]
        n_b = b_shape[0]
        n_in = layer_sizes[layer_idx]   # fan-in for this layer

        if fan_in_scaling:
            sigma_w_l = prior_std_weight / (n_in ** 0.5)
        else:
            sigma_w_l = prior_std_weight

        prec[offset : offset + n_W] = 1.0 / sigma_w_l ** 2
        offset += n_W
        prec[offset : offset + n_b] = 1.0 / prior_std_bias ** 2
        offset += n_b

    return prec


# ===========================================================================
# Forward pass (unchanged)
# ===========================================================================

def _forward(beta, X, layer_sizes, activation):
    params = unflatten_params(beta, layer_sizes)
    h = X
    for i, (W, b) in enumerate(params):
        h = h @ W.T + b
        if i < len(params) - 1:
            h = activation(h)
    return h


# ===========================================================================
# Negative log-posterior (now takes per-coord prior_precision)
# ===========================================================================

def _neg_log_posterior_regression(
    beta: Tensor,
    X: Tensor,
    y: Tensor,
    layer_sizes: List[int],
    activation,
    prior_precision: Tensor,
    noise_std: float,
) -> Tensor:
    preds = _forward(beta, X, layer_sizes, activation).squeeze(-1)
    log_lik = -0.5 * ((y - preds) ** 2).sum() / noise_std ** 2
    log_prior = -0.5 * (prior_precision * beta ** 2).sum()
    return -(log_lik + log_prior)


def _neg_log_posterior_classification(
    beta: Tensor,
    X: Tensor,
    y: Tensor,
    layer_sizes: List[int],
    activation,
    prior_precision: Tensor,
    n_classes: int,
) -> Tensor:
    logits = _forward(beta, X, layer_sizes, activation)

    if n_classes == 2:
        log_odds = logits.squeeze(-1)
        log_lik = -F.binary_cross_entropy_with_logits(
            log_odds, y.float(), reduction="sum"
        )
    else:
        log_lik = -F.cross_entropy(logits, y.long(), reduction="sum")

    log_prior = -0.5 * (prior_precision * beta ** 2).sum()
    return -(log_lik + log_prior)


# ===========================================================================
# Gradient wrapper (unchanged)
# ===========================================================================

def _make_grad_target(energy_fn):
    def grad_target(beta: Tensor) -> Tensor:
        with torch.enable_grad():
            beta_ = beta.detach().requires_grad_(True)
            E = energy_fn(beta_)
            g, = torch.autograd.grad(E, beta_)
        return g
    return grad_target


# ===========================================================================
# Factories
# ===========================================================================

def make_bnn_regression(
    X: Tensor,
    y: Tensor,
    layer_sizes: Optional[List[int]] = None,
    activation: str = "tanh",
    prior_std_weight: float = 1.0,
    prior_std_bias: float = 1.0,
    fan_in_scaling: bool = True,
    noise_std: float = 0.1,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    adam_steps: int = 2000,
    adam_lr: float = 1e-2,
) -> TorchTarget:
    """
    Bayesian neural network regression target.

    Priors
    ------
    Weights: N(0, (prior_std_weight / sqrt(fan_in))^2) if fan_in_scaling,
             else N(0, prior_std_weight^2).
    Biases:  N(0, prior_std_bias^2).

    prior_std_weight=1.0 with fan_in_scaling=True corresponds to
    standard Glorot-style initialisation variance.
    """
    X = X.to(dtype=dtype, device=device)
    y = y.to(dtype=dtype, device=device)

    d_in = X.shape[1]
    if layer_sizes is None:
        layer_sizes = [d_in, 32, 1]
    else:
        assert layer_sizes[0] == d_in, \
            f"layer_sizes[0]={layer_sizes[0]} must match X.shape[1]={d_in}"

    act = get_activation(activation)
    D = count_params(layer_sizes)

    prior_precision = _build_prior_precision(
        layer_sizes, prior_std_weight, prior_std_bias,
        fan_in_scaling, dtype, device,
    )

    def energy_fn(beta):
        return _neg_log_posterior_regression(
            beta, X, y, layer_sizes, act, prior_precision, noise_std
        )

    grad_target = _make_grad_target(energy_fn)

    print(f"Finding reference for regression BNN (D={D}, {adam_steps} Adam steps)...")
    x_ref, Sigma_inv = find_reference_bnn(
        energy_fn, D,
        layer_slices=layer_slices_from_sizes(layer_sizes),
        prior_precision=prior_precision,
        dtype=dtype, device=device,
        n_steps=adam_steps, lr=adam_lr,
    )

    return TorchTarget(
        name=f"bnn_regression_{'x'.join(str(s) for s in layer_sizes)}_{activation}",
        D=D,
        grad_target=grad_target,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={
            "layer_sizes": layer_sizes,
            "activation": activation,
            "prior_std_weight": prior_std_weight,
            "prior_std_bias": prior_std_bias,
            "fan_in_scaling": fan_in_scaling,
            "prior_precision": prior_precision,
            "noise_std": noise_std,
            "task": "regression",
            "energy_fn": energy_fn,
        },
    )


def make_bnn_classification(
    X: Tensor,
    y: Tensor,
    layer_sizes: Optional[List[int]] = None,
    activation: str = "tanh",
    prior_std_weight: float = 1.0,
    prior_std_bias: float = 1.0,
    fan_in_scaling: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    adam_steps: int = 2000,
    adam_lr: float = 1e-2,
) -> TorchTarget:
    """
    Bayesian neural network classification target.

    Priors same convention as make_bnn_regression.
    """
    X = X.to(dtype=dtype, device=device)
    y = y.to(device=device)

    d_in = X.shape[1]
    n_classes = int(y.max().item()) + 1
    d_out = 1 if n_classes == 2 else n_classes

    if layer_sizes is None:
        layer_sizes = [d_in, 32, d_out]
    else:
        assert layer_sizes[0] == d_in, \
            f"layer_sizes[0]={layer_sizes[0]} must match X.shape[1]={d_in}"
        assert layer_sizes[-1] == d_out, (
            f"layer_sizes[-1]={layer_sizes[-1]} must be 1 (binary) "
            f"or n_classes={n_classes} (multiclass)"
        )

    act = get_activation(activation)
    D = count_params(layer_sizes)

    prior_precision = _build_prior_precision(
        layer_sizes, prior_std_weight, prior_std_bias,
        fan_in_scaling, dtype, device,
    )

    def energy_fn(beta):
        return _neg_log_posterior_classification(
            beta, X, y, layer_sizes, act, prior_precision, n_classes
        )

    grad_target = _make_grad_target(energy_fn)

    print(f"Finding reference for classification BNN "
          f"(D={D}, {n_classes} classes, {adam_steps} Adam steps)...")
    x_ref, Sigma_inv = find_reference_bnn(
        energy_fn, D,
        layer_slices=layer_slices_from_sizes(layer_sizes),
        prior_precision=prior_precision,
        dtype=dtype, device=device,
        n_steps=adam_steps, lr=adam_lr,
    )

    return TorchTarget(
        name=(f"bnn_classification_{'x'.join(str(s) for s in layer_sizes)}"
              f"_{activation}_{n_classes}cls"),
        D=D,
        grad_target=grad_target,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={
            "layer_sizes": layer_sizes,
            "activation": activation,
            "prior_std_weight": prior_std_weight,
            "prior_std_bias": prior_std_bias,
            "fan_in_scaling": fan_in_scaling,
            "prior_precision": prior_precision,
            "n_classes": n_classes,
            "task": "classification",
            "energy_fn": energy_fn,
        },
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
    """
    Posterior predictive mean and std for regression.

    Parameters
    ----------
    samples : Tensor [S, D]   — posterior samples (e.g. from resample_pdmp_path)
    X_test  : Tensor [N, d_in]

    Returns
    -------
    mean : Tensor [N]
    std  : Tensor [N]
    """
    layer_sizes = target.meta["layer_sizes"]
    act = get_activation(target.meta["activation"])
    preds = []
    for beta in samples:
        p = _forward(beta, X_test, layer_sizes, act).squeeze(-1)
        preds.append(p)
    preds = torch.stack(preds)        # [S, N]
    return preds.mean(0), preds.std(0)


@torch.no_grad()
def predict_classification(
    samples: Tensor,
    X_test: Tensor,
    target: TorchTarget,
) -> tuple[Tensor, Tensor]:
    """
    Posterior predictive class probabilities and entropy for classification.

    Returns
    -------
    probs   : Tensor [N, n_classes]  — mean predictive probabilities
    entropy : Tensor [N]             — predictive entropy
    """
    layer_sizes = target.meta["layer_sizes"]
    act = get_activation(target.meta["activation"])
    n_classes = target.meta["n_classes"]
    probs_list = []

    for beta in samples:
        logits = _forward(beta, X_test, layer_sizes, act)
        if n_classes == 2:
            p1 = torch.sigmoid(logits.squeeze(-1))
            p = torch.stack([1 - p1, p1], dim=-1)
        else:
            p = torch.softmax(logits, dim=-1)
        probs_list.append(p)

    probs = torch.stack(probs_list).mean(0)           # [N, n_classes]
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum(-1)  # [N]
    return probs, entropy
