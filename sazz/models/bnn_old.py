"""
models/bnn.py

Functional BNN targets for the Boomerang and Sticky Boomerang samplers.

Two factories:
    make_bnn_regression      — Gaussian likelihood, any architecture
    make_bnn_classification  — Categorical/Bernoulli likelihood, any architecture

Both return a TorchTarget with:
    .grad_target : (beta: Tensor[D]) -> Tensor[D]
    .x_ref       : Tensor[D]   (from Adam warm-start)
    .Sigma_inv   : Tensor[D,D] (diagonal, from empirical Fisher or Hessian diag)

Architecture is specified as layer_sizes = [d_in, h1, ..., h_L, d_out].

Usage
-----
    target = make_bnn_regression(X_train, y_train, layer_sizes=[8, 32, 32, 1])
    # or
    target = make_bnn_classification(X_train, y_train, layer_sizes=[8, 32, 32, 3])

    sampler = AutomaticBoomerangSampler(
        grad_target=target.grad_target, D=target.D, ...
    )
    sampler.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
"""

from typing import List, Optional
import torch
from torch import Tensor
import torch.nn.functional as F

from .smoke_test import TorchTarget      # reuse the same container
#from ..utils.warmup import find_reference
from ..utils.bnn_utils import (
    get_activation,
    count_params,
    unflatten_params,
    find_reference_bnn,
    layer_slices_from_sizes
)

# ===========================================================================
# Forward pass (functional, no nn.Module)
# ===========================================================================

def _forward(
    beta: Tensor,
    X: Tensor,
    layer_sizes: List[int],
    activation,
) -> Tensor:
    """
    Forward pass of a fully-connected network.

    Parameters
    ----------
    beta : Tensor [D]
        Flat parameter vector.
    X : Tensor [N, d_in]
        Input data.
    layer_sizes : list[int]
        e.g. [d_in, h1, h2, d_out]
    activation : callable
        Applied after every hidden layer (not the output layer).

    Returns
    -------
    Tensor [N, d_out]  — raw logits / predictions (no final activation).
    """
    params = unflatten_params(beta, layer_sizes)
    h = X
    for i, (W, b) in enumerate(params):
        h = h @ W.T + b
        if i < len(params) - 1:       # hidden layers only
            h = activation(h)
    return h                           # [N, d_out]

# ===========================================================================
# Negative log-posterior and its gradient
# ===========================================================================

def _neg_log_posterior_regression(
    beta: Tensor,
    X: Tensor,
    y: Tensor,
    layer_sizes: List[int],
    activation,
    prior_std: float,
    noise_std: float,
) -> Tensor:
    """
    E(beta) = -log p(y|beta,X) - log p(beta)
            = (1/2 noise_std^2) ||y - f(X;beta)||^2
            + (1/2 prior_std^2) ||beta||^2
            + const
    """
    preds = _forward(beta, X, layer_sizes, activation).squeeze(-1)  # [N]
    n = X.shape[0]
    log_lik = -0.5 * ((y - preds) ** 2).sum() / noise_std ** 2
    log_prior = -0.5 * (beta ** 2).sum() / prior_std ** 2
    return -(log_lik + log_prior)


def _neg_log_posterior_classification(
    beta: Tensor,
    X: Tensor,
    y: Tensor,
    layer_sizes: List[int],
    activation,
    prior_std: float,
    n_classes: int,
) -> Tensor:
    """
    E(beta) = -log p(y|beta,X) - log p(beta)

    Binary  (n_classes=2): Bernoulli likelihood with sigmoid output.
    Multi   (n_classes>2): Categorical likelihood with softmax output.
    """
    logits = _forward(beta, X, layer_sizes, activation)   # [N, d_out]

    if n_classes == 2:
        # logits: [N, 1] or [N, 2] — use first output as log-odds
        log_odds = logits.squeeze(-1)                      # [N]
        log_lik = -F.binary_cross_entropy_with_logits(
            log_odds, y.float(), reduction="sum"
        )
    else:
        log_lik = -F.cross_entropy(logits, y.long(), reduction="sum")

    log_prior = -0.5 * (beta ** 2).sum() / prior_std ** 2
    return -(log_lik + log_prior)

# ===========================================================================
# Gradient of the energy w.r.t. beta
# ===========================================================================

def _make_grad_target(energy_fn):
    def grad_target(beta: Tensor) -> Tensor:
        # Always compute gradient regardless of outer no_grad context
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
    prior_std: float = 1.0,
    noise_std: float = 0.1,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    adam_steps: int = 2000,
    adam_lr: float = 1e-2,
) -> TorchTarget:
    """
    Bayesian neural network regression target.

    Parameters
    ----------
    X : Tensor [N, d_in]
    y : Tensor [N]
    layer_sizes : list[int] | None
        e.g. [d_in, 32, 32, 1]. If None, defaults to [d_in, 32, 1].
    activation : str
        "relu" or "tanh" (or any key in bnn_utils.get_activation).
    prior_std : float
        Standard deviation of the isotropic Gaussian prior on weights.
    noise_std : float
        Observation noise standard deviation.
    adam_steps : int
        Number of Adam steps to find the reference point.
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

    def energy_fn(beta):
        return _neg_log_posterior_regression(
            beta, X, y, layer_sizes, act, prior_std, noise_std
        )

    grad_target = _make_grad_target(energy_fn)

    print(f"Finding reference for regression BNN (D={D}, {adam_steps} Adam steps)...")
    prior_precision = torch.full(
        (D,), 1.0 / prior_std**2, dtype=dtype, device=device
    )
    x_ref, Sigma_inv = find_reference_bnn(
        energy_fn, D,
        layer_slices=layer_slices_from_sizes(layer_sizes),
        prior_precision=prior_precision,
        dtype=dtype, device=device,
        n_steps=adam_steps, lr=adam_lr,
    )
    # x_ref, Sigma_inv = find_reference_bnn(
    #     energy_fn, D, dtype=dtype, device=device, #layer_slices=layer_slices_from_sizes(layer_sizes),
    #     n_steps=adam_steps, lr=adam_lr,
    # )

    return TorchTarget(
        name=f"bnn_regression_{'x'.join(str(s) for s in layer_sizes)}_{activation}",
        D=D,
        grad_target=grad_target,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={
            "layer_sizes": layer_sizes,
            "activation": activation,
            "prior_std": prior_std,
            "noise_std": noise_std,
            "task": "regression",
            "energy_fn": energy_fn,   # kept for prediction
        },
    )


def make_bnn_classification(
    X: Tensor,
    y: Tensor,
    layer_sizes: Optional[List[int]] = None,
    activation: str = "tanh",
    prior_std: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    adam_steps: int = 2000,
    adam_lr: float = 1e-2,
) -> TorchTarget:
    """
    Bayesian neural network classification target.

    Handles binary (n_classes=2) and multi-class (n_classes>2).
    For binary: output layer has 1 unit, Bernoulli likelihood.
    For multi-class: output layer has n_classes units, Categorical likelihood.

    Parameters
    ----------
    X : Tensor [N, d_in]
    y : Tensor [N]    — integer class labels starting at 0
    layer_sizes : list[int] | None
        e.g. [d_in, 32, 32, n_classes].
        If None, inferred from data: [d_in, 32, n_classes].
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

    def energy_fn(beta):
        return _neg_log_posterior_classification(
            beta, X, y, layer_sizes, act, prior_std, n_classes
        )

    grad_target = _make_grad_target(energy_fn)

    print(f"Finding reference for classification BNN "
          f"(D={D}, {n_classes} classes, {adam_steps} Adam steps)...")
    prior_precision = torch.full(
        (D,), 1.0 / prior_std**2, dtype=dtype, device=device
    )
    x_ref, Sigma_inv = find_reference_bnn(
        energy_fn, D,
        layer_slices=layer_slices_from_sizes(layer_sizes),
        prior_precision=prior_precision,
        dtype=dtype, device=device,
        n_steps=adam_steps, lr=adam_lr,
    )
    # x_ref, Sigma_inv = find_reference_bnn(
    #     energy_fn, D, dtype=dtype, device=device, layer_slices=layer_slices_from_sizes(layer_sizes),
    #     n_steps=adam_steps, lr=adam_lr,
    # )

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
            "prior_std": prior_std,
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
