"""
models/glm.py

Generalised linear model targets for the Boomerang and Sticky samplers.

Factories
---------
    make_linear_regression   — Gaussian likelihood, identity link
    make_logistic_regression — Bernoulli/Categorical likelihood, logit link
    make_glm                 — Generic GLM dispatcher (poisson | gamma)

All factories:
- Prepend a column of 1s to X for the intercept (implicit handling).
- Put a very loose prior on the intercept (std = intercept_prior_std,
  default 10) and a tighter prior on the remaining coefficients
  (prior_std, default 1).
- Return a TorchTarget with an energy_fn stored in .meta for
  downstream prediction / diagnostics.

Stickiness
----------
For the sticky sampler, use make_kappa_vector_glm to build a kappa
vector that applies stickiness only to the coefficient entries and
leaves the intercept always thawed.

Usage
-----
    target = make_linear_regression(X, y)
    sampler = AutomaticBoomerangSampler(
        grad_target=target.grad_target, D=target.D, thinning="pli"
    )
    sampler.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
"""

import math
from typing import Optional, Literal
import torch
from torch import Tensor

from .smoke_test import TorchTarget
from ..utils.warmup import find_reference_glm   # reuse the Adam warm-start


# ===========================================================================
# Helpers
# ===========================================================================

def _prepend_intercept(X: Tensor) -> Tensor:
    """Add a column of 1s at position 0 for the implicit intercept."""
    N = X.shape[0]
    ones = torch.ones(N, 1, dtype=X.dtype, device=X.device)
    return torch.cat([ones, X], dim=1)


def _prior_precision_diag(
    D: int,
    prior_std: float,
    intercept_prior_std: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Diagonal prior precision: very loose on coord 0 (intercept),
    tight on the rest.
    """
    prec = torch.full((D,), 1.0 / prior_std ** 2, dtype=dtype, device=device)
    prec[0] = 1.0 / intercept_prior_std ** 2
    return prec


def make_kappa_vector_glm(
    D: int,
    kappa_coef: float = 1.0,
    kappa_intercept: float = 1e6,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Per-coordinate kappa for GLM-type targets with implicit intercept.

    Position 0 corresponds to the intercept (always thaws instantly —
    should never be subject to sparsity). Positions 1..D-1 are the
    feature coefficients, sticky with `kappa_coef`.
    """
    kappa = torch.full((D,), kappa_coef, dtype=dtype, device=device)
    kappa[0] = kappa_intercept
    return kappa


# ===========================================================================
# Linear regression
# ===========================================================================

def make_linear_regression(
    X: Tensor,
    y: Tensor,
    prior_std: float = 1.0,
    intercept_prior_std: float = 10.0,
    noise_std: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    adam_steps: int = 1000,
    adam_lr: float = 1e-2,
    diagonal_only: bool = True
) -> TorchTarget:
    """
    Bayesian linear regression with Gaussian prior.

        y = X_aug @ beta + noise,  noise ~ N(0, noise_std^2)
        beta_0 ~ N(0, intercept_prior_std^2)   (intercept, loose)
        beta_j ~ N(0, prior_std^2)             (j >= 1, tight)

    X_aug is X with a column of 1s prepended for the intercept.

    Energy (closed form):
        E(beta) = 0.5/noise_std^2 * ||y - X_aug @ beta||^2
                + 0.5 * sum_j (prec_j) * beta_j^2
    """
    X = X.to(dtype=dtype, device=device)
    y = y.to(dtype=dtype, device=device)
    X_aug = _prepend_intercept(X)

    D = X_aug.shape[1]
    prec = _prior_precision_diag(D, prior_std, intercept_prior_std, dtype, device)

    def energy_fn(beta: Tensor) -> Tensor:
        residual = y - X_aug @ beta
        neg_log_lik = 0.5 * (residual ** 2).sum() / noise_std ** 2
        neg_log_prior = 0.5 * (prec * beta ** 2).sum()
        return neg_log_lik + neg_log_prior

    def grad_target(beta: Tensor) -> Tensor:
        # Closed-form gradient — no autograd needed
        residual = y - X_aug @ beta
        grad_lik = -X_aug.T @ residual / noise_std ** 2
        grad_prior = prec * beta
        return grad_lik + grad_prior

    print(f"Finding reference for linear regression (D={D}, {adam_steps} Adam steps)...")
    x_ref, Sigma_inv = find_reference_glm(
        energy_fn, D, dtype=dtype, device=device, diagonal_only=diagonal_only,
        n_steps=adam_steps, lr=adam_lr,
    )

    return TorchTarget(
        name=f"linear_regression_D{D-1}",
        D=D,
        grad_target=grad_target,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={
            "task": "regression",
            "link": "identity",
            "prior_std": prior_std,
            "intercept_prior_std": intercept_prior_std,
            "noise_std": noise_std,
            "X_aug": X_aug,
            "y": y,
            "energy_fn": energy_fn,
            "has_intercept": True,
        },
    )


# ===========================================================================
# Logistic regression (binary + multinomial)
# ===========================================================================

def make_logistic_regression(
    X: Tensor,
    y: Tensor,
    prior_std: float = 1.0,
    intercept_prior_std: float = 10.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    adam_steps: int = 1000,
    adam_lr: float = 1e-2,
    diagonal_only: bool = True
) -> TorchTarget:
    """
    Bayesian logistic regression. Binary if y has 2 classes,
    multinomial (softmax) if y has >2.

    Parameters layout
    -----------------
    Binary (K=2):
        beta has shape (d+1,) — one intercept + d coefficients.
    Multinomial (K>2):
        beta has shape ((d+1) * K,) — K intercept+coefficient blocks
        flattened. Coefficient for class k, feature j, is at
        position k*(d+1) + j. Class 0 is the reference (softmax
        normalises so we still fit it).

    Prior: intercept entries loose, coefficient entries tight.
    """
    X = X.to(dtype=dtype, device=device)
    y = y.to(device=device)
    X_aug = _prepend_intercept(X)
    N, d_plus_1 = X_aug.shape

    n_classes = int(y.max().item()) + 1
    is_binary = n_classes == 2

    if is_binary:
        D = d_plus_1
    else:
        D = d_plus_1 * n_classes

    # Per-coordinate prior precision: every "intercept slot" (position 0
    # of each class block) gets the loose intercept prior.
    prec = torch.full((D,), 1.0 / prior_std ** 2, dtype=dtype, device=device)
    if is_binary:
        prec[0] = 1.0 / intercept_prior_std ** 2
    else:
        for k in range(n_classes):
            prec[k * d_plus_1] = 1.0 / intercept_prior_std ** 2

    y_float = y.to(dtype=dtype)
    y_long = y.long()

    def energy_fn(beta: Tensor) -> Tensor:
        if is_binary:
            logits = X_aug @ beta                              # [N]
            # Stable Bernoulli NLL
            neg_log_lik = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_float, reduction="sum"
            )
        else:
            # Reshape flat beta to [K, d+1]
            B = beta.reshape(n_classes, d_plus_1)
            logits = X_aug @ B.T                               # [N, K]
            neg_log_lik = torch.nn.functional.cross_entropy(
                logits, y_long, reduction="sum"
            )
        neg_log_prior = 0.5 * (prec * beta ** 2).sum()
        return neg_log_lik + neg_log_prior

    def grad_target(beta: Tensor) -> Tensor:
        # Closed-form gradient
        if is_binary:
            logits = X_aug @ beta
            probs = torch.sigmoid(logits)
            grad_lik = X_aug.T @ (probs - y_float)
        else:
            B = beta.reshape(n_classes, d_plus_1)
            logits = X_aug @ B.T                               # [N, K]
            probs = torch.softmax(logits, dim=-1)              # [N, K]
            # one-hot
            y_oh = torch.zeros_like(probs)
            y_oh.scatter_(1, y_long.unsqueeze(1), 1.0)
            # grad w.r.t. B is X_aug.T @ (probs - y_oh), shape [d+1, K]
            grad_B = X_aug.T @ (probs - y_oh)                  # [d+1, K]
            grad_lik = grad_B.T.reshape(-1)                    # flatten to [K*(d+1)]
        grad_prior = prec * beta
        return grad_lik + grad_prior

    print(f"Finding reference for logistic regression "
          f"(D={D}, {n_classes} classes, {adam_steps} Adam steps)...")
    x_ref, Sigma_inv = find_reference_glm(
        energy_fn, D, dtype=dtype, device=device, diagonal_only=diagonal_only,
        n_steps=adam_steps, lr=adam_lr,
    )

    return TorchTarget(
        name=f"logistic_regression_D{d_plus_1-1}_{n_classes}cls",
        D=D,
        grad_target=grad_target,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={
            "task": "classification",
            "link": "logit",
            "n_classes": n_classes,
            "d_plus_1": d_plus_1,
            "prior_std": prior_std,
            "intercept_prior_std": intercept_prior_std,
            "X_aug": X_aug,
            "y": y,
            "energy_fn": energy_fn,
            "has_intercept": True,
        },
    )


# ===========================================================================
# Generic GLM (Poisson, Gamma)
# ===========================================================================

def make_glm(
    X: Tensor,
    y: Tensor,
    family: Literal["poisson", "gamma"] = "poisson",
    prior_std: float = 1.0,
    intercept_prior_std: float = 10.0,
    gamma_shape: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    adam_steps: int = 1000,
    adam_lr: float = 1e-2,
) -> TorchTarget:
    """
    GLM with log link.

    family = "poisson":
        y ~ Poisson(mu),  mu = exp(X_aug @ beta)
        NLL = sum [ mu - y * (X_aug @ beta) ]   (dropping log y! constant)

    family = "gamma":
        y ~ Gamma(shape=gamma_shape, rate=shape/mu),  mu = exp(X_aug @ beta)
        NLL = sum [ shape * (X_aug @ beta) + shape * y / mu ]
             (dropping log y and shape terms independent of beta)

    The log link ensures mu > 0 without constraints on beta.
    """
    X = X.to(dtype=dtype, device=device)
    y = y.to(dtype=dtype, device=device)
    X_aug = _prepend_intercept(X)
    D = X_aug.shape[1]

    prec = _prior_precision_diag(D, prior_std, intercept_prior_std, dtype, device)

    if family == "poisson":
        if (y < 0).any():
            raise ValueError("Poisson requires non-negative y.")

        def energy_fn(beta: Tensor) -> Tensor:
            linpred = X_aug @ beta                  # [N]
            mu = torch.exp(linpred)
            neg_log_lik = (mu - y * linpred).sum()
            neg_log_prior = 0.5 * (prec * beta ** 2).sum()
            return neg_log_lik + neg_log_prior

        def grad_target(beta: Tensor) -> Tensor:
            linpred = X_aug @ beta
            mu = torch.exp(linpred)
            grad_lik = X_aug.T @ (mu - y)
            grad_prior = prec * beta
            return grad_lik + grad_prior

    elif family == "gamma":
        if (y <= 0).any():
            raise ValueError("Gamma requires strictly positive y.")

        s = gamma_shape

        def energy_fn(beta: Tensor) -> Tensor:
            linpred = X_aug @ beta
            mu = torch.exp(linpred)
            # NLL ∝ shape * linpred + shape * y / mu
            neg_log_lik = (s * linpred + s * y / mu).sum()
            neg_log_prior = 0.5 * (prec * beta ** 2).sum()
            return neg_log_lik + neg_log_prior

        def grad_target(beta: Tensor) -> Tensor:
            linpred = X_aug @ beta
            mu = torch.exp(linpred)
            # d/dbeta of (s * linpred + s*y/mu) = s * X_aug.T @ (1 - y/mu)
            grad_lik = s * X_aug.T @ (1.0 - y / mu)
            grad_prior = prec * beta
            return grad_lik + grad_prior

    else:
        raise ValueError(f"Unknown family '{family}'. Use 'poisson' or 'gamma'.")

    print(f"Finding reference for {family} GLM (D={D}, {adam_steps} Adam steps)...")
    x_ref, Sigma_inv = find_reference_glm(
        energy_fn, D, dtype=dtype, device=device,
        n_steps=adam_steps, lr=adam_lr,
    )

    return TorchTarget(
        name=f"glm_{family}_D{D-1}",
        D=D,
        grad_target=grad_target,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={
            "task": "regression" if family != "binomial" else "classification",
            "family": family,
            "link": "log",
            "prior_std": prior_std,
            "intercept_prior_std": intercept_prior_std,
            "gamma_shape": gamma_shape if family == "gamma" else None,
            "X_aug": X_aug,
            "y": y,
            "energy_fn": energy_fn,
            "has_intercept": True,
        },
    )


# ===========================================================================
# Prediction utilities
# ===========================================================================

@torch.no_grad()
def predict_linear(samples: Tensor, X_test: Tensor, target: TorchTarget):
    """Posterior predictive mean and std for linear regression."""
    X_aug = _prepend_intercept(X_test.to(dtype=samples.dtype, device=samples.device))
    preds = samples @ X_aug.T             # [S, N]
    return preds.mean(0), preds.std(0)


@torch.no_grad()
def predict_logistic(samples: Tensor, X_test: Tensor, target: TorchTarget):
    """Posterior predictive class probabilities and entropy."""
    X_aug = _prepend_intercept(X_test.to(dtype=samples.dtype, device=samples.device))
    n_classes = target.meta["n_classes"]

    if n_classes == 2:
        logits = samples @ X_aug.T                         # [S, N]
        p1 = torch.sigmoid(logits)
        p = torch.stack([1 - p1, p1], dim=-1).mean(0)      # [N, 2]
    else:
        d_plus_1 = target.meta["d_plus_1"]
        S = samples.shape[0]
        B = samples.reshape(S, n_classes, d_plus_1)         # [S, K, d+1]
        logits = torch.einsum("skj,nj->snk", B, X_aug)      # [S, N, K]
        probs = torch.softmax(logits, dim=-1)
        p = probs.mean(0)                                   # [N, K]

    entropy = -(p * p.clamp(min=1e-12).log()).sum(-1)
    return p, entropy


@torch.no_grad()
def predict_glm(samples: Tensor, X_test: Tensor, target: TorchTarget):
    """Posterior predictive mean (on response scale) for GLMs with log link."""
    X_aug = _prepend_intercept(X_test.to(dtype=samples.dtype, device=samples.device))
    linpred = samples @ X_aug.T           # [S, N]
    mu = torch.exp(linpred)               # log link
    return mu.mean(0), mu.std(0)
