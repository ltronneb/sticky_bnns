"""Likelihoods as pure functions of a flat parameter vector `beta`, built on
torch.func.functional_call so they compose with torch.func.grad/vmap/jvp."""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, Bernoulli, Categorical


def make_gaussian_likelihood(module: nn.Module, param_dict_fn, X: Tensor,
                              y: Tensor, noise_std: Optional[float|Tensor] = None,
                              prior_sigma_scale: float = 1.0):
    """
    Gaussian regression likelihood via functional_call.

    noise_std: fixed observation noise if given. If None (default), noise
    is LEARNED -- beta carries an extra `log_sigma` as its last coordinate
    (sigma = exp(log_sigma)) with a HalfNormal(prior_sigma_scale) prior on
    sigma; param_dict_fn then only ever sees beta[:-1] (network weights).

    Returns log_prob(beta) -> scalar, plus a `.single(beta, X_i, y_i)`
    per-point variant used by warmup.py's empirical Fisher estimate.
    """
    if noise_std is not None:
        noise_std = torch.as_tensor(noise_std, dtype=X.dtype, device=X.device)
        def log_prob_single(beta, X_i, y_i):
            preds = torch.func.functional_call(module, param_dict_fn(beta), (X_i,)).squeeze(-1)
            return Normal(preds, noise_std, validate_args=False).log_prob(y_i).sum()
    else:
        prior_sigma_scale_t = torch.as_tensor(prior_sigma_scale, dtype=X.dtype, device=X.device)

        def log_prob_single(beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
            weights, log_sigma = beta[:-1], beta[-1]
            sigma = log_sigma.exp()
            preds = torch.func.functional_call(module, param_dict_fn(weights), (X_i,)).squeeze(-1)
            log_lik = Normal(preds, sigma, validate_args=False).log_prob(y_i).sum()
            # HalfNormal(prior_sigma_scale) on sigma, with Jacobian to log_sigma:
            log_prior = -0.5 * (sigma / prior_sigma_scale_t) ** 2 + log_sigma
            return log_lik + log_prior

    def log_prob(beta: Tensor) -> Tensor:
        return log_prob_single(beta, X, y)

    log_prob.single = log_prob_single
    return log_prob


def make_bernoulli_likelihood(module: nn.Module, param_dict_fn, X: Tensor,
                               y: Tensor):
    """Binary classification likelihood. `module` returns logits, shape [N] or [N, 1]."""
    y = y.to(X.dtype)

    def log_prob_single(beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        logits = torch.func.functional_call(module, param_dict_fn(beta), (X_i,)).squeeze(-1)
        return Bernoulli(logits=logits).log_prob(y_i).sum()

    def log_prob(beta: Tensor) -> Tensor:
        return log_prob_single(beta, X, y)

    log_prob.single = log_prob_single
    return log_prob


def make_categorical_likelihood(module: nn.Module, param_dict_fn, X: Tensor,
                                 y: Tensor):
    """Multiclass classification likelihood. `module` returns logits, shape [N, n_classes]."""
    y = y.long()

    def log_prob_single(beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        logits = torch.func.functional_call(module, param_dict_fn(beta), (X_i,))
        return Categorical(logits=logits, validate_args=False).log_prob(y_i.long()).sum()

    def log_prob(beta: Tensor) -> Tensor:
        return log_prob_single(beta, X, y)

    log_prob.single = log_prob_single
    return log_prob
