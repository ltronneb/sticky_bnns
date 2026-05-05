"""Posterior-predictive metrics and chain diagnostics.

Pure tensor-level functions. Nothing here knows about BNNs, GLMs, or any
specific target structure — all functions take tensors and return tensors
or floats. The "predict from samples" step lives in each target's
prediction utilities (e.g. predict_regression in bnn_torch.py); this
module just scores the predictions.

Usage:
    from sazz.utils.metrics import (
        rmse, gaussian_log_lik, classification_log_lik,
        accuracy, ess_per_coord,
    )

    # 1) Get predictions however you like (BNN, GLM, ...):
    mean_pred, std_pred = predict_regression(samples, X_test, target)

    # 2) Score them:
    r = rmse(y_test, mean_pred)
    ll = gaussian_log_lik(y_test, mean_pred, std_pred, noise_std)
    ess = ess_per_coord(samples)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


# ===========================================================================
# Regression metrics
# ===========================================================================

def rmse(y_true: Tensor, mean_pred: Tensor) -> Tensor:
    """Root-mean-square error of point predictions."""
    return ((mean_pred - y_true) ** 2).mean().sqrt()


def gaussian_log_lik(y_true: Tensor, mean_pred: Tensor,
                     pred_std: Tensor, noise_std: float) -> Tensor:
    """Mean per-point log-likelihood under N(mean_pred, pred_std^2 + noise_std^2).

    The total predictive variance combines epistemic (pred_std from the
    posterior over functions) and aleatoric (noise_std from the likelihood
    model) uncertainty. Higher is better.
    """
    total_std = (pred_std ** 2 + noise_std ** 2).sqrt()
    return (
        -0.5 * ((y_true - mean_pred) / total_std) ** 2
        - total_std.log()
        - 0.5 * math.log(2 * math.pi)
    ).mean()


# ===========================================================================
# Classification metrics
# ===========================================================================

def accuracy(y_true: Tensor, probs: Tensor) -> Tensor:
    """Top-1 accuracy. probs is [N, n_classes]."""
    return (probs.argmax(-1) == y_true.long()).float().mean()


def classification_log_lik(y_true: Tensor, probs: Tensor,
                           eps: float = 1e-12) -> Tensor:
    """Mean per-point log-likelihood log p(y_true | probs).

    probs is [N, n_classes]. Higher is better.
    """
    p_true = probs.gather(-1, y_true.long().unsqueeze(-1)).squeeze(-1)
    return p_true.clamp(min=eps).log().mean()


def predictive_entropy(probs: Tensor, eps: float = 1e-12) -> Tensor:
    """Mean predictive entropy across test points. Useful as a calibration /
    uncertainty-magnitude diagnostic."""
    return -(probs * probs.clamp(min=eps).log()).sum(-1).mean()


# ===========================================================================
# Chain diagnostics
# ===========================================================================

def ess_per_coord(samples: Tensor, max_lag: Optional[int] = None) -> Tensor:
    """Initial-positive-sequence ESS per coordinate (Geyer's IPS truncation).

    samples is [n, d]. Returns an [d] tensor of effective sample sizes,
    one per coordinate. Capped below at 1 to avoid pathologies on chains
    where the autocorrelation never drops.
    """
    x = samples - samples.mean(0, keepdim=True)
    n, d = x.shape
    var = (x ** 2).mean(0)
    if max_lag is None:
        max_lag = min(n - 1, 1000)

    rho_sum = torch.zeros(d, dtype=samples.dtype)
    prev_pair = torch.full((d,), float("inf"), dtype=samples.dtype)
    active = torch.ones(d, dtype=torch.bool)

    k = 1
    while k + 1 <= max_lag:
        c_k   = (x[:n - k]     * x[k:]).mean(0)     / var.clamp(min=1e-30)
        c_kp1 = (x[:n - k - 1] * x[k + 1:]).mean(0) / var.clamp(min=1e-30)
        pair = c_k + c_kp1
        kill = active & ((pair <= 0) | (pair >= prev_pair))
        active = active & ~kill
        rho_sum = rho_sum + torch.where(active, pair, torch.zeros_like(pair))
        prev_pair = torch.where(active, pair, prev_pair)
        if not active.any():
            break
        k += 2

    tau = 1.0 + 2.0 * rho_sum
    return n / tau.clamp(min=1.0)


def ess_summary(samples: Tensor) -> dict[str, float]:
    """Convenience: per-chain summary of per-coord ESS."""
    ess = ess_per_coord(samples)
    return {
        "ess_min":    float(ess.min()),
        "ess_median": float(ess.median()),
        "ess_mean":   float(ess.mean()),
    }


# ===========================================================================
# High-level convenience wrappers
# ===========================================================================

def regression_metrics(y_true: Tensor, mean_pred: Tensor, pred_std: Tensor,
                       noise_std: float, y_std: float = 1.0,
                       samples: Optional[Tensor] = None) -> dict[str, float]:
    """All regression-side metrics in one call.

    `y_std` is the standard deviation used to standardise the target during
    preprocessing — pass 1.0 if the data is unstandardised. `rmse_orig`
    and `nll_orig` give you the metrics on the original scale.

    Pass `samples` to also include ESS diagnostics.
    """
    rmse_std = rmse(y_true, mean_pred)
    log_lik  = gaussian_log_lik(y_true, mean_pred, pred_std, noise_std)
    out = {
        "rmse_std":      float(rmse_std),
        "rmse_orig":     float(rmse_std) * y_std,
        "log_lik":       float(log_lik),
        "nll_orig":      -float(log_lik) + math.log(y_std),
        "pred_std_mean": float(pred_std.mean()),
    }
    if samples is not None:
        out.update(ess_summary(samples))
    return out


def classification_metrics(y_true: Tensor, probs: Tensor,
                           samples: Optional[Tensor] = None
                           ) -> dict[str, float]:
    """All classification-side metrics in one call."""
    out = {
        "accuracy": float(accuracy(y_true, probs)),
        "log_lik":  float(classification_log_lik(y_true, probs)),
        "entropy":  float(predictive_entropy(probs)),
    }
    if samples is not None:
        out.update(ess_summary(samples))
    return out

