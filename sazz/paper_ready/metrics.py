"""Predictive metrics of the paper.

Regression. The posterior predictive at x is the equally weighted Gaussian
mixture (1/S) sum_s N(y; f_s(x), sigma_s^2) over the S draws. RMSE is of the
predictive mean, NLL and CRPS of the mixture, coverage of its central 90%
interval. All on the original y scale (the data are standardized, y_std undoes it).

Classification. Probabilities are averaged over draws (Bayesian model average),
then accuracy, NLL, ECE (15 equal-width confidence bins) and the Brier score.
"""

import math

import torch
from torch import Tensor
from torch.distributions import Normal

CRPS_SUBSAMPLE = 256
COVERAGE_LEVEL = 0.9
ECE_BINS = 15


# ------------------------------------------------------------- regression
def rmse(y: Tensor, preds: Tensor, y_std: float) -> float:
    """preds [S, N], y [N], both standardized."""
    return float(((preds.mean(0) - y) ** 2).mean().sqrt()) * y_std


def nll_mixture(y: Tensor, preds: Tensor, sigma: Tensor, y_std: float) -> float:
    lp = Normal(preds, sigma[:, None]).log_prob(y)                   # [S, N]
    ll = torch.logsumexp(lp, dim=0) - math.log(preds.shape[0])       # [N]
    return float(-ll.mean() + math.log(y_std))


def _crps_A(m: Tensor, s: Tensor) -> Tensor:
    """E|X - m| for X ~ N(0, s^2)."""
    z = m / s
    d = Normal(0.0, 1.0)
    return m * (2 * d.cdf(z) - 1) + 2 * s * d.log_prob(z).exp()


def crps_mixture(y: Tensor, preds: Tensor, sigma: Tensor, y_std: float,
                 n_sub: int = CRPS_SUBSAMPLE, seed: int = 0) -> float:
    """Closed-form CRPS of the Gaussian mixture on a fixed subsample of n_sub draws
    (same seed for every sampler):
    E_s A(y - mu_s, sigma_s) - 0.5 E_{s,s'} A(mu_s - mu_s', sqrt(sigma_s^2 + sigma_s'^2))."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(preds.shape[0], generator=g)[:n_sub]
    mu = preds[idx] * y_std
    sd = (sigma[idx] * y_std)[:, None].expand_as(mu)
    yt = y * y_std
    term1 = _crps_A(yt[None, :] - mu, sd).mean(0)
    term2 = _crps_A(mu[:, None, :] - mu[None, :, :],
                    (sd[:, None, :] ** 2 + sd[None, :, :] ** 2).sqrt()).mean((0, 1))
    return float((term1 - 0.5 * term2).mean())


def coverage(y: Tensor, preds: Tensor, sigma: Tensor, level: float = COVERAGE_LEVEL) -> float:
    """Share of test points with F(y) in [(1 - level)/2, (1 + level)/2], F the predictive
    CDF. Equivalent to y lying in the central `level` interval of the mixture."""
    u = Normal(0.0, 1.0).cdf((y[None, :] - preds) / sigma[:, None]).mean(0)
    return float(((u - 0.5).abs() <= level / 2).double().mean())


def regression_metrics(y: Tensor, preds: Tensor, sigma: Tensor, y_std: float) -> dict:
    return {"RMSE": rmse(y, preds, y_std), "NLL": nll_mixture(y, preds, sigma, y_std),
            "CRPS": crps_mixture(y, preds, sigma, y_std), "Cov": coverage(y, preds, sigma)}


# --------------------------------------------------------- classification
def classification_metrics(y: Tensor, probs: Tensor, n_bins: int = ECE_BINS) -> dict:
    """probs [N, C], the model-averaged class probabilities."""
    y = y.long()
    conf, pred = probs.max(-1)
    correct = (pred == y).double()
    nll = -probs.gather(-1, y[:, None]).squeeze(-1).clamp_min(1e-12).log().mean()
    brier = ((probs - torch.nn.functional.one_hot(y, probs.shape[-1])) ** 2).sum(-1).mean()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += float(m.double().mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return {"Acc": float(correct.mean()), "NLL": float(nll), "ECE": ece, "Brier": float(brier)}


# --------------------------------------------------------------- sparsity
def sparsity(samples: Tensor, freezable_mask: Tensor) -> float:
    """Percentage of the freezable weights (no biases, BatchNorm or log sigma) that are
    exactly zero, averaged over draws."""
    return 100 * float((samples[:, freezable_mask] == 0).double().mean())
