"""Reference-measure finder for BayesianModule targets: MAP via Adam, then
diagonal Sigma_inv = prior precision + empirical Fisher at the MAP (same
math as sazz.utils.warmup's "laplace_diag" option, reimplemented against
BayesianModule's plain attributes to stay independent of BayesianModel)."""

import torch
from torch import Tensor

from ..models.model import BayesianModule


def find_reference_bnn(
    bm: BayesianModule,
    n_steps: int = 4000,
    lr: float = 1e-2,
    n_fisher_batch: int = 128,
    floor_eps: float = 1e-8,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, Tensor]:
    """
    MAP via Adam, then diagonal Laplace approx:
        Sigma_inv_ii = prior_prec_i + (1/n) sum (d log p(y_i|beta) / d beta_i)^2
    with the Fisher term estimated from n_fisher_batch data points.

    If bm.learns_noise, the log_sigma coordinate's entry is further
    patched with its exact analytic prior curvature (the empirical Fisher
    estimate is noisy for that one coordinate; the true curvature is known
    in closed form) -- matches sazz/scripts/bnns/toy_bnn.py's
    build_target_learned_noise.

    Returns (x_ref, diag_prec).
    """
    beta = torch.randn(bm.D, dtype=dtype, device=device)
    beta.requires_grad_(True)
    optimizer = torch.optim.Adam([beta], lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = bm.energy(beta)
        loss.backward()
        optimizer.step()
    x_ref = beta.detach().clone()

    prior_prec = bm.prior_precision.to(dtype=dtype, device=device)
    fisher_diag = _empirical_fisher_diag(bm, x_ref, n_fisher_batch, dtype, device)
    diag_prec = (prior_prec + fisher_diag).clamp(min=floor_eps)

    if bm.learns_noise:
        sigma_map = x_ref[-1].exp()
        prior_curvature = 2.0 * sigma_map ** 2 / bm.prior_sigma_scale ** 2
        diag_prec[-1] = diag_prec[-1] + prior_curvature

    return x_ref, diag_prec


def _empirical_fisher_diag(
    bm: BayesianModule, x_ref: Tensor, n_batch: int,
    dtype: torch.dtype, device: torch.device,
) -> Tensor:
    """Diagonal empirical Fisher at x_ref, subsampling n_batch points from bm.X/bm.y."""
    X, y = bm.X, bm.y
    N = X.shape[0]
    n = min(n_batch, N)
    idx = torch.randperm(N, device=device)[:n]

    D = x_ref.shape[0]
    fisher = torch.zeros(D, dtype=dtype, device=device)

    log_prob_single = bm.log_likelihood.single

    for i in idx:
        b = x_ref.clone().requires_grad_(True)
        log_p_i = log_prob_single(b, X[i:i + 1], y[i:i + 1])
        g, = torch.autograd.grad(log_p_i, b)
        fisher += g.detach() ** 2

    fisher /= n
    return fisher
