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
        Sigma_inv_ii = prior_prec_i + N * mean_{n_fisher_batch} (d log p(y_i|beta) / d beta_i)^2
    i.e. the Fisher term is a full-data-sum estimate (matching the Hessian of
    bm.energy, whose log_likelihood sums over all N points), obtained by
    scaling an n_fisher_batch-point subsample mean up by N.

    If bm.learns_noise, the log_sigma coordinate's entry is REPLACED (not
    added to) with its PRIOR-ONLY curvature (2*sigma_map^2/prior_sigma_scale^2)
    -- NOT likelihood + prior. The likelihood curvature at the MAP
    (2*resid_sq_sum/sigma_map^2, ~910 for boston D=752/N=455) is the exact
    analytic Fisher info for log_sigma held fixed, but that's a CONDITIONAL
    curvature -- it ignores log_sigma's correlation with the network-weight
    directions (with D >> N, sigma trades off against an effective-
    complexity direction in weight space), so a diagonal Laplace built from
    it is severely overconfident: checked against a real NUTS posterior on
    boston, the true posterior mean for sigma sat ~11 of that conditional
    std away from x_ref, while Boomerang's excess-potential correction
    couldn't pull the chain that far, biasing its sigma estimate low
    (~0.23 vs NUTS's ~0.28) -- ZigZag has no reference-measure-coupled
    dynamics so it wasn't affected. A profile-likelihood check (re-optimizing
    weights at fixed log_sigma out to sigma=0.35) showed the true marginal
    curvature is watered down by a weight-Hessian volume/entropy term that
    grows with sigma -- not cheaply computable in closed form at D~750 -- so
    rather than chase that analytically, log_sigma's reference precision
    here is prior-only, giving it a wide, prior-controlled reference std
    (~1.0 vs ~0.03) so the sampler has room to find the true posterior
    rather than being pinned near an overconfident data-driven curvature.

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

    N = bm.X.shape[0]
    prior_prec = bm.prior_precision.to(dtype=dtype, device=device)
    fisher_diag = _empirical_fisher_diag(bm, x_ref, n_fisher_batch, dtype, device) * N
    diag_prec = (prior_prec + fisher_diag).clamp(min=floor_eps)

    if bm.learns_noise:
        sigma_map = x_ref[-1].exp()
        # Prior-only reference precision for log_sigma -- see docstring above
        # for why the likelihood-curvature term is deliberately dropped here.
        # Old (likelihood + prior) version kept for reference, not deleted:
        #
        # with torch.no_grad():
        #     preds = torch.func.functional_call(
        #         bm.module, bm.param_dict_fn(x_ref[:-1]), (bm.X,)
        #     ).squeeze(-1)
        #     resid_sq_sum = ((bm.y - preds) ** 2).sum()
        # likelihood_curvature = 2.0 * resid_sq_sum / sigma_map ** 2
        # prior_curvature = 2.0 * sigma_map ** 2 / bm.prior_sigma_scale ** 2
        # diag_prec[-1] = likelihood_curvature + prior_curvature
        prior_curvature = 2.0 * sigma_map ** 2 / bm.prior_sigma_scale ** 2
        diag_prec[-1] = prior_curvature

    return x_ref, diag_prec


def _empirical_fisher_diag(
    bm: BayesianModule, x_ref: Tensor, n_batch: int,
    dtype: torch.dtype, device: torch.device,
) -> Tensor:
    """Per-datapoint MEAN diagonal empirical Fisher at x_ref, subsampling
    n_batch points from bm.X/bm.y -- caller must scale by N to match the
    full-data-sum Hessian of bm.energy."""
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


# ===========================================================================
# GGN (Gauss-Newton) reference -- alternative to find_reference_bnn above.
# Same MAP, different curvature diagonal. Kept as a separate function (the
# original find_reference_bnn is byte-for-byte unchanged) so a caller picks
# which reference it wants.
# ===========================================================================

def _ggn_diag(
    bm: BayesianModule, x_ref: Tensor, n_batch: int,
    dtype: torch.dtype, device: torch.device,
) -> Tensor:
    """Per-datapoint MEAN diagonal Gauss-Newton curvature at x_ref for a
    Gaussian regression likelihood, subsampling n_batch points -- caller
    must scale by N to match the full-data-sum Hessian of bm.energy.

        GGN_ii = (1/sigma^2) * mean_n (df_n/dtheta_i)^2

    where f is the scalar network output (bm.module applied to one x_n via
    functional_call, weights = x_ref[:-1] when noise is learned) and sigma =
    exp(x_ref[-1]). Unlike the empirical Fisher, there is NO residual (y_n -
    f_n)^2 weighting, so well-fit points still contribute their full local
    curvature. The trailing log_sigma coordinate's entry is left at 0 here
    (df/dlog_sigma == 0); find_reference_bnn_ggn overwrites it with the
    prior-only curvature, same as find_reference_bnn.

    Requires bm.learns_noise (Gaussian + learned sigma). For a fixed-noise
    or non-Gaussian target, use find_reference_bnn (empirical Fisher).
    """
    if not bm.learns_noise:
        raise ValueError(
            "_ggn_diag assumes a Gaussian likelihood with learned noise "
            "(bm.learns_noise). Use find_reference_bnn for fixed-noise or "
            "non-Gaussian targets."
        )

    X, y = bm.X, bm.y
    N = X.shape[0]
    n = min(n_batch, N)
    idx = torch.randperm(N, device=device)[:n]

    D = x_ref.shape[0]
    weights_ref = x_ref[:-1].detach()
    sigma = x_ref[-1].detach().exp()
    inv_sigma2 = 1.0 / sigma ** 2

    ggn = torch.zeros(D, dtype=dtype, device=device)

    for i in idx:
        w = weights_ref.clone().requires_grad_(True)
        f_i = torch.func.functional_call(
            bm.module, bm.param_dict_fn(w), (X[i:i + 1],)
        ).squeeze()
        g, = torch.autograd.grad(f_i, w)
        ggn[:-1] += g.detach() ** 2

    ggn[:-1] *= inv_sigma2 / n
    return ggn


def find_reference_bnn_ggn(
    bm: BayesianModule,
    n_steps: int = 4000,
    lr: float = 1e-2,
    n_fisher_batch: int = 128,
    floor_eps: float = 1e-8,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, Tensor]:
    """
    Drop-in alternative to find_reference_bnn: identical MAP-via-Adam, then a
    diagonal Laplace approx built from the GAUSS-NEWTON diagonal instead of
    the empirical Fisher:

        Sigma_inv_ii = prior_prec_i + N * mean_{n_fisher_batch} (df_i/dtheta_i)^2 / sigma_map^2

    See _ggn_diag and this module's header for why GGN is preferred for the
    deep BNN variants (the empirical Fisher's residual^2 weighting makes it
    underestimate curvature near a mode, forcing the Boomerang reference-
    precision scale far above 1.0). The log_sigma coordinate is handled
    EXACTLY as in find_reference_bnn -- prior-only curvature
    (2*sigma_map^2/prior_sigma_scale^2), see that function's docstring for
    the full rationale.

    Returns (x_ref, diag_prec). Requires bm.learns_noise (Gaussian + learned
    sigma); raises otherwise.
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

    N = bm.X.shape[0]
    prior_prec = bm.prior_precision.to(dtype=dtype, device=device)
    ggn_diag = _ggn_diag(bm, x_ref, n_fisher_batch, dtype, device) * N
    diag_prec = (prior_prec + ggn_diag).clamp(min=floor_eps)

    sigma_map = x_ref[-1].exp()
    # Prior-only reference precision for log_sigma -- see find_reference_bnn's
    # docstring for why the likelihood-curvature term is deliberately dropped.
    prior_curvature = 2.0 * sigma_map ** 2 / bm.prior_sigma_scale ** 2
    diag_prec[-1] = prior_curvature

    return x_ref, diag_prec
