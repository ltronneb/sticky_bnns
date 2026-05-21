"""
utils/warmup.py

Two ways to find a reference (x_ref, Sigma_inv) for the sampler:

1. find_reference : Adam-based MAP + diagonal Fisher. 
                    No sampler needed, cheap.
2. warmup         : Iterative pilot runs with the sampler itself.
                    Refines an existing reference using samples from 
                    the target.

Typical pipeline:
    target = make_bnn_regression(X, y, ...)   # calls find_reference internally
    sampler = AutomaticBoomerangSampler(...)
    warmup(sampler, target=target, n_rounds=3)  # refines further
    result = sampler.sample(N=...)
"""

import numpy as np
import torch
import math
from torch import Tensor
from typing import Optional, Literal
from ..models.model import BayesianModel

def find_reference_glm(
    energy_fn,
    D: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    n_steps: int = 1000,
    lr: float = 1e-2,
    prec_min: float = 1e-6,
    prec_max: float = 1e8,
    diagonal_only: bool = False,
    jitter: float = 1e-8,
    model: Optional["BayesianModel"] = None,
) -> tuple[Tensor, Tensor]:
    """
    MAP via Adam + Hessian at the MAP.

    If `model` is provided, uses model.energy for MAP finding and can
    exploit model.prior.precision_diag() for the prior contribution.
    Otherwise falls back to energy_fn.
    """
    fn = model.energy if model is not None else energy_fn

    # MAP via Adam
    beta = torch.randn(D, dtype=dtype, device=device) * 0.01
    beta.requires_grad_(True)
    optimizer = torch.optim.Adam([beta], lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = fn(beta)
        loss.backward()
        optimizer.step()
    x_ref = beta.detach().clone()

    if diagonal_only:
        diag_H = torch.zeros(D, dtype=dtype, device=device)
        for i in range(D):
            b = x_ref.clone().requires_grad_(True)
            g, = torch.autograd.grad(fn(b), b, create_graph=True)
            hi, = torch.autograd.grad(g[i], b, retain_graph=False)
            diag_H[i] = hi[i]
        diag_prec = diag_H.clamp(min=prec_min, max=prec_max)
        Sigma_inv = torch.diag(diag_prec)
    else:
        H = torch.autograd.functional.hessian(fn, x_ref)
        H = 0.5 * (H + H.T) + jitter * torch.eye(D, dtype=dtype, device=device)
        Sigma_inv = H

    return x_ref, Sigma_inv


def find_reference_bnn(
    energy_fn,
    D: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    n_steps: int = 4000,
    lr: float = 1e-2,
    model: Optional["BayesianModel"] = None,
    reference: Literal["prior", "laplace_diag", "adam"] = "laplace_diag",
    n_fisher_batch: int = 64,
    floor_eps: float = 1e-8,
    #init_beta=None, 
) -> tuple[Tensor, Tensor]:
    """
    MAP via Adam + choice of diagonal reference precision for BNNs.

    All three options return a *diagonal* Sigma_inv. The full-Hessian
    branches were removed: for BNNs the full Hessian is both expensive
    and unreliable (near-singular directions from symmetry, locality
    around a single mode), and the right object — the diagonal posterior
    precision — is what the dynamics actually use for per-coordinate
    velocity scaling.

    Parameters
    ----------
    reference : {"prior", "laplace_diag", "adam"}
        "prior"        — ignore the data; use the diagonal prior precision
                         as Sigma_inv. Robust, no tuning, and a sensible
                         baseline for highly overparameterised BNNs where
                         most coordinates are barely informed by the data.

        "laplace_diag" — diagonal Laplace approximation. Adds the prior
                         precision and the diagonal empirical Fisher at
                         the MAP:
                             Sigma_inv_ii = prior_prec_i
                                          + (1/N) * sum_n (d log p(y_n|...) / d beta_i)^2
                         This is the principled choice: each term has a
                         clear meaning, the result is positive by
                         construction, and it reduces to "prior" when the
                         data is uninformative. Recommended default.

        "adam"         — heuristic. Uses Adam's bias-corrected second-
                         moment estimate v_hat as a per-coordinate
                         precision proxy, floored by the prior. Not a
                         principled estimator of any well-defined
                         quantity, but works well in practice as a
                         per-coordinate scaling and is essentially free
                         (already maintained by the optimiser).

    n_fisher_batch : int
        Number of data points used to estimate the empirical Fisher
        diagonal. Only used when reference="laplace_diag". The Fisher
        is averaged over this many independent data points (sampled
        without replacement from the likelihood's stored data).

    floor_eps : float
        Numerical safety floor on the precision diagonal. Should
        essentially never bind in practice (the prior contribution is
        already strictly positive).
    """
    fn = model.energy if model is not None else energy_fn

    if reference != "adam" and model is None:
        raise ValueError(
            f"reference='{reference}' requires a model argument "
            "(prior precision is needed)."
        )

    # --- MAP via Adam ---
    if model is not None and hasattr(model.likelihood, "module"):
        init_module_beta = torch.cat([
            p.detach().flatten() for p in model.likelihood.module.parameters()
        ]).to(dtype=dtype, device=device)
        
        if init_module_beta.numel() == D:
            beta = init_module_beta
        elif init_module_beta.numel() < D:
            # Learned-noise case: pad with zeros for log_sigma (and any future
            # extra latents). exp(0) = 1, sensible starting noise scale.
            pad = torch.zeros(D - init_module_beta.numel(), dtype=dtype, device=device)
            beta = torch.cat([init_module_beta, pad])
        else:
            raise ValueError(
                f"Module has more parameters ({init_module_beta.numel()}) than "
                f"the sampler dimension D={D}."
            )
    else:
        beta = torch.randn(D, dtype=dtype, device=device)

    beta.requires_grad_(True)
    #beta = torch.randn(D, dtype=dtype, device=device) #* 0.01
    #beta.requires_grad_(True)
    optimizer = torch.optim.Adam([beta], lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = fn(beta)
        loss.backward()
        optimizer.step()
    x_ref = beta.detach().clone()

    # --- Diagonal precision ---
    if reference == "prior":
        diag_prec = model.prior.precision_diag().to(dtype=dtype, device=device)

    elif reference == "laplace_diag":
        prior_prec = model.prior.precision_diag().to(dtype=dtype, device=device)
        fisher_diag = _empirical_fisher_diag(
            model, x_ref, n_fisher_batch, dtype, device
        )
        diag_prec = prior_prec + fisher_diag

    elif reference == "adam":
        # Adam's bias-corrected second moment as a curvature proxy.
        state = optimizer.state[beta]
        v_hat = state["exp_avg_sq"].clone()
        step = state["step"]
        if isinstance(step, torch.Tensor):
            step = step.item()
        beta2 = optimizer.defaults["betas"][1]
        bias_correction = 1.0 - beta2 ** step
        v_hat = v_hat / bias_correction

        # Floor by prior precision so flat directions inherit the prior
        # scale rather than blowing up.
        if model is not None:
            prior_prec = model.prior.precision_diag().to(dtype=dtype, device=device)
            diag_prec = torch.maximum(v_hat, prior_prec)
        else:
            diag_prec = v_hat

    else:
        raise ValueError(
            f"Unknown reference type: {reference!r}. "
            "Choose from 'prior', 'laplace_diag', 'adam'."
        )

    diag_prec = diag_prec.clamp(min=floor_eps)
    # Return as 1-D diagonal vector — preprocess handles both 1-D and 2-D.
    # Avoids allocating a [D, D] matrix (critical for large D like CNNs).
    # Original: Sigma_inv = torch.diag(diag_prec)
    return x_ref, diag_prec


def _empirical_fisher_diag(
    model: "BayesianModel",
    x_ref: Tensor,
    n_batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Diagonal of the empirical Fisher information at x_ref.

        F_ii = (1/N) * sum_n (d/dbeta_i log p(y_n | x_n, beta_ref))^2

    Uses the likelihood's stored (X, y) buffers. Subsamples n_batch
    points without replacement. Returns a length-D tensor.
    """
    likelihood = model.likelihood
    X = likelihood.X
    y = likelihood.y
    N = X.shape[0]

    # Subsample without replacement
    n = min(n_batch, N)
    idx = torch.randperm(N, device=device)[:n]

    D = x_ref.shape[0]
    fisher = torch.zeros(D, dtype=dtype, device=device)

    for i in idx:
        b = x_ref.clone().requires_grad_(True)
        log_p_n = likelihood.log_prob_single(b, X[i:i+1], y[i:i+1])
        g, = torch.autograd.grad(log_p_n, b)
        fisher += g.detach() ** 2

    fisher /= n
    return fisher



def warmup(sampler, n_rounds=3, n_pilot=500, target=None, zero_tol=1e-8):
    """
    Refine (x_ref, Sigma_inv) from short pilot runs of the sampler.

    Works for both regular and sticky samplers. For sticky, only updates
    coordinates that are active often enough for reliable moment estimation.
    """
    dtype = sampler.dtype
    device = sampler.device
    D = sampler.D

    is_sticky = hasattr(sampler, "frozen_mask")

    # Round 0: initial reference (from target, or default)
    if sampler.x_ref is None:
        if target is not None and target.x_ref is not None:
            sampler.preprocess(
                x_ref=target.x_ref.to(dtype=dtype, device=device),
                Sigma_inv=target.Sigma_inv.to(dtype=dtype, device=device),
            )
        else:
            sampler.preprocess(
                x_ref=torch.zeros(D, dtype=dtype, device=device),
                Sigma_inv=torch.eye(D, dtype=dtype, device=device),
            )

    for _ in range(n_rounds):
        result = sampler.sample(N=n_pilot, diagnostics=False)

        pos_np = result["positions"].cpu().numpy()
        vel_np = result["velocities"].cpu().numpy()
        tim_np = result["times"].cpu().numpy()
        x_ref_np = sampler.x_ref.cpu().numpy()

        if is_sticky:
            from .sampling import resample_boomerang_path_sticky as rsm
        else:
            from .sampling import resample_boomerang_path as rsm

        samples = rsm(pos_np, vel_np, tim_np, x_ref_np,
                      N_resample=n_pilot, burnin_frac=0.2)

        # Only update coordinates with enough active samples
        if is_sticky:
            frac_active = np.mean(np.abs(samples) > zero_tol, axis=0)
            update_mask = frac_active > 0.1
        else:
            update_mask = np.ones(D, dtype=bool)

        x_ref_new = x_ref_np.copy()

        # Build a 2-D Sigma_inv matrix to hold the full-covariance update.
        # sampler.Sigma_inv may be 1-D (diagonal path) or 2-D (full-matrix path).
        raw = sampler.Sigma_inv.cpu().numpy()
        Sigma_inv_full = np.diag(raw) if raw.ndim == 1 else raw.copy()

        if update_mask.any():
            active_samples = samples[:, update_mask]
            x_ref_new[update_mask] = active_samples.mean(axis=0)

            cov = np.cov(active_samples.T)  # shape [n_active, n_active]
            cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
            Sigma_inv_block = np.linalg.inv(cov_reg)

            active_idx = np.where(update_mask)[0]
            Sigma_inv_full[np.ix_(active_idx, active_idx)] = Sigma_inv_block

        sampler.preprocess(
            x_ref=torch.tensor(x_ref_new, dtype=dtype, device=device),
            Sigma_inv=torch.tensor(Sigma_inv_full, dtype=dtype, device=device),
        )



def tune_refresh_rate(
    sampler,
    n_pilot: int = 500,
    floor: float = 1.0 / math.pi,
    target_ratio: float = 0.7812,
):
    """
    Tune sampler.refresh_rate via a single pilot run.

    Sets
        lambda_r = max( target_ratio / (1 - target_ratio) * lambda_hat,
                        floor )

    where lambda_hat = (number of accepted reflections) / (final pilot time).

    The target_ratio = 0.7812 figure is the BPS optimal refresh-to-event
    ratio from Bouchard-Côté et al. (2018), borrowed here as a heuristic
    for the Boomerang. The geometric floor 1/pi is Boomerang-specific:
    refreshment must occur before the trajectory completes a half-orbit
    (Section 5.4 of the docs), otherwise bounces within an orbit can
    cancel and the sampler degenerates toward the reference measure.

    Sticky-aware: freeze and thaw events are NOT counted as reflections.
    Only velocity-flip events ('bounce' with accepted=True) contribute to
    lambda_hat.

    Parameters
    ----------
    sampler : an Automatic{Sticky}BoomerangSampler
        Must already have x_ref / Sigma_inv set (call preprocess first).
    n_pilot : int
        Number of skeleton points in the pilot run.
    floor : float
        Lower bound on lambda_r. Default 1/pi.
    target_ratio : float
        Desired lambda_r / (lambda_r + lambda_hat). Default 0.7812.

    Returns
    -------
    dict with keys:
        "lambda_r_old"   : refresh rate before adaptation
        "lambda_r_new"   : refresh rate after adaptation
        "lambda_hat"     : empirical reflection rate
        "n_bounces"      : number of accepted reflection events
        "T_pilot"        : final simulation time of the pilot run
        "floor_active"   : True if the geometric floor was hit
    """
    assert sampler.x_ref is not None, "Call preprocess() before tuning."

    lambda_r_old = sampler.refresh_rate
    result = sampler.sample(N=n_pilot, diagnostics=False)

    # Count accepted reflections (NOT freeze, thaw, refresh, or rejected bounces)
    diag = result["diagnostics"]
    n_bounces = sum(
        1 for row in diag
        if row["event_type"] == "bounce" and row["accepted"] is True
    )

    T_pilot = float(result["times"][-1])
    if T_pilot <= 0.0:
        raise RuntimeError("Pilot run produced no simulation time.")

    lambda_hat = n_bounces / T_pilot
    lambda_bps = (target_ratio / (1.0 - target_ratio)) * lambda_hat
    lambda_r_new = max(lambda_bps, floor)
    floor_active = lambda_bps < floor

    sampler.refresh_rate = lambda_r_new
    return {
        "lambda_r_old": lambda_r_old,
        "lambda_r_new": lambda_r_new,
        "lambda_hat": lambda_hat,
        "n_bounces": n_bounces,
        "T_pilot": T_pilot,
        "floor_active": floor_active,
    }
     
