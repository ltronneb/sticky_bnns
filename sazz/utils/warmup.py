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
from torch import Tensor
from typing import Optional
from ..models.models_torch import BayesianModel

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

# def find_reference_glm(
#     energy_fn,
#     D: int,
#     dtype: torch.dtype = torch.float64,
#     device: torch.device = torch.device("cpu"),
#     n_steps: int = 1000,
#     lr: float = 1e-2,
#     prec_min: float = 1e-6,
#     prec_max: float = 1e8,
# ) -> tuple[Tensor, Tensor]:
#     """
#     MAP via Adam + diagonal Hessian at the MAP.

#     Exact Laplace-approximation precision for convex, smooth,
#     deterministic energies. Use for GLMs (linreg, logreg,
#     Poisson/Gamma) and similar low-to-moderate dimensional
#     closed-form targets.

#     Returns
#     -------
#     x_ref : Tensor [D]
#         MAP estimate.
#     Sigma_inv : Tensor [D, D]
#         Diagonal matrix whose entries are the Hessian diagonal at the MAP,
#         clipped to [prec_min, prec_max].
#     """
#     # MAP via Adam
#     beta = torch.randn(D, dtype=dtype, device=device) * 0.01
#     beta.requires_grad_(True)
#     optimizer = torch.optim.Adam([beta], lr=lr)
#     for _ in range(n_steps):
#         optimizer.zero_grad()
#         loss = energy_fn(beta)
#         loss.backward()
#         optimizer.step()
#     x_ref = beta.detach().clone()

#     # Diagonal of the Hessian via D Hessian-vector products
#     diag_H = torch.zeros(D, dtype=dtype, device=device)
#     for i in range(D):
#         b = x_ref.clone().requires_grad_(True)
#         g, = torch.autograd.grad(energy_fn(b), b, create_graph=True)
#         hi, = torch.autograd.grad(g[i], b, retain_graph=False)
#         diag_H[i] = hi[i]

#     diag_prec = diag_H.clamp(min=prec_min, max=prec_max)
#     Sigma_inv = torch.diag(diag_prec)
#     return x_ref, Sigma_inv

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
            from .sampling import resample_pdmp_path_sticky as rsm
        else:
            from .sampling import resample_pdmp_path as rsm

        samples = rsm(pos_np, vel_np, tim_np, x_ref_np,
                      N_resample=n_pilot, burnin_frac=0.2)

        # Only update coordinates with enough active samples
        if is_sticky:
            frac_active = np.mean(np.abs(samples) > zero_tol, axis=0)
            update_mask = frac_active > 0.1
        else:
            update_mask = np.ones(D, dtype=bool)

        Sigma_inv_diag = np.diag(sampler.Sigma_inv.cpu().numpy()).copy()
        x_ref_new = x_ref_np.copy()

        if update_mask.any():
            active_samples = samples[:, update_mask]
            x_ref_new[update_mask] = active_samples.mean(axis=0)
            var_diag = np.clip(active_samples.var(axis=0), 1e-8, None)
            Sigma_inv_diag[update_mask] = 1.0 / var_diag

        sampler.preprocess(
            x_ref=torch.tensor(x_ref_new, dtype=dtype, device=device),
            Sigma_inv=torch.tensor(np.diag(Sigma_inv_diag), dtype=dtype, device=device),
        )

# def warmup(sampler, n_rounds=3, n_pilot=500,
#                      target=None, tune_refresh=False):
#     """
#     Iterative warm-up: run short pilots, update (x_ref, Sigma_inv)
#     from resampled path moments.
#     """
#     dtype = sampler.dtype
#     device = sampler.device
#     D = sampler.D

#     # --- Round 0: initial reference ---
#     if target is not None and target.x_ref is not None:
#         sampler.preprocess(
#             x_ref=target.x_ref.to(dtype=dtype, device=device),
#             Sigma_inv=target.Sigma_inv.to(dtype=dtype, device=device),
#         )
#     else:
#         # No reference available — start with prior-like defaults
#         sampler.preprocess(
#             x_ref=torch.zeros(D, dtype=dtype, device=device),
#             Sigma_inv=torch.eye(D, dtype=dtype, device=device),
#         )

#     # --- Iterative refinement ---
#     for i in range(n_rounds):
#         result = sampler.sample(N=n_pilot, diagnostics=False)

#         pos_np = result["positions"].cpu().numpy()
#         vel_np = result["velocities"].cpu().numpy()
#         tim_np = result["times"].cpu().numpy()
#         x_ref_np = sampler.x_ref.cpu().numpy()

#         samples = resample_pdmp_path(
#             pos_np, vel_np, tim_np, x_ref_np,
#             N_resample=n_pilot, burnin_frac=0.0,
#         )

#         x_ref_new = np.mean(samples, axis=0)
#         var_diag = np.clip(np.var(samples, axis=0), 1e-8, None)
#         Sigma_inv_new = np.diag(1.0 / var_diag)

#         sampler.preprocess(
#             x_ref=torch.tensor(x_ref_new, dtype=dtype, device=device),
#             Sigma_inv=torch.tensor(Sigma_inv_new, dtype=dtype, device=device),
#         )

#     # if tune_refresh:
#     #     _tune_refresh_rate(sampler, n_pilot)
        
        
