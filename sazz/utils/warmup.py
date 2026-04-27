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
from typing import Optional, Literal, Sequence
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


def find_reference_bnn(
    energy_fn,
    D: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    n_steps: int = 4000,
    lr: float = 1e-2,
    model: Optional["BayesianModel"] = None,
    reference: Literal["prior", "fisher", "hessian"] = "prior",
    layer_slices: Optional[Sequence[slice]] = None,
    n_samples_fisher: int = 100,
    per_layer_clip: bool = False,
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    MAP via Adam + choice of reference precision for BNNs.

    Parameters
    ----------
    reference : {"prior", "fisher", "hessian"}
        "prior"   — use the prior precision as Sigma_inv. Requires `model`
                    with a prior exposing precision_diag(). Recommended
                    default for moderate-to-large BNNs: orbits matched to
                    prior scale, no tuning needed.
        "fisher"  — empirical Fisher diagonal with optional per-layer
                    clipping. Historical behaviour, kept for reproducibility.
        "hessian" — full Hessian of the energy at the MAP, symmetrised and
                    jittered. Gives a posterior-matched Sigma_inv that
                    captures correlations. Suitable for small BNNs (D up
                    to a few hundred) where the D x D Hessian is cheap
                    and the posterior is approximately Gaussian around
                    the MAP.
    jitter : float
        Diagonal jitter added to the Hessian for numerical positive-
        definiteness. Only used when reference="hessian".
    """
    fn = model.energy if model is not None else energy_fn

    # --- MAP via Adam ---
    beta = torch.randn(D, dtype=dtype, device=device) * 0.01
    beta.requires_grad_(True)
    optimizer = torch.optim.Adam([beta], lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = fn(beta)
        loss.backward()
        optimizer.step()
    x_ref = beta.detach().clone()

    if reference == "prior":
        if model is None:
            raise ValueError("reference='prior' requires a model argument.")
        prec = model.prior.precision_diag().to(dtype=dtype, device=device)
        Sigma_inv = torch.diag(prec)

    elif reference == "fisher":
        grad_sq = torch.zeros(D, dtype=dtype, device=device)
        for _ in range(max(n_samples_fisher, 1)):
            b = x_ref.clone().requires_grad_(True)
            g, = torch.autograd.grad(fn(b), b)
            grad_sq += g ** 2
        grad_sq /= max(n_samples_fisher, 1)

        if layer_slices is None:
            layer_slices = [slice(0, D)]

        diag_prec = grad_sq.clone()
        if per_layer_clip:
            for sl in layer_slices:
                block = diag_prec[sl]
                if block.numel() == 0:
                    continue
                diag_prec[sl] = block.clamp(min=1.0, max=1e5)
        Sigma_inv = torch.diag(diag_prec)

    elif reference == "hessian_weights_only":
        if model is None:
            raise ValueError("reference='hessian_weights_only' requires a model.")
        
        # Full Hessian
        H = torch.autograd.functional.hessian(fn, x_ref)
        H = 0.5 * (H + H.T)
        
        # Build a mask for bias coordinates from the layer structure
        # (Requires knowing which coordinates are biases — pass via model or layer_sizes)
        prior_prec = model.prior.precision_diag().to(dtype=dtype, device=device)
        bias_mask = _make_bias_mask(model)  
        
        # Replace bias rows/cols with diagonal prior structure
        for i in range(D):
            if bias_mask[i]:
                H[i, :] = 0.0
                H[:, i] = 0.0
                H[i, i] = prior_prec[i]
        
        H = H + jitter * torch.eye(D, dtype=dtype, device=device)
        Sigma_inv = H
        
    elif reference == "hessian_prior_floor":
        if model is None:
            raise ValueError("requires a model.")
        
        H = torch.autograd.functional.hessian(fn, x_ref)
        H = 0.5 * (H + H.T)
        
        prior_prec = model.prior.precision_diag().to(dtype=dtype, device=device)
        
        # Floor the diagonal: H_ii ← max(H_ii, prior_prec_i)
        diag_current = H.diagonal()
        diag_floored = torch.maximum(diag_current, prior_prec)
        # Replace diagonal
        H = H - torch.diag(diag_current) + torch.diag(diag_floored)
        
        Sigma_inv = H
        
    elif reference == "adam":
        # Use Adam's running second-moment estimate as Sigma_inv diagonal
        state = optimizer.state[beta]
        v_hat = state["exp_avg_sq"].clone()
        
        # Bias correction (match Adam's internal computation)
        step = state["step"]
        if isinstance(step, torch.Tensor):
            step = step.item()
        bias_correction = 1.0 - 0.999 ** step  # Adam's beta2
        v_hat = v_hat / bias_correction
        
        # Floor by prior precision to avoid coordinates with near-zero gradient
        if model is not None:
            prior_prec = model.prior.precision_diag().to(dtype=dtype, device=device)
            diag_prec = torch.maximum(v_hat.sqrt(), prior_prec.sqrt()) ** 2
        else:
            diag_prec = v_hat.clamp(min=1e-4)
        # diag_prec = v_hat.clamp(min=1e-4)
        
        Sigma_inv = torch.diag(diag_prec)

    else:
        raise ValueError(f"Unknown reference type: {reference}")

    return x_ref, Sigma_inv

# def find_reference_bnn(
#     energy_fn,
#     D: int,
#     dtype: torch.dtype = torch.float64,
#     device: torch.device = torch.device("cpu"),
#     n_steps: int = 4000,
#     lr: float = 1e-2,
#     model: Optional["BayesianModel"] = None,
#     reference: Literal["prior", "fisher"] = "prior",
#     layer_slices: Optional[Sequence[slice]] = None,
#     n_samples_fisher: int = 100,
#     per_layer_clip: bool = False,
# ) -> tuple[Tensor, Tensor]:
#     """
#     MAP via Adam + choice of reference precision for BNNs.

#     Parameters
#     ----------
#     reference : {"prior", "fisher"}
#         "prior"  — use the prior precision as Sigma_inv. Requires `model`
#                    with a prior exposing precision_diag(). This is the
#                    recommended default for BNNs: it gives orbits matched
#                    to the prior scale and requires no tuning.
#         "fisher" — empirical Fisher diagonal with per-layer clipping.
#                    Historical behaviour, kept for reproducibility.
#     """
#     fn = model.energy if model is not None else energy_fn

#     # --- MAP via Adam ---
#     beta = torch.randn(D, dtype=dtype, device=device) * 0.01
#     beta.requires_grad_(True)
#     optimizer = torch.optim.Adam([beta], lr=lr)
#     for _ in range(n_steps):
#         optimizer.zero_grad()
#         loss = fn(beta)
#         loss.backward()
#         optimizer.step()
#     x_ref = beta.detach().clone()

#     if reference == "prior":
#         if model is None:
#             raise ValueError("reference='prior' requires a model argument.")
#         prec = model.prior.precision_diag().to(dtype=dtype, device=device)
#         Sigma_inv = torch.diag(prec)

#     elif reference == "fisher":
#         grad_sq = torch.zeros(D, dtype=dtype, device=device)
#         for _ in range(max(n_samples_fisher, 1)):
#             b = x_ref.clone().requires_grad_(True)
#             g, = torch.autograd.grad(fn(b), b)
#             grad_sq += g ** 2
#         grad_sq /= max(n_samples_fisher, 1)

#         if layer_slices is None:
#             layer_slices = [slice(0, D)]

#         diag_prec = grad_sq.clone()
#         if per_layer_clip:
#             for sl in layer_slices:
#                 block = diag_prec[sl]
#                 if block.numel() == 0:
#                     continue
#                 diag_prec[sl] = block.clamp(min=1.0, max=1e5)
#         Sigma_inv = torch.diag(diag_prec)

#     else:
#         raise ValueError(f"Unknown reference type: {reference}")

#     return x_ref, Sigma_inv


def _make_bias_mask(model) -> Tensor:
    """Return a boolean mask [D] where True indicates a bias coordinate."""
    layer_sizes = model.likelihood.layer_sizes
    D = sum(layer_sizes[i+1] * layer_sizes[i] + layer_sizes[i+1]
            for i in range(len(layer_sizes)-1))
    mask = torch.zeros(D, dtype=torch.bool)
    offset = 0
    for i in range(len(layer_sizes) - 1):
        n_W = layer_sizes[i+1] * layer_sizes[i]
        n_b = layer_sizes[i+1]
        # Weights: offset to offset+n_W → False
        offset += n_W
        # Biases: offset to offset+n_b → True
        mask[offset:offset+n_b] = True
        offset += n_b
    return mask




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

        # if update_mask.any():
        #     active_samples = samples[:, update_mask]
        #     x_ref_new[update_mask] = active_samples.mean(axis=0)
        #     var_diag = np.clip(active_samples.var(axis=0), 1e-8, None)
        #     Sigma_inv_diag[update_mask] = 1.0 / var_diag
        if update_mask.any():
            active_samples = samples[:, update_mask]
            x_ref_new[update_mask] = active_samples.mean(axis=0)

            # Full empirical covariance instead of diagonal variance
            cov = np.cov(active_samples.T)  # shape [n_active, n_active]

            # Regularise to ensure PD
            cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
            Sigma_inv_block = np.linalg.inv(cov_reg)

            # Place into the full Sigma_inv (preserves masked-out coords' values)
            Sigma_inv_full = sampler.Sigma_inv.cpu().numpy().copy()
            active_idx = np.where(update_mask)[0]
            Sigma_inv_full[np.ix_(active_idx, active_idx)] = Sigma_inv_block

        else:
            # No active coords — keep current Sigma_inv unchanged
            Sigma_inv_full = sampler.Sigma_inv.cpu().numpy().copy()

        sampler.preprocess(
            x_ref=torch.tensor(x_ref_new, dtype=dtype, device=device),
            Sigma_inv=torch.tensor(Sigma_inv_full, dtype=dtype, device=device),
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
        
        
