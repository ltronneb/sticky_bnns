"""Reference point x_ref (MAP) and diagonal precision Sigma_inv (Laplace with
the empirical Fisher), plus pruning of the MAP for the image experiments."""

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..models.bnn import BNN


def fit_map(bm: BNN, n_steps: int, lr: float = 1e-2, init: Optional[Tensor] = None,
            batch_size: Optional[int] = None, cosine: bool = False,
            mask: Optional[Tensor] = None, log_every: int = 0) -> Tensor:
    """Adam on the energy from init (default N(0, I)), with a minibatch estimate
    of the likelihood when batch_size is given. Only coordinates with mask = 1 move."""
    beta = (torch.randn(bm.D, dtype=bm.X.dtype, device=bm.device) if init is None
            else init.clone()).requires_grad_(True)
    opt = torch.optim.Adam([beta], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, lr * 0.1) if cosine else None
    grad_fn = bm.minibatch_grad(batch_size) if batch_size else None
    for step in range(1, n_steps + 1):
        opt.zero_grad()
        if grad_fn is None:
            bm.energy(beta).backward()
        else:
            grad_fn[1]()
            beta.grad = grad_fn[0](beta.detach())
        if mask is not None:
            beta.grad.mul_(mask)
        opt.step()
        if sched:
            sched.step()
        if log_every and step % log_every == 0:
            print(f"  MAP step {step}/{n_steps}  energy={float(bm.energy(beta.detach())):.2f}")
    return beta.detach()


def laplace_precision(bm: BNN, x_ref: Tensor, n_fisher: int = 128) -> Tensor:
    """Sigma_inv = prior precision + N * mean empirical Fisher diagonal over
    n_fisher points. For a learned noise the log_sigma entry uses the prior
    curvature only, since its likelihood curvature is overconfident as it
    ignores the coupling to the weights."""
    N = bm.X.shape[0]
    idx = torch.randperm(N, device=bm.device)[:min(n_fisher, N)]
    per_point = torch.func.vmap(torch.func.grad(bm.log_lik.single), in_dims=(None, 0, 0))
    g = per_point(x_ref, bm.X[idx].unsqueeze(1), bm.y[idx].unsqueeze(1))
    prec = (bm.prior_precision + N * (g ** 2).mean(0)).clamp(min=1e-8)
    if bm.learns_noise:
        prec[-1] = 2.0 * x_ref[-1].exp() ** 2 / bm.prior_sigma_scale ** 2
    return prec


@torch.no_grad()
def accuracy(bm: BNN, beta: Tensor, X: Tensor, y: Tensor) -> float:
    return float((bm.predict(beta, X).argmax(-1) == y).float().mean())


def prune_and_refit(bm: BNN, x_ref: Tensor, can_freeze: Tensor, X_val: Tensor, y_val: Tensor,
                    tol: float = 0.01, refit_steps: int = 200, refit_lr: float = 1e-3,
                    batch_size: Optional[int] = None):
    """Zero the freezable coordinates with |x_ref_i| < m * prior_std_i for the
    largest m in logspace(-3, 0, 60) that keeps validation accuracy within tol
    of the unpruned MAP, then refit the rest with the pruned coordinates fixed
    at zero. Returns (x_pruned, pruned_mask)."""
    std = bm.prior_precision.clamp(min=1e-12).rsqrt()
    base = accuracy(bm, x_ref, X_val, y_val)
    mask = torch.zeros_like(can_freeze)
    for m in np.logspace(-3, 0, 60):
        cand = (x_ref.abs() < m * std) & can_freeze
        if accuracy(bm, torch.where(cand, torch.zeros_like(x_ref), x_ref), X_val, y_val) >= base - tol:
            mask = cand
    x = torch.where(mask, torch.zeros_like(x_ref), x_ref)
    x = fit_map(bm, refit_steps, refit_lr, init=x, batch_size=batch_size, cosine=True,
                mask=(~mask).to(x.dtype))
    x[mask] = 0.0
    print(f"  pruned {int(mask.sum())}/{int(can_freeze.sum())} weights, "
          f"val acc {base:.4f} -> {accuracy(bm, x, X_val, y_val):.4f}")
    return x, mask
