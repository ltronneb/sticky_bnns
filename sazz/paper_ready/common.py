"""Device, sampler construction and result files shared by the scripts."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .samplers import Boomerang, StickyBoomerang, StickyZigZag, ZigZag

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64
PDMPS = ("zigzag", "sticky_zigzag", "boomerang", "sticky_boomerang")


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_pdmp(name: str, bm, x_ref, Sigma_inv, *, t_max_init: float, gamma: float = 0.01,
              refresh_rate: float = 1.0, std_weight: Optional[float] = None,
              inclusion: Optional[float] = None, cold_start=None, batch_size: Optional[int] = None,
              alpha: float = 1.01, alpha_violation: float = 2.0):
    """One of PDMPS on the BNN target bm. Sticky samplers use a spike-and-slab
    prior with weight inclusion probability `inclusion`."""
    if batch_size:
        grad, resample = bm.minibatch_grad(batch_size)
    else:
        grad, resample = torch.func.grad(bm.energy), None
    kw = dict(t_max_init=t_max_init, alpha=alpha, alpha_violation=alpha_violation,
              resample_grad_batch=resample, dtype=bm.X.dtype, device=bm.device)
    if name.startswith("sticky"):
        kappa, mask = bm.sticky_prior(std_weight, inclusion)
        kw.update(kappa=kappa, can_freeze=mask, cold_start=cold_start)
    if name.endswith("zigzag"):
        cls = StickyZigZag if name.startswith("sticky") else ZigZag
        return cls(grad, bm.D, gamma=gamma, **kw)
    cls = StickyBoomerang if name.startswith("sticky") else Boomerang
    return cls(grad, bm.D, x_ref, Sigma_inv, refresh_rate=refresh_rate, **kw)


def save(path: Path, **payload):
    """Atomic, so parallel jobs never read a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in payload.items()}, tmp)
    os.replace(tmp, path)
    print(f"  saved -> {path}")


def summary(out: dict) -> str:
    return (f"{out['n_events']} events, {out['grad_evals']} grads, {out['elapsed_sec']:.0f}s, "
            f"{out['bound_violations']} bound violations, "
            f"{float(out['frozen_final'].float().mean()):.2f} frozen at the end")
