"""Sampler construction, result names and summaries for the SG-PDMP scripts."""

from typing import Optional

import torch

from .gradients import ControlVariateGradient
from .samplers import SGBPS, SGZigZag

SG_PDMPS = ("sg_zigzag", "sg_sticky_zigzag", "sg_bps")
LABELS = {"sg_zigzag": "SG-ZZ", "sg_sticky_zigzag": "SG-SZZ", "sg_bps": "SG-BPS"}


def make_sg_pdmp(name: str, bm, x_hat, *, step_size: float, batch_size: int,
                 cv: bool = True, std_weight: Optional[float] = None,
                 inclusion: Optional[float] = None, cold_start=None, gamma: float = 0.0,
                 refresh_rate: float = 1.0, metric=None, cache_bytes: int = 2 ** 30):
    """One of SG_PDMPS on the BNN target bm, with the control variate centred
    at x_hat (cv=False drops it). The sticky sampler uses the spike-and-slab
    prior of the paper's sticky samplers, bm.sticky_prior(std_weight, inclusion).
    metric is the diagonal velocity covariance of SG-BPS (None for identity)."""
    grad = ControlVariateGradient(bm, batch_size, x_hat if cv else None, cache_bytes)
    kw = dict(dtype=bm.X.dtype, device=bm.device)
    if name == "sg_bps":
        return SGBPS(grad, bm.D, step_size, refresh_rate=refresh_rate, metric=metric, **kw)
    if name == "sg_zigzag":
        return SGZigZag(grad, bm.D, step_size, gamma=gamma, **kw)
    if name == "sg_sticky_zigzag":
        kappa, mask = bm.sticky_prior(std_weight, inclusion)
        return SGZigZag(grad, bm.D, step_size, gamma=gamma, kappa=kappa, can_freeze=mask,
                        cold_start=cold_start, **kw)
    raise ValueError(name)


def run_name(name: str, step_size: float, batch_size: Optional[int] = None) -> str:
    """File stem, e.g. sg_zigzag_h1e-04 or sg_zigzag_h1e-04_b32."""
    return f"{name}_h{step_size:.0e}" + ("" if batch_size is None else f"_b{batch_size}")


def parse_run_name(stem: str):
    """(sampler, step_size, batch_size or None) of a run_name, or None."""
    name = next((n for n in sorted(SG_PDMPS, key=len, reverse=True) if stem.startswith(n + "_h")), None)
    if name is None:
        return None
    parts = stem[len(name) + 2:].split("_b")
    return name, float(parts[0]), int(parts[1]) if len(parts) > 1 else None


def label(stem: str) -> str:
    name, h, b = parse_run_name(stem)
    return f"{LABELS[name]} h={h:g}" + ("" if b is None else f" n={b}")


def summary(out: dict) -> str:
    s = (f"{out['n_iters']:,} its, {out['n_events']:,} events ({out['bounce']:,} bounces), "
         f"{out['grad_evals']:,.0f} full grads, T={out['final_time']:.4g}, {out['elapsed_sec']:.0f}s")
    if out["freeze"] or out["thaw"]:
        s += f", {float(out['frozen_final'].float().mean()):.2f} frozen at the end"
    return s + (", DIVERGED" if out["diverged"] else "")


def nan_samples(n: int, D: int) -> torch.Tensor:
    return torch.full((n, D), float("nan"), dtype=torch.float64)
