"""(Sticky) ZigZag sampler with velocities in {-1, +1}^D and linear flow.

Event rate sum_j (v_j d_j U(x_t))_+ + n_active * gamma, bounded coordinate-wise
(see bound.py). At an event one coordinate flips, chosen proportionally to its
own rate.
"""

import math
from functools import partial

import torch
from torch import Tensor

from .bound import tangent_bound
from .pdmp import PDMP


class ZigZag(PDMP):
    def __init__(self, grad_target, D: int, gamma: float = 0.01, **kwargs):
        super().__init__(grad_target, D, **kwargs)
        self.gamma = gamma

    def trajectory(self, t, x: Tensor, v: Tensor):
        return x + t * v, v

    def _initial_velocity(self) -> Tensor:
        return (2 * torch.randint(0, 2, (self.D,), device=self.device) - 1).to(self.dtype)

    def _hitting_times(self, x: Tensor, v: Tensor) -> Tensor:
        t = -x / torch.where(v.abs() < 1e-14, torch.ones_like(v), v)
        bad = (v.abs() < 1e-14) | (x.abs() < 1e-14) | (t <= 0)
        return torch.where(bad, torch.full_like(t, math.inf), t)

    def _coordinate_rates(self, x: Tensor, v: Tensor) -> Tensor:
        return (torch.clamp(v * self.grad_target(x), min=0.0) + self.gamma) * (~self.frozen)

    def _rate_closures(self, x: Tensor, v: Tensor, n_active: int):
        def signed(t):
            x_t, v_t = self._flow(t, x, v)
            return v_t * self.grad_target(x_t)

        def rate_and_slope(ts):
            ts = ts.to(dtype=self.dtype, device=self.device)
            y, d = torch.func.vmap(lambda t: torch.func.jvp(signed, (t,), (torch.ones_like(t),)))(ts)
            return y.T, d.T

        @torch.no_grad()
        def rate(t: float) -> float:
            x_t, v_t = self._at(t, x, v)
            return float(self._coordinate_rates(x_t, v_t).sum())

        return rate_and_slope, rate, partial(tangent_bound, per_coord=True,
                                             offset=n_active * self.gamma)

    @torch.no_grad()
    def _bounce(self, x: Tensor, v: Tensor) -> Tensor:
        r = self._coordinate_rates(x, v)
        i = int(torch.multinomial(r / r.sum(), 1))
        v = v.clone()
        v[i] = -v[i]
        return v


class StickyZigZag(ZigZag):
    """ZigZag with spike-and-slab stickiness at zero, with thaw rates kappa [D],
    a bool mask can_freeze [D] and cold_start (bool mask or |x0| threshold)."""

    def __init__(self, grad_target, D: int, kappa, can_freeze=None, cold_start=None, **kwargs):
        super().__init__(grad_target, D, kappa=kappa, can_freeze=can_freeze,
                         cold_start=cold_start, **kwargs)
