"""(Sticky) Boomerang sampler (Bierkens et al., 2020) with a diagonal Gaussian
reference N(x_ref, Sigma), Sigma = 1 / Sigma_inv.

The flow rotates around x_ref, the event rate is <v_t, grad U_excess(x_t)>_+
with U_excess = U - reference, an event reflects v in the Sigma-metric, and
velocities are refreshed at rate refresh_rate. In the sticky version a
refreshment keeps a frozen coordinate's sign and redraws its thaw time.
"""

import math
from functools import partial

import torch
from torch import Tensor

from .bound import tangent_bound
from .pdmp import PDMP


class Boomerang(PDMP):
    def __init__(self, grad_target, D: int, x_ref: Tensor, Sigma_inv: Tensor,
                 refresh_rate: float = 1.0, **kwargs):
        super().__init__(grad_target, D, **kwargs)
        kw = dict(dtype=self.dtype, device=self.device)
        self.x_ref = x_ref.to(**kw)
        self.Sigma_inv = Sigma_inv.to(**kw)
        self.Sigma = 1.0 / self.Sigma_inv
        self.Sigma_sqrt = self.Sigma.sqrt()
        self.refresh_rate = refresh_rate

    def trajectory(self, t, x: Tensor, v: Tensor):
        c, s, dx = torch.cos(t), torch.sin(t), x - self.x_ref
        return self.x_ref + dx * c + v * s, v * c - dx * s

    def grad_excess(self, x: Tensor) -> Tensor:
        return self.grad_target(x) - self.Sigma_inv * (x - self.x_ref)

    def _initial_velocity(self) -> Tensor:
        return self.Sigma_sqrt * torch.randn(self.D, dtype=self.dtype, device=self.device)

    def _hitting_times(self, x: Tensor, v: Tensor) -> Tensor:
        """Smallest t > 0 with x_ref + (x - x_ref) cos t + v sin t = 0."""
        a, b, c = x - self.x_ref, v, -self.x_ref
        R = torch.hypot(a, b)
        feasible = (R >= 1e-14) & (c.abs() <= R + 1e-14)
        phi = torch.atan2(b, a)
        delta = torch.acos(torch.clamp(c / R.clamp_min(1e-14), -1.0, 1.0))
        cand = torch.stack([torch.remainder(phi - delta, 2 * math.pi),
                            torch.remainder(phi + delta, 2 * math.pi)])
        cand = torch.where(cand < 1e-10, cand + 2 * math.pi, cand).min(dim=0).values
        return torch.where(feasible, cand, torch.full_like(cand, math.inf))

    def _rate_closures(self, x: Tensor, v: Tensor, n_active: int):
        def signed(t):
            x_t, v_t = self._flow(t, x, v)
            return torch.dot(v_t, self.grad_excess(x_t))

        def rate_and_slope(ts):
            ts = ts.to(dtype=self.dtype, device=self.device)
            return torch.func.vmap(lambda t: torch.func.jvp(signed, (t,), (torch.ones_like(t),)))(ts)

        @torch.no_grad()
        def rate(t: float) -> float:
            x_t, v_t = self._at(t, x, v)
            return float(torch.dot(v_t, self.grad_excess(x_t)))

        return rate_and_slope, rate, partial(tangent_bound, per_coord=False)

    @torch.no_grad()
    def _bounce(self, x: Tensor, v: Tensor) -> Tensor:
        """Reflect the active velocity components in the Sigma-metric."""
        g = self.grad_excess(x) * (~self.frozen)
        Sg = self.Sigma * g
        denom = torch.dot(g, Sg)
        if float(denom) <= 1e-14:
            return v
        return torch.where(self.frozen, torch.zeros_like(v), v - 2.0 * torch.dot(v, g) / denom * Sg)

    @torch.no_grad()
    def _refresh(self, now: float) -> Tensor:
        z = torch.randn(self.D, dtype=self.dtype, device=self.device)
        v = torch.where(self.frozen, torch.zeros_like(z), self.Sigma_sqrt * z)
        idx = torch.nonzero(self.frozen).squeeze(-1)
        if idx.numel():
            sign = torch.sign(self.frozen_v[idx])
            sign = torch.where(sign == 0, torch.randint(0, 2, sign.shape, device=self.device)
                               .to(self.dtype) * 2 - 1, sign)
            new_v = sign * self.Sigma_sqrt[idx] * torch.randn(idx.numel(), dtype=self.dtype,
                                                            device=self.device).abs()
            self.frozen_v[idx] = new_v
            rate = self.kappa[idx] * new_v.abs()
            draw = torch.distributions.Exponential(rate.clamp_min(1e-14)).sample()
            self.deadline[idx] = torch.where(rate > 1e-14, now + draw,
                                             torch.full_like(draw, math.inf))
        return v


class StickyBoomerang(Boomerang):
    """Boomerang with spike-and-slab stickiness at zero (see StickyZigZag)."""

    def __init__(self, grad_target, D: int, x_ref: Tensor, Sigma_inv: Tensor, kappa,
                 can_freeze=None, cold_start=None, **kwargs):
        super().__init__(grad_target, D, x_ref, Sigma_inv, kappa=kappa,
                         can_freeze=can_freeze, cold_start=cold_start, **kwargs)
