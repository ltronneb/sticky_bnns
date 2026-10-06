"""Stochastic gradient PDMP samplers of Fearnhead, Grazzi, Nemeth and Roberts
(2024), SG-ZZ (their Algorithm 2), SG-BPS (Algorithm 3) and the sticky SG-ZZ
(SG-SZZ, Algorithm 4).

Time is cut into intervals of length step_size. Every iteration draws a fresh
minibatch gradient estimate g at the current state, holds the event rates
fixed at their values under g, and moves to the first of

    a velocity event (ZZ flip of coordinate i at rate (v_i g_i)_+, BPS reflection
        at rate (v . g)_+, BPS refreshment at rate refresh_rate),
    a sticky coordinate hitting zero (SG-SZZ),
    a frozen coordinate thawing at rate kappa_i |v_i| (SG-SZZ),
    the end of the current interval.

After an event inside an interval the next iteration simulates the rest of the
same interval with a new minibatch, as in the multi-event versions of the
paper (their Appendix C). Each iteration costs one gradient estimate.

Between velocity changes the path is linear, so only rows where v changes
(and the start and end) are written to the skeleton. The skeleton goes to
on_chunk(pos, vel, times) in the format of sazz.paper_ready.samplers, so
run_and_resample from there gives uniform-in-time posterior draws.

The loop never moves data from the device to the host inside an iteration.
The number of written rows is read every `sync_every` iterations.
"""

import math
import time
from typing import Callable, Optional

import torch
from torch import Tensor
from tqdm import tqdm

GRID, BOUNCE, REFRESH, FREEZE, THAW = range(5)
KINDS = ("grid_steps", "bounce", "refresh", "freeze", "thaw")


class SGPDMP:
    def __init__(self, grad_est: Callable[[Tensor], Tensor], D: int, step_size: float,
                 dtype=torch.float64, device="cpu"):
        self.grad, self.D, self.step_size = grad_est, D, step_size
        self.dtype, self.device = dtype, torch.device(device)
        self.kw = dict(dtype=dtype, device=self.device)
        self.inf = torch.tensor(math.inf, **self.kw)
        self.tiny = torch.finfo(dtype).tiny

    def trajectory(self, t, x: Tensor, v: Tensor):
        return x + t * v, v

    def _first(self, rate: Tensor):
        """Time and index of the first of independent exponential clocks with
        the given rates (rate 0 never rings)."""
        e = torch.empty_like(rate).exponential_()
        tau = torch.where(rate > 0, e / rate.clamp_min(self.tiny), self.inf)
        return tau.min(0)

    def _setup(self, x: Tensor, v: Tensor):
        pass

    def sample(self, x0: Tensor, n_iters: int, chunk_size: int = 10_000,
               on_chunk: Callable = lambda *a: None, progress: bool = True,
               sync_every: int = 500, max_seconds: Optional[float] = None) -> dict:
        """Runs n_iters iterations, or fewer if max_seconds of wall-clock time pass first."""
        D, kw = self.D, self.kw
        deadline = time.perf_counter() + max_seconds if max_seconds else math.inf
        x = x0.to(**kw).clone()
        v = self._initial_velocity()
        self._setup(x, v)
        x, v = x.clone(), v.clone()

        sync_every = max(1, min(sync_every, chunk_size // 2))
        cap = chunk_size + sync_every + 2
        pos, vel = torch.empty(cap, D, **kw), torch.empty(cap, D, **kw)
        tim = torch.empty(cap, dtype=torch.float64, device=self.device)
        pos[0], vel[0], tim[0] = x, v, 0.0
        row = torch.ones((), dtype=torch.long, device=self.device)    # next free row
        t = torch.zeros((), dtype=torch.float64, device=self.device)
        delta = torch.tensor(self.step_size, **kw)
        counts = torch.zeros(len(KINDS), dtype=torch.long, device=self.device)
        one = torch.ones(1, dtype=torch.long, device=self.device)
        diverged, done = False, 0

        def flush(n: int):
            on_chunk(pos[:n], vel[:n], tim[:n])
            pos[0], vel[0], tim[0] = pos[n - 1], vel[n - 1], tim[n - 1]
            row.fill_(1)

        bar = tqdm(total=n_iters, disable=not progress, desc=type(self).__name__, unit="it")
        while done < n_iters:
            for _ in range(min(sync_every, n_iters - done)):
                x, v, dt, kind = self._step(x, v, delta)
                delta = torch.where(kind == GRID, torch.full_like(delta, self.step_size), delta - dt)
                t = t + dt.to(torch.float64)
                counts.index_add_(0, kind.view(1), one)
                # always write at the next free row, keep the row only if v changed
                pos.index_copy_(0, row.view(1), x[None])
                vel.index_copy_(0, row.view(1), v[None])
                tim.index_copy_(0, row.view(1), t.view(1))
                row += (kind != GRID).long()
            done += min(sync_every, n_iters - done)
            bar.update(min(sync_every, bar.total - bar.n))
            if not bool(torch.isfinite(x).all()):
                diverged = True
                break
            n = int(row)
            if n >= chunk_size:
                flush(n)
            if time.perf_counter() > deadline:
                break
            if done % (20 * sync_every) == 0:
                bar.set_postfix_str(f"t={float(t):.4g} events={int(counts[1:].sum())}"
                                    + self._postfix(), refresh=False)
        bar.close()
        if not diverged:
            n = int(row)
            pos[n], vel[n], tim[n] = x, v, t      # the end of the path
            flush(n + 1)
        c = dict(zip(KINDS, counts.tolist()))
        N = self.grad.N if hasattr(self.grad, "N") else 1
        data_grads = getattr(self.grad, "data_grads", done)
        return {**c, "n_iters": done, "n_events": sum(c.values()) - c["grid_steps"],
                "data_grads": data_grads, "grad_evals": data_grads / N, "bound_violations": 0,
                "final_time": float(t), "diverged": diverged, "step_size": self.step_size,
                "frozen_final": self._frozen_final()}

    def _postfix(self) -> str:
        return ""

    def _frozen_final(self) -> Tensor:
        return torch.zeros(self.D, dtype=torch.bool)


class SGZigZag(SGPDMP):
    """SG-ZZ with velocities in {-1, +1}^D. gamma adds a constant flip rate per
    coordinate (0 in the paper). With kappa it is the sticky SG-SZZ, where a
    coordinate in can_freeze that hits zero freezes and thaws at rate
    kappa_i |v_i| with its old velocity. cold_start is a bool mask of
    coordinates that start frozen at zero."""

    def __init__(self, grad_est, D: int, step_size: float, gamma: float = 0.0,
                 kappa: Optional[Tensor] = None, can_freeze: Optional[Tensor] = None,
                 cold_start: Optional[Tensor] = None, **kwargs):
        super().__init__(grad_est, D, step_size, **kwargs)
        self.gamma = gamma
        self.sticky = kappa is not None
        if self.sticky:
            self.kappa = torch.as_tensor(kappa, **self.kw).expand(D).clone()
            mask = torch.ones(D, dtype=torch.bool) if can_freeze is None else can_freeze
            self.can_freeze = mask.to(self.device, torch.bool)
            self.cold_start = cold_start
        self.arange = torch.arange(D, device=self.device)
        self.sticky_kinds = torch.tensor([BOUNCE, GRID, FREEZE, THAW], device=self.device)

    def _initial_velocity(self) -> Tensor:
        return (2 * torch.randint(0, 2, (self.D,), device=self.device) - 1).to(self.dtype)

    def _setup(self, x: Tensor, v: Tensor):
        if not self.sticky:
            return
        self.frozen = torch.zeros(self.D, dtype=torch.bool, device=self.device)
        self.v_frozen = v.clone()
        if self.cold_start is not None:
            start = self.cold_start.to(self.device) & self.can_freeze & (self.kappa > 0)
            self.frozen |= start
            x[start], v[start] = 0.0, 0.0

    def _step(self, x: Tensor, v: Tensor, delta: Tensor):
        g = self.grad(x)
        rate = torch.clamp(v * g, min=0.0) + self.gamma
        if not self.sticky:
            tau, i = self._first(rate)
            dt, k = torch.stack([tau, delta]).min(0)
            x = x + v * dt
            v = torch.where((k == 0) & (self.arange == i), -v, v)
            return x, v, dt, torch.where(k == 0, BOUNCE, GRID)

        active = ~self.frozen
        tau, i = self._first(rate * active)
        towards_zero = active & self.can_freeze & (x * v < 0)
        hit = torch.where(towards_zero, -x / torch.where(towards_zero, v, torch.ones_like(v)), self.inf)
        tau_hit, i_hit = hit.min(0)
        tau_thaw, i_thaw = self._first(self.kappa * self.v_frozen.abs() * self.frozen)
        dt, k = torch.stack([tau, delta, tau_hit, tau_thaw]).min(0)
        x = x + v * dt
        v = torch.where((k == 0) & (self.arange == i), -v, v)
        stick = (k == 2) & (self.arange == i_hit)
        x = torch.where(stick, torch.zeros_like(x), x)
        self.v_frozen = torch.where(stick, v, self.v_frozen)
        thaw = (k == 3) & (self.arange == i_thaw)
        v = torch.where(stick, torch.zeros_like(v), torch.where(thaw, self.v_frozen, v))
        self.frozen = (self.frozen | stick) & ~thaw
        kind = self.sticky_kinds[k]
        return x, v, dt, kind

    def _postfix(self) -> str:
        return f" frozen={float(self.frozen.float().mean()):.2f}" if self.sticky else ""

    def _frozen_final(self) -> Tensor:
        return self.frozen.cpu().clone() if self.sticky else super()._frozen_final()


class SGBPS(SGPDMP):
    """SG-BPS with Gaussian velocities v ~ N(0, M) for a diagonal metric M
    (identity in the paper), reflections v - 2 (v . g) / (g . M g) M g at rate
    (v . g)_+ and refreshments v ~ N(0, M) at rate refresh_rate. With
    M = Sigma (the Laplace covariance) it is the BPS preconditioned like the
    Boomerang."""

    def __init__(self, grad_est, D: int, step_size: float, refresh_rate: float = 1.0,
                 metric: Optional[Tensor] = None, **kwargs):
        super().__init__(grad_est, D, step_size, **kwargs)
        self.refresh_rate = refresh_rate
        self.M = (torch.ones(D, **self.kw) if metric is None else metric.to(**self.kw))
        self.M_sqrt = self.M.sqrt()
        self.rates = torch.zeros(2, **self.kw)

    def _initial_velocity(self) -> Tensor:
        return self.M_sqrt * torch.randn(self.D, **self.kw)

    def _step(self, x: Tensor, v: Tensor, delta: Tensor):
        g = self.grad(x)
        vg = torch.dot(v, g)
        rates = torch.stack([vg.clamp(min=0.0), torch.full_like(vg, self.refresh_rate)])
        tau, j = self._first(rates)
        dt, k = torch.stack([tau, delta]).min(0)
        x = x + v * dt
        Mg = self.M * g
        reflected = v - 2.0 * vg / torch.dot(g, Mg).clamp_min(self.tiny) * Mg
        fresh = self.M_sqrt * torch.randn(self.D, **self.kw)
        event = k == 0
        v = torch.where(event & (j == 0), reflected, torch.where(event & (j == 1), fresh, v))
        kind = torch.where(event, torch.where(j == 0, BOUNCE, REFRESH), GRID)
        return x, v, dt, kind
