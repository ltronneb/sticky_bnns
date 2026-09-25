"""Event loop shared by the (sticky) ZigZag and Boomerang samplers.

A subclass supplies the flow (`trajectory`), the event rate and its bound
(`_rate_closures`), the jump at an event (`_bounce`), the times at which
coordinates hit zero (`_hitting_times`) and the initial velocity. A sampler is
sticky when `kappa` is given. A freezable coordinate that hits zero freezes,
and thaws after an Exp(kappa_i |v_i|) time with its old velocity. Without
`kappa` no coordinate can freeze and the sampler is the plain one.

The skeleton is written in chunks of `chunk_size` events to `on_chunk(pos,
vel, times)`, and consecutive chunks share their boundary row. The run stops after
`n_events` events or `grad_budget` gradient evaluations.
"""

import math
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .bound import adapt_t_max, thin


class PDMP:
    refresh_rate = 0.0

    def __init__(self, grad_target: Callable, D: int, kappa=None, can_freeze=None,
                 cold_start=None, t_max_init: float = 0.01, alpha: float = 1.01,
                 alpha_violation: float = 2.0, resample_grad_batch: Optional[Callable] = None,
                 dtype=torch.float64, device="cpu"):
        self.grad_target, self.D = grad_target, D
        self.dtype, self.device = dtype, torch.device(device)
        self.sticky = kappa is not None
        kappa = 0.0 if kappa is None else kappa
        self.kappa = torch.as_tensor(kappa, dtype=dtype, device=self.device).expand(D).clone()
        if not self.sticky:
            can_freeze = torch.zeros(D, dtype=torch.bool)
        elif can_freeze is None:
            can_freeze = torch.ones(D, dtype=torch.bool)
        self.can_freeze = can_freeze.to(device=self.device, dtype=torch.bool)
        self.cold_start = cold_start
        self.t_max, self.alpha, self.alpha_violation = t_max_init, alpha, alpha_violation
        self.resample_grad_batch = resample_grad_batch

    # ---------------------------------------------------------------- flow
    def _flow(self, t, x: Tensor, v: Tensor):
        """trajectory with frozen coordinates held at zero."""
        x_t, v_t = self.trajectory(t, x, v)
        zero = torch.zeros_like(x_t)
        return torch.where(self.frozen, zero, x_t), torch.where(self.frozen, zero, v_t)

    def _at(self, t: float, x: Tensor, v: Tensor):
        with torch.no_grad():
            return self._flow(torch.tensor(t, dtype=self.dtype, device=self.device), x, v)

    # ------------------------------------------------------------ sticky
    def _freeze(self, i: int, v_i: float, now: float):
        self.frozen[i], self.frozen_v[i] = True, v_i
        rate = float(self.kappa[i]) * abs(v_i)
        self.deadline[i] = now + np.random.exponential(1.0 / rate) if rate > 1e-14 else math.inf

    def _next_thaw(self, now: float):
        if not self.frozen.any():
            return math.inf, None
        d = torch.where(self.frozen, self.deadline, torch.full_like(self.deadline, math.inf))
        i = int(torch.argmin(d))
        return max(float(d[i]) - now, 0.0), i

    def _next_hit(self, x: Tensor, v: Tensor):
        if not self.sticky:
            return math.inf, None
        t = self._hitting_times(x, v)
        t = torch.where(~self.frozen & self.can_freeze, t, torch.full_like(t, math.inf))
        dt, i = torch.min(t, dim=0)
        return (float(dt), int(i)) if math.isfinite(float(dt)) else (math.inf, None)

    def _apply_cold_start(self, x: Tensor, v: Tensor):
        """Freeze coordinates given by a bool mask, or with |x0_i| < threshold."""
        if self.cold_start is None:
            return
        near = (self.cold_start.to(self.device) if isinstance(self.cold_start, Tensor)
                else x.abs() < self.cold_start)
        rate = self.kappa * v.abs()
        idx = torch.nonzero(near & self.can_freeze & (rate > 1e-14)).squeeze(-1)
        self.frozen_v[idx], self.frozen[idx] = v[idx], True
        x[idx], v[idx] = 0.0, 0.0
        self.deadline[idx] = torch.distributions.Exponential(rate[idx]).sample()

    # ------------------------------------------------------------- bound
    def _next_event(self, x: Tensor, v: Tensor, dt_refresh: float, dt_hit: float,
                    dt_thaw: float):
        n_active = self.D - int(self.frozen.sum())
        if n_active == 0:
            horizon = min(dt_refresh, dt_thaw)
            if not math.isfinite(horizon):
                raise RuntimeError("all coordinates are permanently frozen")
            return math.inf, {"accepted": False, "rate_evals": 0, "violations": 0,
                              "rejections": 0, "effective_horizon": horizon}
        candidates = {"t_max": self.t_max, "refresh": dt_refresh, "hit": dt_hit, "thaw": dt_thaw}
        binding = min(candidates, key=candidates.get)
        horizon = candidates[binding]
        rate_and_slope, rate, bound = self._rate_closures(x, v, n_active)
        tau, s = thin(rate_and_slope, rate, horizon, bound, left=self._cache)
        self._cache = s["right"] if binding == "t_max" else None
        self.t_max = adapt_t_max(self.t_max, s, tau, horizon, binding == "t_max",
                                 self.alpha, self.alpha_violation)
        return tau, s

    # -------------------------------------------------------------- loop
    def sample(self, x0: Tensor, n_events: Optional[int] = None,
               grad_budget: Optional[int] = None, chunk_size: int = 10_000,
               on_chunk: Callable = lambda *a: None, progress: bool = True) -> dict:
        n_events = n_events or math.inf
        grad_budget = grad_budget or math.inf
        D, kw = self.D, dict(dtype=self.dtype, device=self.device)
        self.frozen = torch.zeros(D, dtype=torch.bool, device=self.device)
        self.frozen_v = torch.zeros(D, **kw)
        self.deadline = torch.full((D,), math.inf, **kw)
        self._cache = None

        x = x0.to(**kw).clone()
        v = self._initial_velocity()
        self._apply_cold_start(x, v)

        pos, vel = torch.empty(chunk_size + 1, D, **kw), torch.empty(chunk_size + 1, D, **kw)
        tim = torch.empty(chunk_size + 1, dtype=torch.float64, device=self.device)
        n_rows = 0
        st = {"x": x, "v": v, "t": 0.0, "elapsed": 0.0}
        counts = dict(bounce=0, freeze=0, thaw=0, refresh=0)
        stats = dict(grad_evals=0, bound_violations=0, rejections=0)

        def commit(x_new: Tensor, v_new: Tensor, kind: str):
            nonlocal n_rows
            st["t"] += st["elapsed"]
            st["x"], st["v"], st["elapsed"] = x_new, v_new, 0.0
            pos[n_rows], vel[n_rows], tim[n_rows] = x_new, v_new, st["t"]
            n_rows += 1
            counts[kind] += 1
            self._cache = None
            if n_rows == chunk_size + 1:
                on_chunk(pos, vel, tim)
                pos[0], vel[0], tim[0] = pos[-1], vel[-1], tim[-1]
                n_rows = 1

        pos[0], vel[0], tim[0] = x, v, 0.0
        n_rows = 1
        dt_refresh = np.random.exponential(1.0 / self.refresh_rate) if self.refresh_rate else math.inf
        bar = tqdm(total=None if math.isinf(grad_budget) else grad_budget,
                   disable=not progress, desc=type(self).__name__, unit="grad")
        it = 0
        while sum(counts.values()) < n_events and stats["grad_evals"] < grad_budget:
            it += 1
            now = st["t"] + st["elapsed"]
            x_t, v_t = self._at(st["elapsed"], st["x"], st["v"])
            dt_hit, i_hit = self._next_hit(x_t, v_t)
            dt_thaw, i_thaw = self._next_thaw(now)
            if self.resample_grad_batch is not None:
                self.resample_grad_batch()
                self._cache = None

            tau, s = self._next_event(x_t, v_t, dt_refresh, dt_hit, dt_thaw)
            used = s["rate_evals"]
            stats["bound_violations"] += s["violations"]
            stats["rejections"] += s["rejections"]
            if s["accepted"]:
                xe, ve = self._at(st["elapsed"] + tau, st["x"], st["v"])
                st["elapsed"] += tau
                dt_refresh -= tau
                used += 1
                commit(xe, self._bounce(xe, ve), "bounce")
            else:
                adv = s["effective_horizon"]
                st["elapsed"] += adv
                dt_refresh -= adv
                if i_hit is not None and abs(adv - dt_hit) < 1e-12:
                    xe, ve = self._at(st["elapsed"], st["x"], st["v"])
                    self._freeze(i_hit, float(ve[i_hit]), st["t"] + st["elapsed"])
                    xe[i_hit], ve[i_hit] = 0.0, 0.0
                    commit(xe, ve, "freeze")
                elif i_thaw is not None and abs(adv - dt_thaw) < 1e-12:
                    xe, ve = self._at(st["elapsed"], st["x"], st["v"])
                    ve[i_thaw] = self.frozen_v[i_thaw]
                    self.frozen[i_thaw], self.deadline[i_thaw] = False, math.inf
                    commit(xe, ve, "thaw")
            if dt_refresh <= 1e-14:
                xe, _ = self._at(st["elapsed"], st["x"], st["v"])
                commit(xe, self._refresh(st["t"] + st["elapsed"]), "refresh")
                dt_refresh = np.random.exponential(1.0 / self.refresh_rate)

            stats["grad_evals"] += used
            bar.update(used)
            if it % 200 == 0:
                bar.set_postfix_str(f"t={now:.4g} t_max={self.t_max:.2e} "
                                    f"frozen={float(self.frozen.float().mean()):.2f} "
                                    f"viol={stats['bound_violations']}", refresh=False)
        bar.close()
        if n_rows > 1:
            on_chunk(pos[:n_rows], vel[:n_rows], tim[:n_rows])
        return {**stats, **counts, "n_events": sum(counts.values()),
                "final_time": st["t"] + st["elapsed"], "t_max_final": self.t_max,
                "frozen_final": self.frozen.cpu().clone()}
