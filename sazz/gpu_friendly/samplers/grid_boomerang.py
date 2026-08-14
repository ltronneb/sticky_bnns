"""
Boomerang sampler using ONLY the grid-based upper bound (Andral & Kamatani
2024) for Poisson thinning. Not a subclass of
sazz.samplers.AutomaticBoomerangSampler -- independent copy, kept isolated
from sazz/samplers/.

Differs from AutomaticBoomerangSampler.sample(): (1) the rate-and-derivative
closure is forward-mode (grad + vmap + jvp) against a plain-function energy,
not the existing scalar Brent/PLI path -- see grid_bound.py; (2) on a
no-event grid_thinning() return, advances by stats["effective_horizon"],
not the raw horizon -- they differ only on a violation-driven early exit,
where the raw horizon would silently advance past unexamined trajectory time.
"""

import math
import time as _time
from functools import partial
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from ..utils.grid_bound import grid_thinning


class GridBoomerangSampler(nn.Module):
    """
    Boomerang sampler with the grid-based (Andral & Kamatani 2024) upper
    bound as its only thinning strategy.

    grad_target: PLAIN FUNCTION gradient of the (negative) log-target,
    built via torch.func.grad(energy) -- must not detach its input, so
    grid times can be differentiated through it via jvp.
    grid_t_max_init: initial adaptive horizon; period-appropriate for the
    Boomerang's ~pi-period rate (default pi/4), not the smaller Brent-tuned
    default used elsewhere.
    alpha_plus/alpha_minus: horizon growth/shrink (Algorithm 4) -- grown
    when the grid horizon (not refresh clock) was binding with no event;
    shrunk on ordinary rejections.
    alpha_violation: separate, more aggressive shrink on a bound violation.
    grid_kwargs: forwarded to grid_thinning (max_iter, min_window,
    max_violations, eps).
    """

    def __init__(
        self,
        grad_target: Callable[[Tensor], Tensor],
        D: int,
        refresh_rate: float = 0.1,
        grid_t_max_init: float = math.pi / 4,
        n_segments: int = 20,
        alpha_plus: float = 1.01,
        alpha_minus: float = 1.04,
        alpha_violation: float = 2.0,
        grid_kwargs: Optional[dict] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.D = D
        self.grad_target = grad_target
        self.refresh_rate = refresh_rate
        self.n_segments = n_segments
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.alpha_violation = alpha_violation
        self.grid_kwargs = grid_kwargs or {
            "max_iter": 200,
            "min_window": 1e-8,
            "max_violations": 10,
        }
        self.dtype = dtype
        self.device = torch.device(device)

        # Adaptive horizon -- persists across _grid_bound calls (sampler
        # state; grid_bound.py itself stays stateless).
        self._grid_t_max = float(grid_t_max_init)

        # Reference-measure quantities (set by preprocess)
        self.x_ref: Optional[Tensor] = None
        self.Sigma_inv: Optional[Tensor] = None
        self.Sigma: Optional[Tensor] = None
        self.Sigma_sqrt: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Preprocessing — identical in spirit to AutomaticBoomerangSampler's,
    # duplicated (not imported) per the isolation requirement.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def preprocess(self, x_ref: Tensor, Sigma_inv: Tensor, jitter: float = 1e-8):
        x_ref = x_ref.to(dtype=self.dtype, device=self.device).detach()
        Sigma_inv = Sigma_inv.to(dtype=self.dtype, device=self.device).detach()

        if Sigma_inv.ndim == 1:
            Sigma_inv_diag = Sigma_inv + jitter
            self.x_ref = x_ref
            self.Sigma_inv = Sigma_inv_diag
            self.Sigma = 1.0 / Sigma_inv_diag
            self.Sigma_sqrt = self.Sigma.sqrt()
        else:
            Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)
            Sigma_inv = Sigma_inv + jitter * torch.eye(self.D, dtype=self.dtype, device=self.device)
            Sigma = torch.linalg.inv(Sigma_inv)
            Sigma = 0.5 * (Sigma + Sigma.T)
            self.x_ref = x_ref
            self.Sigma_inv = Sigma_inv
            self.Sigma = Sigma
            self.Sigma_sqrt = torch.linalg.cholesky(Sigma)

    # ------------------------------------------------------------------
    # Dynamics — ON the computational graph. `t` must be a differentiable
    # tensor op (torch.cos/sin), not math.cos/sin, so that torch.func.jvp
    # can trace a tangent through it.
    # ------------------------------------------------------------------
    def trajectory(self, t: Tensor, x: Tensor, v: Tensor):
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)
        dx = x - self.x_ref
        x_t = self.x_ref + dx * cos_t + v * sin_t
        v_t = -dx * sin_t + v * cos_t
        return x_t, v_t

    def grad_U_excess(self, x: Tensor) -> Tensor:
        """
        Gradient of the EXCESS potential (target minus reference Gaussian).
        Named explicitly, not `gradU`: using the plain target gradient
        would silently give the BPS rate instead of Boomerang's, and the
        bug vanishes (excess gradient == 0) on an exactly matched Gaussian
        reference, only surfacing on a non-trivial target.
        """
        dx = x - self.x_ref
        prec_dx = self.Sigma_inv * dx if self.Sigma_inv.ndim == 1 else self.Sigma_inv @ dx
        return self.grad_target(x) - prec_dx

    def reflect_velocity(self, v: Tensor, grad: Tensor) -> Tensor:
        rate = torch.dot(v, grad)
        if self.Sigma_sqrt.ndim == 1:
            skew = self.Sigma_sqrt * grad
            denom = torch.dot(skew, skew)
            return v - 2.0 * rate / denom * (self.Sigma_sqrt * skew)
        else:
            skew = self.Sigma_sqrt.T @ grad
            denom = torch.dot(skew, skew)
            return v - 2.0 * rate / denom * (self.Sigma_sqrt @ skew)

    # ------------------------------------------------------------------
    # Rate closures for the grid bound
    # ------------------------------------------------------------------
    def _rate_scalar(self, t: float, x: Tensor, v: Tensor) -> float:
        """Off-graph, single-point signed rate -- for the accept/reject check."""
        with torch.no_grad():
            t_t = torch.as_tensor(t, dtype=self.dtype, device=self.device)
            x_t, v_t = self.trajectory(t_t, x, v)
            return float(torch.dot(v_t, self.grad_U_excess(x_t)))

    def _make_rate_and_grad_fn(self, x: Tensor, v: Tensor):
        """
        Batched (value, derivative) closure for g(t) = <v_t, grad_U_excess(x_t)>,
        anchored at (x, v). x, v are detached; only t is differentiable,
        per grid_bound.py's contract.
        """
        x = x.detach()
        v = v.detach()

        def g_scalar(t: Tensor) -> Tensor:
            x_t, v_t = self.trajectory(t, x, v)
            return torch.dot(v_t, self.grad_U_excess(x_t))

        def rate_and_grad_fn(t_batch: Tensor):
            # NOTE: jvp(grad(...)) here is currently broken on MPS (confirmed
            # torch 2.11.0 -- fails inside functorch's dual-tensor machinery
            # with a minimal repro having nothing to do with this codebase).
            # Works correctly on CPU; CUDA untested but should work (this is
            # a standard, well-supported torch.func composition there).
            y, dy = torch.func.vmap(
                lambda ti: torch.func.jvp(g_scalar, (ti,), (torch.ones_like(ti),))
            )(t_batch)
            return y, dy

        return rate_and_grad_fn

    # ------------------------------------------------------------------
    # Grid bound dispatch — single window per call (design decision #2).
    # ------------------------------------------------------------------
    def _grid_bound(self, pos: Tensor, vel: Tensor, dt_refresh: float) -> tuple[float, dict]:
        """
        Handle exactly one window, horizon = min(self._grid_t_max, dt_refresh)
        -- sample()'s own loop provides "the next window" on its next
        iteration rather than grid_thinning needing an internal loop.
        Mutates self._grid_t_max (grow/shrink) after the call resolves.
        """
        grid_was_binding = self._grid_t_max <= dt_refresh
        horizon = min(self._grid_t_max, dt_refresh)

        rate_and_grad_fn = self._make_rate_and_grad_fn(pos, vel)
        rate_scalar_fn = partial(self._rate_scalar, x=pos.detach(), v=vel.detach())

        tau, stats = grid_thinning(
            rate_and_grad_fn, rate_scalar_fn, horizon,
            n_segments=self.n_segments, device=self.device, dtype=self.dtype,
            diagnostics=True, **self.grid_kwargs,
        )

        # --- t_max adaptation (Algorithm 4) ---
        if stats["violated"]:
            self._grid_t_max /= self.alpha_violation
        elif stats["rejected_in_window"]:
            self._grid_t_max /= self.alpha_minus
        elif tau == math.inf and grid_was_binding:
            # No event, and the grid horizon (not the refresh clock) was
            # the binding constraint over the FULL window -- only then is
            # it safe to conclude the horizon could have been longer.
            eff = stats["effective_horizon"]
            if eff is not None and eff >= horizon - 1e-12:
                self._grid_t_max *= self.alpha_plus

        return tau, stats

    # ------------------------------------------------------------------
    # Velocity refresh
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _refresh_velocity(self) -> Tensor:
        z = torch.randn(self.D, dtype=self.dtype, device=self.device)
        return self.Sigma_sqrt * z if self.Sigma_sqrt.ndim == 1 else self.Sigma_sqrt @ z

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------
    def sample(self, N: int, x0: Optional[Tensor] = None, diagnostics: bool = True) -> dict:
        assert self.x_ref is not None, "Call preprocess() first."

        positions = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        velocities = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        times = torch.zeros(N, dtype=self.dtype, device=self.device)

        if x0 is None:
            positions[0] = self.x_ref + self._refresh_velocity()
        else:
            positions[0] = x0.to(dtype=self.dtype, device=self.device)
        velocities[0] = self._refresh_velocity()
        times[0] = 0.0

        dt_refresh = float(np.random.exponential(1.0 / self.refresh_rate))
        time_passed = 0.0
        current_time = 0.0
        grad_evals = 0
        total_bound_violations = 0
        diag_log = []
        grid_t_max_log = []

        pbar = tqdm(total=N, desc="GridBoomerang", unit="skel")
        pbar.update(1)
        iteration = 1

        while iteration < N:
            _t0 = _time.perf_counter()
            n = iteration

            x_prev = positions[n - 1]
            v_prev = velocities[n - 1]

            with torch.no_grad():
                pos, vel = self.trajectory(
                    torch.as_tensor(time_passed, dtype=self.dtype, device=self.device),
                    x_prev, v_prev,
                )

            tau, stats = self._grid_bound(pos.detach(), vel.detach(), dt_refresh)
            grad_evals += stats.get("rate_evals", 0)
            total_bound_violations += stats.get("bound_violations", 0)
            grid_t_max_log.append(self._grid_t_max)

            row = {
                "rate_evals": stats.get("rate_evals", 0),
                "horizon": min(self._grid_t_max, dt_refresh),
                "time": current_time,
                "event_type": None,
                "accepted": None,
                "wall_seconds": None,
                "bound_violations": stats.get("bound_violations", 0),
                "max_ratio": stats.get("max_ratio", 0.0),
            }

            event_accepted = False
            horizon_used = min(self._grid_t_max, dt_refresh)

            if tau < horizon_used:
                row["event_type"] = "bounce"
                row["accepted"] = True
                event_accepted = True
            else:
                # Advance by effective_horizon, not horizon_used -- see
                # grid_bound.py's module docstring.
                row["event_type"] = "no_event"
                eff = stats.get("effective_horizon")
                advance = eff if eff is not None else horizon_used
                time_passed += advance
                current_time += advance
                dt_refresh -= advance

            if event_accepted:
                current_time += tau

                pos_prop, vel_prop = self.trajectory(
                    torch.as_tensor(time_passed + tau, dtype=self.dtype, device=self.device),
                    x_prev, v_prev,
                )
                grad_at_prop = self.grad_U_excess(pos_prop)
                vel_reflected = self.reflect_velocity(vel_prop, grad_at_prop)

                positions[n] = pos_prop.detach()
                velocities[n] = vel_reflected.detach()
                times[n] = times[n - 1] + time_passed + tau

                grad_evals += 1
                row["rate_evals"] += 1

                iteration += 1
                time_passed = 0.0
                dt_refresh -= tau
                pbar.update(1)

            if dt_refresh <= 1e-14:
                row["wall_seconds"] = _time.perf_counter() - _t0
                diag_log.append(row)

                refresh_row = {
                    "rate_evals": 0, "horizon": 0.0, "time": current_time,
                    "event_type": "refresh", "accepted": None, "wall_seconds": 0.0,
                    "bound_violations": 0, "max_ratio": 0.0,
                }

                if iteration < N:
                    n = iteration
                    with torch.no_grad():
                        pos_ref, _ = self.trajectory(
                            torch.as_tensor(time_passed, dtype=self.dtype, device=self.device),
                            positions[n - 1], velocities[n - 1],
                        )
                    positions[n] = pos_ref
                    velocities[n] = self._refresh_velocity()
                    times[n] = times[n - 1] + time_passed

                    iteration += 1
                    time_passed = 0.0
                    pbar.update(1)

                dt_refresh = float(np.random.exponential(1.0 / self.refresh_rate))
                diag_log.append(refresh_row)
                pbar.set_postfix_str(
                    f"t={current_time:.3f} t_max={self._grid_t_max:.4f} viol={total_bound_violations}",
                    refresh=False,
                )
                continue

            row["wall_seconds"] = _time.perf_counter() - _t0
            pbar.set_postfix_str(
                f"t={current_time:.3f} t_max={self._grid_t_max:.4f} viol={total_bound_violations}",
                refresh=False,
            )
            diag_log.append(row)

        pbar.close()

        if diagnostics:
            self._print_diagnostics(diag_log, N, grad_evals, times[iteration - 1], total_bound_violations)

        return {
            "positions": positions,
            "velocities": velocities,
            "times": times,
            "diagnostics": diag_log,
            "gradient_evals": grad_evals,
            "bound_violations": total_bound_violations,
            "grid_t_max_log": grid_t_max_log,
        }

    @staticmethod
    def _print_diagnostics(diag_log, N, grad_evals, final_time, total_bound_violations):
        import pandas as pd

        df = pd.DataFrame(diag_log)
        n_accept = len(df[df["event_type"] == "bounce"])

        print("\n=== GridBoomerang Diagnostics ===")
        print(f"Total gradient evals  : {grad_evals}")
        print(f"Grad evals / skeleton : {grad_evals / max(N, 1):.1f}")
        print(f"Accepted bounces      : {n_accept}")

        if total_bound_violations > 0:
            print(
                f"\n*** BOUND VIOLATIONS: {total_bound_violations} — the grid was too coarse "
                f"for the local curvature at least once. Samples drawn in the affected "
                f"window(s) used a bound that was not actually valid there; this is a "
                f"validity signal for the run, not a tunable knob. Consider a finer grid "
                f"(higher n_segments) or a smaller grid_t_max_init. ***\n"
            )
        else:
            print("Bound violations      : 0")

        print("\n=== Event breakdown ===")
        for etype in ["bounce", "no_event", "refresh"]:
            sub = df[df["event_type"] == etype]
            if len(sub) > 0:
                print(f"  {etype:10s}: {len(sub):5d} events, mean horizon={sub['horizon'].mean():.4f}")

        wall = df["wall_seconds"].dropna()
        if len(wall) > 0:
            print(f"\nMean wall-sec / iter : {wall.mean():.6f}")
            print(f"Total wall-sec       : {wall.sum():.2f}")

        print(f"\nSimulation time reached: {float(final_time):.4f}")
