"""
AutomaticStickyBoomerangSampler.py

Sticky Boomerang sampler in PyTorch, inheriting from AutomaticBoomerangSampler.
Merges the Brent and PLI sticky variants into a single class.

Adds three mechanics on top of the base Boomerang:
  - **Freeze**: when an active coordinate's trajectory hits x_i = 0,
    that coordinate is frozen (set to zero with zero velocity) and a
    thaw deadline is drawn from Exp(kappa_i * |v_i|).
  - **Thaw**: when the thaw deadline fires, the coordinate is
    reactivated with its stored pre-freeze velocity.
  - **Active-subspace reflection & refresh**: reflections and velocity
    refreshments act only on the unfrozen (active) coordinates.

All freeze/thaw/hit-detection logic is OFF the computational graph
(scheduling decisions).  The on-graph chain remains:
    trajectory → gradU → reflect → store skeleton point.

"""

import math
import time as _time
from functools import partial
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from .AutomaticBoomerangSampler import AutomaticBoomerangSampler


class StickyAutomaticBoomerangSampler(AutomaticBoomerangSampler):
    """
    Sticky Boomerang sampler with automatic rate bounding.

    Parameters
    ----------
    kappa : float | Tensor
        Stickiness parameter(s).  Scalar broadcasts to all coordinates;
        a 1-D tensor of length D gives per-coordinate stickiness.
    cold_start_threshold : float | None
        If set, coordinates with |x_ref_i| < threshold are frozen at
        initialisation with a synthetic thaw deadline.
    (all other parameters inherited from AutomaticBoomerangSampler)
    """

    def __init__(
        self,
        grad_target,
        D: int,
        kappa: float = 1.0,
        can_freeze: list[bool] = None,
        refresh_rate: float = 0.1,
        thinning: Literal["brent", "pli"] = "pli",
        t_max: float = 0.1,
        pli_kwargs: Optional[dict] = None,
        cold_start_threshold: Optional[float] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ):
        super().__init__(
            grad_target=grad_target,
            D=D,
            refresh_rate=refresh_rate,
            thinning=thinning,
            t_max=t_max,
            pli_kwargs=pli_kwargs,
            dtype=dtype,
            device=device,
        )

        # --- Sticky specifics ---
        if isinstance(kappa, (int, float)):
            self.kappa = torch.full((D,), kappa, dtype=dtype, device=self.device)
        else:
            self.kappa = torch.as_tensor(kappa, dtype=dtype, device=self.device)
            
        if can_freeze is None:
            self.can_freeze = torch.ones(D, dtype=torch.bool, device=self.device)
        else:
            self.can_freeze = torch.as_tensor(can_freeze, dtype=torch.bool, device=self.device)
    
        self.cold_start_threshold = cold_start_threshold
        # Mutable state (reset between runs)
        self.frozen_mask = torch.zeros(D, dtype=torch.bool, device=self.device)
        self.frozen_velocity = torch.zeros(D, dtype=dtype, device=self.device)
        self.thaw_deadline = torch.full((D,), float("inf"), dtype=dtype, device=self.device)

    # ------------------------------------------------------------------
    # Sticky dynamics — trajectory with frozen coords zeroed
    # ------------------------------------------------------------------
    def trajectory_sticky(self, t: float, x: Tensor, v: Tensor):
        """
        Boomerang trajectory with frozen coordinates clamped to zero.
        """
        x_t, v_t = self.trajectory(t, x, v)
        # Zero out frozen coords (detach-safe: in-place on clones)
        x_t = x_t.clone()
        v_t = v_t.clone()
        x_t[self.frozen_mask] = 0.0
        v_t[self.frozen_mask] = 0.0
        return x_t, v_t

    # ------------------------------------------------------------------
    # Freeze / thaw bookkeeping — OFF graph
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _freeze(self, i: int, v: Tensor, current_time: float):
        """Freeze coordinate i: record velocity, draw thaw deadline."""
        self.frozen_mask[i] = True
        self.frozen_velocity[i] = v[i].item()
        rate_i = self.kappa[i].item() * abs(v[i].item())
        if rate_i > 1e-14:
            self.thaw_deadline[i] = current_time + float(
                np.random.exponential(1.0 / rate_i)
            )
        else:
            self.thaw_deadline[i] = float("inf")

    @torch.no_grad()
    def _thaw(self, i: int):
        """Thaw coordinate i: clear mask and deadline."""
        self.frozen_mask[i] = False
        self.thaw_deadline[i] = float("inf")

    # ------------------------------------------------------------------
    # Hitting-time detection — OFF graph (pure scheduling)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _next_hitting_event(self, x: Tensor, v: Tensor):
        """
        Find the earliest time an active coordinate's trajectory hits x_i = 0.

        Solves  a*cos(t) + b*sin(t) = c  per active coordinate, where
            a = x_i - x_ref_i,  b = v_i,  c = -x_ref_i.

        Returns (dt_hit, i_hit).  dt_hit = inf if no hit exists.
        """
        t_hit = float("inf")
        i_hit = None

        #active_indices = torch.where(~self.frozen_mask)[0]
        active_indices = torch.where(~self.frozen_mask & self.can_freeze)[0]
        x_np = x.cpu().numpy()
        v_np = v.cpu().numpy()
        xref_np = self.x_ref.cpu().numpy()

        for idx in active_indices:
            i = idx.item()
            a = float(x_np[i] - xref_np[i])
            b = float(v_np[i])
            c = float(-xref_np[i])

            R = math.hypot(a, b)
            if R < 1e-14:
                continue
            if abs(c) > R + 1e-14:
                continue  # no solution exists

            phi = math.atan2(b, a)
            delta = math.acos(max(-1.0, min(1.0, c / R)))

            t_i = float("inf")
            for base in (phi - delta, phi + delta):
                candidate = base % (2.0 * math.pi)
                if candidate < 1e-10:
                    candidate += 2.0 * math.pi
                t_i = min(t_i, candidate)

            if t_i < t_hit:
                t_hit = t_i
                i_hit = i

        return t_hit, i_hit

    @torch.no_grad()
    def _next_thaw_event(self, current_time: float):
        """
        Return (dt_thaw, i_thaw) — time until next thaw and which coord.
        dt_thaw is relative to current_time.
        """
        frozen_idx = torch.where(self.frozen_mask)[0]
        if len(frozen_idx) == 0:
            return float("inf"), None
        deadlines = self.thaw_deadline[frozen_idx]
        local_min = torch.argmin(deadlines)
        i_thaw = frozen_idx[local_min].item()
        dt_thaw = max(self.thaw_deadline[i_thaw].item() - current_time, 0.0)
        return dt_thaw, i_thaw

    # ------------------------------------------------------------------
    # Active-subspace reflection — ON graph
    # ------------------------------------------------------------------
    def reflect_velocity_sticky(self, v: Tensor, grad: Tensor) -> Tensor:
        """
        Boomerang reflection restricted to the active (unfrozen) subspace.
        Falls back to full reflection when nothing is frozen.
        """
        if not torch.any(self.frozen_mask):
            return self.reflect_velocity(v, grad)

        active = ~self.frozen_mask
        v_new = v.clone()

        v_a = v[active]
        g_a = grad[active]
        Sigma_a = self.Sigma[active][:, active]

        rate = torch.dot(v_a, g_a)
        Sigma_g = Sigma_a @ g_a
        denom = torch.dot(g_a, Sigma_g)

        if denom.item() <= 1e-14:
            return v_new

        v_new[active] = v_a - 2.0 * rate / denom * Sigma_g
        v_new[self.frozen_mask] = 0.0
        return v_new

    # ------------------------------------------------------------------
    # Active-subspace velocity refresh — OFF graph
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _refresh_velocity_sticky(self) -> Tensor:
        """Sample velocity from N(0, Sigma) on active coords only."""
        v = torch.zeros(self.D, dtype=self.dtype, device=self.device)
        active = ~self.frozen_mask

        if torch.any(active):
            n_active = int(active.sum().item())
            Sigma_a = self.Sigma[active][:, active]
            jitter = 1e-12 * torch.eye(n_active, dtype=self.dtype, device=self.device)
            chol_a = torch.linalg.cholesky(Sigma_a + jitter)
            z = torch.randn(n_active, dtype=self.dtype, device=self.device)
            v[active] = chol_a @ z

        return v

    # ------------------------------------------------------------------
    # Rate function for sticky dynamics — OFF graph
    # ------------------------------------------------------------------
    def _rate_numpy_sticky(self, t: float, x_np: np.ndarray, v_np: np.ndarray) -> float:
        """
        Rate along the sticky trajectory (frozen coords zeroed).
        Used by the bounding utilities.
        """
        with torch.no_grad():
            x = torch.as_tensor(x_np, dtype=self.dtype, device=self.device)
            v = torch.as_tensor(v_np, dtype=self.dtype, device=self.device)
            x_t, v_t = self.trajectory_sticky(t, x, v)
            inner = torch.dot(v_t, self.gradU(x_t))
        return float(inner)

    # ------------------------------------------------------------------
    # Override: event-time finding uses sticky rate
    # ------------------------------------------------------------------
    def _find_event_time(
        self, pos: Tensor, vel: Tensor, horizon: float
    ) -> tuple[float, dict]:
        """
        Dispatch to bounding strategy using the sticky rate function.
        """
        pos_np = pos.cpu().numpy()
        vel_np = vel.cpu().numpy()

        if self.thinning == "brent":
            return self._brent_bound_sticky(pos_np, vel_np, horizon)
        elif self.thinning == "pli":
            return self._pli_bound_sticky(pos_np, vel_np, horizon)
        else:
            raise ValueError(f"Unknown thinning method: {self.thinning}")

    def _brent_bound_sticky(
        self, pos_np: np.ndarray, vel_np: np.ndarray, horizon: float
    ) -> tuple[float, dict]:
        from sazz.utils.bounding import brent

        neg_rate_fn = lambda t: -max(self._rate_numpy_sticky(t, pos_np, vel_np), 0.0)

        x_star, stats = brent(neg_rate_fn, 0.0, horizon, diagnostics=True)
        lambda_max = max(-neg_rate_fn(x_star), 0.0)

        if lambda_max <= 1e-14:
            tau_star = float("inf")
        else:
            tau_star = -math.log(np.random.random()) / lambda_max

        stats["lambda_max"] = lambda_max
        stats["tau"] = tau_star
        return tau_star, stats

    def _pli_bound_sticky(
        self, pos_np: np.ndarray, vel_np: np.ndarray, horizon: float
    ) -> tuple[float, dict]:
        from sazz.utils.bounding import piecewise_thinning_sinusoidal_second_order

        rate_fn = partial(self._rate_numpy_sticky, x_np=pos_np, v_np=vel_np)
        tau, stats = piecewise_thinning_sinusoidal_second_order(
            rate_fn, horizon, diagnostics=True, **self.pli_kwargs
        )
        return tau, stats

    # ------------------------------------------------------------------
    # Reset sticky state
    # ------------------------------------------------------------------
    def _reset_sticky_state(self):
        """Clear all freeze/thaw bookkeeping."""
        self.frozen_mask.zero_()
        self.frozen_velocity.zero_()
        self.thaw_deadline.fill_(float("inf"))
#near_zero_mask = (positions[0].abs() < self.cold_start_threshold) & self.can_freeze
    # ------------------------------------------------------------------
    # Cold start: optionally freeze near-zero coordinates at init
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_cold_start(self, positions, velocities):
        if self.cold_start_threshold is None:
            return
        for i in range(self.D):
            #if abs(positions[0, i].item()) < self.cold_start_threshold \
            #and self.kappa[i].item() < 1e6:        # skip never-stick coords
            if positions[0].abs() < self.cold_start_threshold \
            and self.can_freeze:
                v_init = float(velocities[0, i].item())
                rate_i = self.kappa[i].item() * abs(v_init)
                if rate_i < 1e-14:
                    continue
                positions[0, i] = 0.0
                velocities[0, i] = 0.0
                self.frozen_mask[i] = True
                self.frozen_velocity[i] = v_init
                self.thaw_deadline[i] = float(np.random.exponential(1.0 / rate_i))

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------
    def sample(
        self,
        N: int,
        x0: Optional[Tensor] = None,
        diagnostics: bool = True,
    ) -> dict:
        """
        Run the Sticky Boomerang sampler for N skeleton points.

        Returns dict with keys:
            "positions", "velocities", "times", "diagnostics",
            "gradient_evals", "frozen_mask_final"
        """
        assert self.x_ref is not None, "Call preprocess() first."

        # --- Storage ---
        positions = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        velocities = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        times = torch.zeros(N, dtype=self.dtype, device=self.device)

        # --- Initialise ---
        self._reset_sticky_state()

        if x0 is None:
            positions[0] = self.x_ref + self._refresh_velocity()
        else:
            positions[0] = x0.to(dtype=self.dtype, device=self.device)
        velocities[0] = self._refresh_velocity()
        times[0] = 0.0

        # Optional cold start
        self._apply_cold_start(positions, velocities)

        dt_refresh = float(np.random.exponential(1.0 / self.refresh_rate))
        time_passed = 0.0
        current_time = 0.0
        grad_evals = 0
        diag_log = []

        pbar = tqdm(total=N, desc="Sticky Boomerang", unit="skel")
        pbar.update(1)
        iteration = 1

        while iteration < N:
            _t0 = _time.perf_counter()
            n = iteration
            x_prev = positions[n - 1]
            v_prev = velocities[n - 1]

            # Current position along sticky trajectory
            with torch.no_grad():
                pos, vel = self.trajectory_sticky(time_passed, x_prev, v_prev)

            # --- Sticky event scheduling (all OFF graph) ---
            dt_hit, i_hit = self._next_hitting_event(pos, vel)
            dt_thaw, i_thaw = self._next_thaw_event(current_time)

            if self.thinning == "brent":
                horizon = min(self.t_max, dt_refresh, dt_hit, dt_thaw)
            else:
                horizon = min(dt_refresh, dt_hit, dt_thaw)

            # --- Find bounce event time (OFF graph) ---
            tau, stats = self._find_event_time(pos.detach(), vel.detach(), horizon)
            grad_evals += stats.get("rate_evals", 0)

            n_frozen = int(self.frozen_mask.sum().item())
            n_active = self.D - n_frozen

            row = {
                "rate_evals": stats.get("rate_evals", 0),
                "horizon": horizon,
                "time": current_time,
                "event_type": None,
                "accepted": None,
                "n_frozen": n_frozen,
                "n_active": n_active,
                "sparsity": n_frozen / self.D,
                "wall_seconds": None,
                "proposals": stats.get("proposals", 0),
                "bound_violations": stats.get("bound_violations", 0),
                "max_ratio": stats.get("max_ratio", 0.0),
            }

            event_accepted = False

            # ---- Bounce proposal beat the horizon ----
            if tau < horizon:
                if self.thinning == "brent":
                    row["event_type"] = "bounce"
                    u = np.random.random()
                    pos_np = pos.detach().cpu().numpy()
                    vel_np = vel.detach().cpu().numpy()
                    lambda_star = max(
                        self._rate_numpy_sticky(tau, pos_np, vel_np), 0.0
                    )
                    lambda_max = stats.get("lambda_max", 1.0)
                    p_accept = lambda_star / lambda_max if lambda_max > 1e-14 else 0.0
                    grad_evals += 1
                    row["rate_evals"] += 1

                    if u <= p_accept:
                        event_accepted = True
                        row["accepted"] = True
                    else:
                        row["accepted"] = False
                        time_passed += tau
                        current_time += tau
                        dt_refresh -= tau
                else:  # PLI: returned event is already accepted
                    row["event_type"] = "bounce"
                    row["accepted"] = True
                    event_accepted = True

            # ---- Materialise accepted bounce ON graph ----
            if event_accepted:
                current_time += tau
                pos_prop, vel_prop = self.trajectory_sticky(
                    time_passed + tau, x_prev, v_prev
                )
                grad_at_prop = self.gradU(pos_prop)
                vel_reflected = self.reflect_velocity_sticky(vel_prop, grad_at_prop)

                positions[n] = pos_prop.detach()
                velocities[n] = vel_reflected.detach()
                times[n] = times[n - 1] + time_passed + tau

                grad_evals += 1
                row["rate_evals"] += 1

                iteration += 1
                time_passed = 0.0
                dt_refresh -= tau
                pbar.update(1)

            # ---- Horizon fired: freeze, thaw, or fall-through ----
            elif tau >= horizon:
                time_passed += horizon
                current_time += horizon
                dt_refresh -= horizon

                tol = 1e-12

                # --- Freeze event ---
                if abs(horizon - dt_hit) < tol and i_hit is not None:
                    row["event_type"] = "freeze"
                    with torch.no_grad():
                        pos_now, vel_now = self.trajectory_sticky(
                            time_passed, x_prev, v_prev
                        )
                    self._freeze(i_hit, vel_now, current_time)
                    pos_now = pos_now.clone()
                    vel_now = vel_now.clone()
                    pos_now[i_hit] = 0.0
                    vel_now[i_hit] = 0.0

                    positions[n] = pos_now
                    velocities[n] = vel_now
                    times[n] = times[n - 1] + time_passed

                    iteration += 1
                    time_passed = 0.0
                    pbar.update(1)

                # --- Thaw event ---
                elif abs(horizon - dt_thaw) < tol and i_thaw is not None:
                    row["event_type"] = "thaw"
                    with torch.no_grad():
                        pos_now, vel_now = self.trajectory_sticky(
                            time_passed, x_prev, v_prev
                        )
                    vel_now = vel_now.clone()
                    vel_now[i_thaw] = self.frozen_velocity[i_thaw]
                    self._thaw(i_thaw)

                    positions[n] = pos_now
                    velocities[n] = vel_now
                    times[n] = times[n - 1] + time_passed

                    iteration += 1
                    time_passed = 0.0
                    pbar.update(1)

                else:
                    # No structural event — just time advance (no_event)
                    if row["event_type"] is None:
                        row["event_type"] = "no_event"

            # ---- Velocity refresh ----
            if dt_refresh <= 1e-14:
                row["wall_seconds"] = _time.perf_counter() - _t0
                diag_log.append(row)

                refresh_row = {
                    "rate_evals": 0,
                    "horizon": 0.0,
                    "time": current_time,
                    "event_type": "refresh",
                    "accepted": None,
                    "n_frozen": int(self.frozen_mask.sum().item()),
                    "n_active": self.D - int(self.frozen_mask.sum().item()),
                    "sparsity": int(self.frozen_mask.sum().item()) / self.D,
                    "wall_seconds": 0.0,
                    "proposals": 0,
                    "bound_violations": 0,
                    "max_ratio": 0.0,
                }

                if iteration < N:
                    n = iteration
                    with torch.no_grad():
                        pos_ref, _ = self.trajectory_sticky(
                            time_passed, positions[n - 1], velocities[n - 1]
                        )
                    positions[n] = pos_ref
                    velocities[n] = self._refresh_velocity_sticky()
                    times[n] = times[n - 1] + time_passed

                    iteration += 1
                    time_passed = 0.0
                    pbar.update(1)

                dt_refresh = float(np.random.exponential(1.0 / self.refresh_rate))
                diag_log.append(refresh_row)
                pbar.set_postfix_str(f"t={current_time:.3f}", refresh=False)
                continue

            row["wall_seconds"] = _time.perf_counter() - _t0
            pbar.set_postfix_str(f"t={current_time:.3f}", refresh=False)
            diag_log.append(row)

        pbar.close()

        if diagnostics:
            self._print_sticky_diagnostics(diag_log, N, grad_evals, times[iteration - 1])

        return {
            "positions": positions,
            "velocities": velocities,
            "times": times,
            "diagnostics": diag_log,
            "gradient_evals": grad_evals,
            "frozen_mask_final": self.frozen_mask.clone(),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _print_sticky_diagnostics(
        diag_log: list[dict], N: int, grad_evals: int, final_time: Tensor
    ):
        import pandas as pd

        df = pd.DataFrame(diag_log)

        n_accept = len(
            df[(df["event_type"] == "bounce") & (df["accepted"] == True)]
        )
        n_reject = len(
            df[(df["event_type"] == "bounce") & (df["accepted"] == False)]
        )

        print("\n=== Sampler Diagnostics ===")
        print(f"Total gradient evals  : {grad_evals}")
        print(f"Grad evals / skeleton : {grad_evals / max(N, 1):.1f}")
        if "sparsity" in df.columns:
            print(f"Mean sparsity (frac frozen): {df['sparsity'].mean():.3f}")
            print(f"Max simultaneous frozen    : {int(df['n_frozen'].max())} / {N}")

        print("\n=== Thinning ===")
        if "proposals" in df.columns and df["proposals"].sum() > 0:
            print(f"Bound violations : {int(df['bound_violations'].sum())}")
            print(f"Max ratio        : {df['max_ratio'].max():.4f}")
            print(
                f"Proposals/call   : {df['proposals'].mean():.1f} mean, "
                f"{int(df['proposals'].max())} max"
            )
        if n_reject > 0:
            print(f"Accept / Reject  : {n_accept} / {n_reject}")
        else:
            print(f"Accepted bounces : {n_accept}")

        print("\n=== Event breakdown ===")
        for etype in ["bounce", "freeze", "thaw", "refresh", "no_event"]:
            sub = df[df["event_type"] == etype]
            if len(sub) > 0:
                print(
                    f"  {etype:10s}: {len(sub):5d} events, "
                    f"mean horizon={sub['horizon'].mean():.4f}"
                )

        print("\n=== Timing ===")
        wall = df["wall_seconds"].dropna()
        if len(wall) > 0:
            print(f"Mean wall-sec / iter : {wall.mean():.6f}")
            print(f"Total wall-sec       : {wall.sum():.2f}")

        print(f"\nSimulation time reached: {float(final_time):.4f}")
