"""
StickyAutomaticZigZagSampler.py

Sticky Zig-Zag sampler in PyTorch, inheriting from AutomaticZigZagSampler.
Merges the Brent and PLI sticky variants into a single class.

Adds three mechanics on top of the base Zig-Zag:
  - **Freeze**: when an active coordinate's linear trajectory hits
    x_i = 0, that coordinate is frozen (clamped to zero with zero
    velocity) and a thaw deadline is drawn from Exp(kappa_i * |v_i|),
    where v_i is the *pre-freeze* velocity (always +/- 1 for Zig-Zag).
  - **Thaw**: when the deadline fires, the coordinate is reactivated
    with its stored pre-freeze velocity.
  - **Active-subspace flip**: events flip a single coordinate from the
    active (unfrozen) set, sampled from the categorical proportional to
    the per-coordinate switching rates restricted to active coords. The
    gamma floor only applies to active coords (otherwise frozen coords
    would carry rate gamma and could be picked).

All freeze/thaw/hit-detection logic is OFF the computational graph
(scheduling decisions). The on-graph chain remains:
    trajectory -> grad_target -> flip -> store skeleton point.

Notes vs. sticky Boomerang
--------------------------
- Hitting time is the simplest possible: t_i = -x_i / v_i, taken over
  active coords with v_i moving toward zero (sign(v_i) = -sign(x_i)).
  No 2π wrap, no trig.
- No reference measure, no velocity refresh clock. Cold start, when
  enabled, freezes every coordinate with finite kappa at initialisation
  rather than thresholding |x_ref| (since there is no x_ref).
- A frozen coordinate is detected on the skeleton by v_i == 0 (active
  Zig-Zag velocities are always exactly +/- 1).
"""

import time as _time
from functools import partial
from typing import Optional, Literal

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from sazz.utils.bounding import brent, piecewise_thinning

from .AutomaticZigZagSampler_torch import AutomaticZigZagSampler


class StickyAutomaticZigZagSampler(AutomaticZigZagSampler):
    """
    Sticky Zig-Zag sampler with automatic rate bounding.

    Parameters
    ----------
    kappa : float | Tensor
        Stickiness parameter(s). Scalar broadcasts to all coordinates;
        a 1-D tensor of length D gives per-coordinate stickiness.
        Recall the convention: larger kappa = faster thawing = less
        sticky. kappa = 0 means permanently frozen. To exempt a
        coordinate (e.g. an intercept), set kappa_i to a large value
        like 1e6 so it thaws instantly.
    cold_start_threshold : bool | None
        If True (or any truthy value), freeze all coordinates with
        finite kappa at initialisation. Default None (off).
    (all other parameters inherited from AutomaticZigZagSampler)
    """

    def __init__(
        self,
        grad_target,
        D: int,
        kappa: float = 1.0,
        t_max: float = 1.0,
        gamma: float = 0.01,
        thinning: Literal["brent", "pli"] = "pli",
        adapt_t_max: bool = False,
        pli_kwargs: Optional[dict] = None,
        cold_start_threshold: Optional[bool] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ):
        super().__init__(
            grad_target=grad_target,
            D=D,
            t_max=t_max,
            gamma=gamma,
            thinning=thinning,
            adapt_t_max=adapt_t_max,
            pli_kwargs=pli_kwargs,
            dtype=dtype,
            device=device,
        )

        # --- Sticky specifics ---
        if isinstance(kappa, (int, float)):
            self.kappa = torch.full((D,), kappa, dtype=dtype, device=self.device)
        else:
            self.kappa = torch.as_tensor(kappa, dtype=dtype, device=self.device)

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
        Linear Zig-Zag trajectory with frozen coords clamped to zero.
        """
        x_t, v_t = self.trajectory(t, x, v)
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
        """Freeze coord i: store its (pre-freeze) velocity, draw thaw deadline."""
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
        """Thaw coord i: clear mask and deadline."""
        self.frozen_mask[i] = False
        self.thaw_deadline[i] = float("inf")

    # ------------------------------------------------------------------
    # Hitting-time detection — OFF graph (pure scheduling)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _next_hitting_event(self, x: Tensor, v: Tensor):
        """
        Find the earliest time an active coordinate's linear trajectory
        hits x_i = 0.

        Solves  x_i + t * v_i = 0  =>  t_i = -x_i / v_i  per active coord,
        keeping only positive solutions (i.e. coords moving toward zero).

        Returns (dt_hit, i_hit). dt_hit = inf if no active coord is
        moving toward zero.
        """
        x_np = x.cpu().numpy()
        v_np = v.cpu().numpy()
        active_np = (~self.frozen_mask).cpu().numpy()

        t_hit = float("inf")
        i_hit = None

        for i in np.flatnonzero(active_np):
            xi = float(x_np[i])
            vi = float(v_np[i])

            # Skip degenerate cases:
            #   |v_i| ~ 0: not moving (shouldn't happen for active coords,
            #              but guard anyway)
            #   x_i ~ 0:   already at zero (would freeze instantly; the
            #              previous skeleton point should have caught this)
            if abs(vi) < 1e-14:
                continue
            if abs(xi) < 1e-14:
                continue

            t_i = -xi / vi
            if t_i <= 0.0:
                continue  # moving away from zero

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
    # Rate function for sticky dynamics — OFF graph
    # ------------------------------------------------------------------
    def _rate_numpy_sticky(self, t: float, x_np: np.ndarray, v_np: np.ndarray) -> float:
        """
        Total Zig-Zag switching rate at time t along the *sticky* linear
        trajectory:
            lambda(t) = sum_{i active} max(0, v_i * grad_i U(x + t v)) + gamma
        Frozen coords contribute zero (their v_i and x_i are clamped via
        trajectory_sticky); the gamma floor is applied only to the
        active subset.
        """
        with torch.no_grad():
            x = torch.as_tensor(x_np, dtype=self.dtype, device=self.device)
            v = torch.as_tensor(v_np, dtype=self.dtype, device=self.device)
            x_t, v_t = self.trajectory_sticky(t, x, v)
            grad = self.grad_target(x_t)
            per_coord = torch.clamp(v_t * grad, min=0.0) + self.gamma
            # Restrict gamma floor to active coords
            per_coord = per_coord * (~self.frozen_mask).to(self.dtype)
            rate = per_coord.sum()
        return float(rate)

    def _per_coord_rates_numpy_sticky(
        self, t: float, x_np: np.ndarray, v_np: np.ndarray
    ) -> np.ndarray:
        """
        Per-coordinate rates (active coords only; frozen coords get 0)
        used to draw which active coordinate to flip at an event.
        """
        with torch.no_grad():
            x = torch.as_tensor(x_np, dtype=self.dtype, device=self.device)
            v = torch.as_tensor(v_np, dtype=self.dtype, device=self.device)
            x_t, v_t = self.trajectory_sticky(t, x, v)
            grad = self.grad_target(x_t)
            per_coord = torch.clamp(v_t * grad, min=0.0) + self.gamma
            per_coord = per_coord * (~self.frozen_mask).to(self.dtype)
        return per_coord.detach().cpu().numpy().astype(np.float64)

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
        neg_rate_fn = lambda t: -max(self._rate_numpy_sticky(t, pos_np, vel_np), 0.0)

        x_star, stats = brent(neg_rate_fn, 0.0, horizon, diagnostics=True)
        # if x_star > horizon:
        #     x_star = horizon
        lambda_max = max(-neg_rate_fn(x_star), 0.0)

        if lambda_max <= 1e-14:
            tau_star = float("inf")
        else:
            tau_star = -np.log(np.random.random()) / lambda_max

        stats["lambda_max"] = lambda_max
        stats["tau"] = tau_star
        return tau_star, stats

    def _pli_bound_sticky(
        self, pos_np: np.ndarray, vel_np: np.ndarray, horizon: float
    ) -> tuple[float, dict]:
        rate_fn = partial(self._rate_numpy_sticky, x_np=pos_np, v_np=vel_np)
        tau, stats = piecewise_thinning(
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

    # ------------------------------------------------------------------
    # Cold start: optionally freeze all coords with finite kappa at init
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_cold_start(self, positions, velocities):
        if self.cold_start_threshold is None:
            return
        for i in range(self.D):
            if abs(positions[0, i].item()) < self.cold_start_threshold \
            and self.kappa[i].item() < 1e6:        # skip never-stick coords
                v_init = float(velocities[0, i].item())
                rate_i = self.kappa[i].item() * abs(v_init)
                if rate_i < 1e-14:
                    continue
                positions[0, i] = 0.0
                velocities[0, i] = 0.0
                self.frozen_mask[i] = True
                self.frozen_velocity[i] = v_init
                self.thaw_deadline[i] = float(np.random.exponential(1.0 / rate_i))
    # @torch.no_grad()
    # def _apply_cold_start(self, positions: Tensor, velocities: Tensor):
    #     """
    #     Freeze every coordinate with kappa < 1e5 at initialisation.
    #     Each frozen coord gets a thaw deadline drawn from
    #     Exp(kappa_i * |v_i^(0)|) where v_i^(0) is the initial +/- 1
    #     velocity that was just sampled.
    #     """
    #     if not self.cold_start_threshold:
    #         return

    #     for i in range(self.D):
    #         if self.kappa[i].item() < 1e5:
    #             v_init = float(velocities[0, i].item())
    #             rate_i = self.kappa[i].item() * abs(v_init)
    #             if rate_i < 1e-14:
    #                 continue
    #             positions[0, i] = 0.0
    #             velocities[0, i] = 0.0
    #             self.frozen_mask[i] = True
    #             self.frozen_velocity[i] = v_init
    #             self.thaw_deadline[i] = float(np.random.exponential(1.0 / rate_i))

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
        Run the Sticky Zig-Zag sampler for N skeleton points.

        Returns dict with keys:
            "positions", "velocities", "times", "diagnostics",
            "gradient_evals", "frozen_mask_final"
        """
        # --- Storage ---
        positions = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        velocities = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        times = torch.zeros(N, dtype=self.dtype, device=self.device)

        # --- Initialise ---
        self._reset_sticky_state()

        if x0 is None:
            positions[0] = torch.randn(self.D, dtype=self.dtype, device=self.device)
        else:
            positions[0] = x0.to(dtype=self.dtype, device=self.device)
        velocities[0] = self._initial_velocity()
        times[0] = 0.0

        # Optional cold start
        self._apply_cold_start(positions, velocities)

        time_passed = 0.0
        current_time = 0.0
        grad_evals = 0
        diag_log = []

        # Local working copy of t_max so adaptation doesn't mutate self.t_max
        t_max = self.t_max

        pbar = tqdm(total=N, desc=f"Sticky ZigZag[{self.thinning}]", unit="skel")
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

            horizon = min(t_max, dt_hit, dt_thaw)

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
                        if self.adapt_t_max:
                            t_max *= 1.01
                else:  # PLI: returned event is already accepted
                    row["event_type"] = "bounce"
                    row["accepted"] = True
                    event_accepted = True

            # ---- Materialise accepted bounce ON graph ----
            if event_accepted:
                current_time += tau

                # Re-evaluate trajectory ON graph at the accepted time
                pos_prop, vel_prop = self.trajectory_sticky(
                    time_passed + tau, x_prev, v_prev
                )

                # Pick which active coord to flip from the categorical
                # over per-coord rates at the event time. Off graph.
                pos_np = pos_prop.detach().cpu().numpy()
                vel_np = vel_prop.detach().cpu().numpy()
                rates_i = self._per_coord_rates_numpy_sticky(0.0, pos_np, vel_np)
                total = rates_i.sum()
                if total <= 1e-14:
                    # Degenerate fallback: uniform pick over active coords
                    active_idx = np.flatnonzero(
                        (~self.frozen_mask).cpu().numpy()
                    )
                    i_flip = int(np.random.choice(active_idx))
                else:
                    probs = rates_i / total
                    i_flip = int(np.argmax(np.random.multinomial(1, probs)))

                vel_flipped = self.flip_velocity(vel_prop, i_flip)
                grad_evals += 1
                row["rate_evals"] += 1
                row["flipped_coord"] = i_flip

                positions[n] = pos_prop.detach()
                velocities[n] = vel_flipped.detach()
                times[n] = times[n - 1] + time_passed + tau

                iteration += 1
                time_passed = 0.0
                pbar.update(1)

                if self.adapt_t_max:
                    t_max *= 0.96

            # ---- Horizon fired: freeze, thaw, or no_event ----
            elif tau >= horizon:
                time_passed += horizon
                current_time += horizon

                tol = 1e-12

                # --- Freeze event ---
                if abs(horizon - dt_hit) < tol and i_hit is not None:
                    row["event_type"] = "freeze"
                    row["frozen_coord"] = i_hit
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
                    #row["thawed_coord"] = i_thaw
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
                    # No structural event — t_max ran out without bounce
                    if row["event_type"] is None:
                        row["event_type"] = "no_event"
                    if self.adapt_t_max:
                        t_max *= 1.01

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
        for etype in ["bounce", "freeze", "thaw", "no_event"]:
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
