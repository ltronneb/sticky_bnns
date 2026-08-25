"""
Sticky Boomerang sampler using the grid-based upper bound (Andral & Kamatani
2024) for its bounce Poisson process. Subclasses GridBoomerangSampler --
freeze/thaw are deterministic scheduling events (closed-form hitting time,
pre-drawn thaw deadline), never Poisson-thinned, so `grid_bound.py` needs no
change: only the bounce rate is thinned, exactly as in the base class, just
evaluated on a trajectory with frozen coordinates pinned to zero.

n_segments is the CAP on a per-call segment count derived from grid_spacing,
same as in the base class (see _grid_bound there) -- this class only adds
extra horizon candidates (dt_hit, dt_thaw) to the min() that produces the
horizon the formula is applied to. This is necessary here because each
active coordinate's trajectory crosses zero at most twice per 2*pi, so with
D_active active coordinates the hitting-time horizon dt_hit ~ 2*pi/D_active
shrinks by orders of magnitude as sparsity increases -- a fixed segment
count would be either wildly over-resolved early (when D_active is large,
dt_hit tiny) or too coarse late (when D_active is small, dt_hit large).
grid_spacing is anchored to the rate's first-harmonic curvature scale
(pi/4), not to grid_t_max_init/n_segments, precisely to avoid re-coupling
this knob to parameters that have nothing to do with the rate's curvature.
"""

import math
import time as _time
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .grid_boomerang import GridBoomerangSampler
# from ..utils.grid_bound import grid_thinning
from ..utils.fast_grid_bound import grid_thinning


class GridStickyBoomerangSampler(GridBoomerangSampler):
    """
    Grid-bound Boomerang with freeze/thaw sparsity (see
    sazz.samplers.StickyAutomaticBoomerangSampler for the Brent/PLI
    original this ports).

    kappa: thaw-rate multiplier, scalar or per-coordinate Tensor[D].
    can_freeze: bool Tensor[D], coordinates eligible to freeze (default: all).
    cold_start_threshold: if a float, coordinates with |x0_i| below it are
        frozen at init with a synthetic thaw deadline. If a bool Tensor[D],
        used directly as the freeze-at-init mask instead of thresholding
        |x0_i| -- lets a caller freeze exactly the coordinates it already
        pruned (e.g. per-layer, prior-std-relative), rather than the
        uniform absolute threshold the float form applies indiscriminately
        across layers of very different scale.
    grid_spacing (inherited): target grid-node spacing (NOT segment count --
        see module docstring), anchored to the rate's first-harmonic scale,
        independent of grid_t_max_init/n_segments. Default pi/16 (a 4x
        margin on the Boomerang's delta=pi/4 first harmonic).
    n_segments (inherited): the CAP on the per-call segment count computed
        from grid_spacing, not a fixed count -- same as the base class,
        just applied to a horizon that additionally accounts for dt_hit/
        dt_thaw (see this class's _grid_bound override).
    """

    def __init__(
        self,
        grad_target,
        D: int,
        kappa: float | Tensor = 1.0,
        can_freeze: Optional[Tensor] = None,
        cold_start_threshold: Optional[float | Tensor] = None,
        grid_spacing: float = math.pi / 16,
        refresh_rate: float = 0.1,
        grid_t_max_init: float = math.pi / 4,
        n_segments: int = 20,
        alpha_plus: float = 1.01,
        alpha_minus: float = 1.04,
        alpha_violation: float = 2.0,
        chunk_size: Optional[int] = None,
        grid_kwargs: Optional[dict] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ):
        super().__init__(
            grad_target=grad_target,
            D=D,
            refresh_rate=refresh_rate,
            grid_t_max_init=grid_t_max_init,
            n_segments=n_segments,
            grid_spacing=grid_spacing,
            alpha_plus=alpha_plus,
            alpha_minus=alpha_minus,
            alpha_violation=alpha_violation,
            chunk_size=chunk_size,
            grid_kwargs=grid_kwargs,
            dtype=dtype,
            device=device,
        )

        if isinstance(kappa, (int, float)):
            self.kappa = torch.full((D,), float(kappa), dtype=dtype, device=self.device)
        else:
            self.kappa = torch.as_tensor(kappa, dtype=dtype, device=self.device)

        if can_freeze is None:
            self.can_freeze = torch.ones(D, dtype=torch.bool, device=self.device)
        else:
            self.can_freeze = torch.as_tensor(can_freeze, dtype=torch.bool, device=self.device)

        if isinstance(cold_start_threshold, Tensor):
            self.cold_start_threshold = cold_start_threshold.to(dtype=torch.bool, device=self.device)
        else:
            self.cold_start_threshold = cold_start_threshold

        # Mutable sticky state (reset by _reset_sticky_state at sample() start)
        self.frozen_mask = torch.zeros(D, dtype=torch.bool, device=self.device)
        self.frozen_velocity = torch.zeros(D, dtype=dtype, device=self.device)
        self.thaw_deadline = torch.full((D,), float("inf"), dtype=dtype, device=self.device)

    # ------------------------------------------------------------------
    # Sticky dynamics — trajectory with frozen coords pinned to zero
    # ------------------------------------------------------------------
    def trajectory_sticky(self, t: Tensor, x: Tensor, v: Tensor):
        """
        Boomerang trajectory with frozen coordinates zeroed via torch.where
        (NOT boolean-mask in-place assignment: x_t[mask] = 0.0 is an
        index_put_ on a tensor that may be simultaneously batched (vmap)
        and dual (jvp) when this is called from the rate closures below,
        and functorch handles in-place mutation under those transforms
        inconsistently -- torch.where is a pure functional op, produces
        the identical result, and is unconditionally safe under both.
        """
        x_t, v_t = self.trajectory(t, x, v)
        zeros = torch.zeros_like(x_t)
        x_t = torch.where(self.frozen_mask, zeros, x_t)
        v_t = torch.where(self.frozen_mask, zeros, v_t)
        return x_t, v_t

    # ------------------------------------------------------------------
    # Freeze / thaw bookkeeping — OFF graph, scalar-per-event (not a GPU
    # bottleneck: fires once per freeze/thaw, not once per grid-bound call).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _freeze(self, i: int, v: Tensor, current_time: float):
        self.frozen_mask[i] = True
        v_i = float(v[i])
        self.frozen_velocity[i] = v_i
        rate_i = float(self.kappa[i]) * abs(v_i)
        if rate_i > 1e-14:
            self.thaw_deadline[i] = current_time + float(np.random.exponential(1.0 / rate_i))
        else:
            self.thaw_deadline[i] = float("inf")

    @torch.no_grad()
    def _thaw(self, i: int):
        self.frozen_mask[i] = False
        self.thaw_deadline[i] = float("inf")

    @torch.no_grad()
    def _reset_sticky_state(self):
        self.frozen_mask.zero_()
        self.frozen_velocity.zero_()
        self.thaw_deadline.fill_(float("inf"))

    @torch.no_grad()
    def _next_thaw_event(self, current_time: float):
        """Return (dt_thaw, i_thaw): time until (and index of) the next thaw."""
        frozen_idx = torch.where(self.frozen_mask)[0]
        if len(frozen_idx) == 0:
            return float("inf"), None
        deadlines = self.thaw_deadline[frozen_idx]
        local_min = torch.argmin(deadlines)
        i_thaw = int(frozen_idx[local_min])
        dt_thaw = max(float(self.thaw_deadline[i_thaw]) - current_time, 0.0)
        return dt_thaw, i_thaw

    # ------------------------------------------------------------------
    # Hitting-time detection — OFF graph, fully vectorized (no Python loop
    # over coordinates). See module docstring and grid_sticky_boomerang_plan.md
    # decision #3 for why the feasibility gate and zero-root bump are both
    # required, not just the acos clamp.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _next_hitting_event(self, x: Tensor, v: Tensor, eps: float = 1e-14):
        """
        Earliest time an active, freezable coordinate's trajectory hits
        x_i = 0. Solves a*cos(t) + b*sin(t) = c per coordinate, where
        a = x_i - x_ref_i, b = v_i, c = -x_ref_i, vectorized over all D.

        Returns (dt_hit, i_hit) as plain Python float/int|None.
        """
        active = ~self.frozen_mask & self.can_freeze

        a = x - self.x_ref
        b = v
        c = -self.x_ref

        R = torch.hypot(a, b)
        # Feasibility is a SEPARATE gate from the acos clamp -- an infeasible
        # coordinate (no real root) must be excluded outright, not turned
        # into a spurious root at delta in {0, pi} by clamping alone.
        feasible = (R >= eps) & (c.abs() <= R + eps)

        phi = torch.atan2(b, a)
        delta = torch.acos(torch.clamp(c / R.clamp_min(eps), -1.0, 1.0))

        two_pi = 2.0 * math.pi
        cand1 = torch.remainder(phi - delta, two_pi)
        cand2 = torch.remainder(phi + delta, two_pi)

        # Zero-root bump: a just-thawed coordinate sits at exactly x_i=0, so
        # t=0 is itself a root of its own hitting equation. Without bumping
        # a near-zero candidate to the next period, dt_hit would come back
        # ~0 on the very next iteration, forcing horizon->0 and immediately
        # re-freezing the coordinate that was just thawed.
        bump_eps = 1e-10
        cand1 = torch.where(cand1 < bump_eps, cand1 + two_pi, cand1)
        cand2 = torch.where(cand2 < bump_eps, cand2 + two_pi, cand2)

        t_i = torch.minimum(cand1, cand2)

        valid = active & feasible
        t_i = torch.where(valid, t_i, torch.full_like(t_i, math.inf))

        dt_hit, i_hit = torch.min(t_i, dim=0)
        dt_hit = float(dt_hit)
        if not math.isfinite(dt_hit):
            return math.inf, None
        return dt_hit, int(i_hit)

    # ------------------------------------------------------------------
    # Cold start — vectorized: freeze all coordinates within threshold of
    # zero (and freezable, and with a nonzero thaw rate) in one shot.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_cold_start(self, positions: Tensor, velocities: Tensor):
        if self.cold_start_threshold is None:
            return

        x0 = positions[0]
        v0 = velocities[0]

        if isinstance(self.cold_start_threshold, Tensor):
            near_zero = self.cold_start_threshold
        else:
            near_zero = x0.abs() < self.cold_start_threshold
        rate = self.kappa * v0.abs()
        mask = near_zero & self.can_freeze & (rate > 1e-14)

        if not torch.any(mask):
            return

        idx = torch.where(mask)[0]
        self.frozen_velocity[idx] = v0[idx]
        self.frozen_mask[idx] = True
        positions[0, idx] = 0.0
        velocities[0, idx] = 0.0

        # Absolute deadline with no "+ current_time" term -- correct only
        # because cold start always runs at t=0. Contrast with the general
        # case in _refresh_velocity_sticky, which redraws deadlines as
        # current_time + Exp(...) since it can fire at any t.
        deadlines = torch.distributions.Exponential(rate[idx]).sample()
        self.thaw_deadline[idx] = deadlines

    # ------------------------------------------------------------------
    # Active-subspace reflection & refresh — ON/OFF graph respectively,
    # ported near-verbatim (already torch-tensor, masked-gather, not looped).
    # ------------------------------------------------------------------
    def reflect_velocity_sticky(self, v: Tensor, grad: Tensor) -> Tensor:
        if not torch.any(self.frozen_mask):
            return self.reflect_velocity(v, grad)

        active = ~self.frozen_mask
        v_new = v.clone()

        v_a = v[active]
        g_a = grad[active]
        if self.Sigma.ndim == 1:
            Sigma_g = self.Sigma[active] * g_a
        else:
            Sigma_a = self.Sigma[active][:, active]
            Sigma_g = Sigma_a @ g_a

        rate = torch.dot(v_a, g_a)
        denom = torch.dot(g_a, Sigma_g)

        if float(denom) <= 1e-14:
            return v_new

        v_new[active] = v_a - 2.0 * rate / denom * Sigma_g
        v_new[self.frozen_mask] = 0.0
        return v_new

    @torch.no_grad()
    def _refresh_velocity_sticky(self, current_time: float) -> Tensor:
        """
        Eq. (B.1): active coords get a fresh Z_i ~ N(0,1); frozen coords keep
        their sign and get a fresh |Z_i| magnitude (sign(v_i)|Z_i|), so a
        refreshment event can never flip a frozen coordinate's direction of
        travel. The magnitude change also changes the frozen coordinate's
        thaw rate kappa_i*|v_i|, so thaw_deadline must be redrawn from
        current_time at the new rate -- leaving it at the old rate would
        decouple the sojourn (drawn at the old |v|) from the subsequent
        excursion (at the new |v|), biasing mu({0}) (see plan doc). Frozen
        coordinates with kappa_i*|v_new| underflowing 1e-14 go back to inf,
        mirroring _freeze's own threshold.
        """
        if self.Sigma.ndim != 1 and torch.any(self.frozen_mask):
            raise NotImplementedError(
                "Sticky refresh with a full (non-diagonal) Sigma has no well-defined "
                "per-coordinate marginal std to plug into the sign(v_i)|Z_i| rule -- "
                "only diagonal Sigma is supported once coordinates can freeze."
            )

        v = torch.zeros(self.D, dtype=self.dtype, device=self.device)
        active = ~self.frozen_mask
        frozen = self.frozen_mask

        if torch.any(active):
            if self.Sigma.ndim == 1:
                z = torch.randn(int(active.sum()), dtype=self.dtype, device=self.device)
                v[active] = self.Sigma_sqrt[active] * z
            else:
                n_active = int(active.sum())
                Sigma_a = self.Sigma[active][:, active]
                jitter = 1e-12 * torch.eye(n_active, dtype=self.dtype, device=self.device)
                chol_a = torch.linalg.cholesky(Sigma_a + jitter)
                z = torch.randn(n_active, dtype=self.dtype, device=self.device)
                v[active] = chol_a @ z

        if torch.any(frozen):
            n_frozen = int(frozen.sum())
            z = torch.randn(n_frozen, dtype=self.dtype, device=self.device)
            mag = self.Sigma_sqrt[frozen] * z.abs()

            old_v = self.frozen_velocity[frozen]
            sign = torch.sign(old_v)
            zero_sign = sign == 0
            if torch.any(zero_sign):
                rand_sign = torch.randint(0, 2, (int(zero_sign.sum()),), device=self.device).to(self.dtype) * 2 - 1
                sign = torch.where(zero_sign, rand_sign, sign)

            new_v = sign * mag
            self.frozen_velocity[frozen] = new_v

            rate = self.kappa[frozen] * new_v.abs()
            underflow = rate <= 1e-14
            deadlines = torch.where(
                underflow,
                torch.full_like(rate, float("inf")),
                current_time + torch.distributions.Exponential(rate.clamp_min(1e-14)).sample(),
            )
            self.thaw_deadline[frozen] = deadlines

        return v

    # ------------------------------------------------------------------
    # Rate closures for the grid bound — sticky-aware overrides, built on
    # trajectory_sticky instead of trajectory.
    # ------------------------------------------------------------------
    def _rate_scalar_sticky(self, t: float, x: Tensor, v: Tensor) -> float:
        with torch.no_grad():
            t_t = torch.as_tensor(t, dtype=self.dtype, device=self.device)
            x_t, v_t = self.trajectory_sticky(t_t, x, v)
            return float(torch.dot(v_t, self.grad_U_excess(x_t)))

    def _make_rate_and_grad_fn_sticky(self, x: Tensor, v: Tensor):
        x = x.detach()
        v = v.detach()

        def g_scalar(t: Tensor) -> Tensor:
            x_t, v_t = self.trajectory_sticky(t, x, v)
            return torch.dot(v_t, self.grad_U_excess(x_t))

        def rate_and_grad_fn(t_batch: Tensor):
            y, dy = torch.func.vmap(
                lambda ti: torch.func.jvp(g_scalar, (ti,), (torch.ones_like(ti),)),
                chunk_size=self.chunk_size,
            )(t_batch)
            return y, dy

        return rate_and_grad_fn

    # ------------------------------------------------------------------
    # Grid bound dispatch — overrides the base class: horizon additionally
    # min()'d against dt_hit/dt_thaw, and n_segments is derived from
    # grid_spacing each call rather than fixed (see module docstring).
    # ------------------------------------------------------------------
    def _grid_bound(
        self, pos: Tensor, vel: Tensor, dt_refresh: float, dt_hit: float, dt_thaw: float,
    ) -> tuple[float, dict]:
        candidates = {
            "grid_t_max": self._grid_t_max,
            "dt_refresh": dt_refresh,
            "dt_hit": dt_hit,
            "dt_thaw": dt_thaw,
        }
        binding = min(candidates, key=candidates.get)
        horizon = candidates[binding]

        # n_segments computed ONCE from the incoming horizon and held fixed
        # for the lifetime of this grid_thinning call -- including across
        # any internal Section 4.7 shrink/rebuild on a bound violation.
        # Recomputing it per-shrink would hold effective per-segment
        # spacing constant instead of shrinking it, so a violation would
        # never actually resolve (see grid_sticky_boomerang_plan.md #2b).
        n_segments = int(min(max(math.ceil(horizon / self.grid_spacing), 2), self.n_segments))

        rate_and_grad_fn = self._make_rate_and_grad_fn_sticky(pos, vel)
        rate_scalar_fn = lambda t: self._rate_scalar_sticky(t, pos.detach(), vel.detach())

        tau, stats = grid_thinning(
            rate_and_grad_fn, rate_scalar_fn, horizon,
            n_segments=n_segments, device=self.device, dtype=self.dtype,
            diagnostics=True, **self.grid_kwargs,
        )
        stats["binding"] = binding
        stats["n_segments"] = n_segments

        # Capture the horizon THIS call actually used to draw tau BEFORE the
        # Algorithm-4 adaptation below mutates self._grid_t_max in place --
        # same reasoning as the base class's _grid_bound (see its comment).
        stats["horizon"] = horizon

        # --- t_max adaptation (Algorithm 4), extended to 4 candidates ---
        if stats["violated"]:
            self._grid_t_max /= self.alpha_violation
        elif stats["rejected_in_window"]:
            self._grid_t_max /= self.alpha_minus
        elif tau == math.inf and binding == "grid_t_max":
            # Only grow when grid_t_max was the STRICT argmin of all four
            # candidates and the full window was examined with no event --
            # dormant early in a run (dt_hit small, D_active large) and
            # increasingly live as sparsity climbs (dt_hit grows toward
            # grid_t_max/dt_refresh) -- see plan decision #8.
            eff = stats["effective_horizon"]
            if eff is not None and eff >= horizon - 1e-12:
                self._grid_t_max *= self.alpha_plus

        return tau, stats

    # ------------------------------------------------------------------
    # Main sampling loop — overrides the base class: adds freeze/thaw
    # branches to the event dispatch, keyed off effective_horizon (not the
    # raw horizon) so a bound-violation-truncated window can never be
    # mistaken for a fully-examined one when deciding whether a freeze or
    # thaw actually fired (see plan decision #9).
    # ------------------------------------------------------------------
    def sample(self, N: int, x0: Optional[Tensor] = None, diagnostics: bool = True) -> dict:
        assert self.x_ref is not None, "Call preprocess() first."

        positions = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        velocities = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        times = torch.zeros(N, dtype=self.dtype, device=self.device)

        self._reset_sticky_state()

        if x0 is None:
            positions[0] = self.x_ref + self._refresh_velocity()
        else:
            positions[0] = x0.to(dtype=self.dtype, device=self.device)
        velocities[0] = self._refresh_velocity()
        times[0] = 0.0

        self._apply_cold_start(positions, velocities)

        dt_refresh = float(np.random.exponential(1.0 / self.refresh_rate))
        time_passed = 0.0
        current_time = 0.0
        grad_evals = 0
        total_bound_violations = 0
        diag_log = []
        grid_t_max_log = []

        pbar = tqdm(total=N, desc="GridStickyBoomerang", unit="skel")
        pbar.update(1)
        iteration = 1

        while iteration < N:
            _t0 = _time.perf_counter()
            n = iteration

            x_prev = positions[n - 1]
            v_prev = velocities[n - 1]

            with torch.no_grad():
                pos, vel = self.trajectory_sticky(
                    torch.as_tensor(time_passed, dtype=self.dtype, device=self.device),
                    x_prev, v_prev,
                )

            dt_hit, i_hit = self._next_hitting_event(pos, vel)
            dt_thaw, i_thaw = self._next_thaw_event(current_time)

            tau, stats = self._grid_bound(pos.detach(), vel.detach(), dt_refresh, dt_hit, dt_thaw)
            grad_evals += stats.get("rate_evals", 0)
            total_bound_violations += stats.get("bound_violations", 0)
            grid_t_max_log.append(self._grid_t_max)

            n_frozen = int(self.frozen_mask.sum())
            # stats["horizon"] is the horizon _grid_bound actually used to
            # draw tau, captured BEFORE its Algorithm-4 adaptation mutated
            # self._grid_t_max -- a fresh min(self._grid_t_max, dt_refresh,
            # dt_hit, dt_thaw) recomputed here would use the ALREADY-MUTATED
            # self._grid_t_max, which can silently fail `tau < horizon_used`
            # for a genuinely accepted tau and misroute it into the
            # freeze/thaw/no_event branch below (same bug as the base
            # class's sample(), see its comment).
            horizon_used = stats["horizon"]

            row = {
                "rate_evals": stats.get("rate_evals", 0),
                "horizon": horizon_used,
                "time": current_time,
                "event_type": None,
                "accepted": None,
                "n_frozen": n_frozen,
                "n_active": self.D - n_frozen,
                "sparsity": n_frozen / self.D,
                "binding": stats.get("binding"),
                "wall_seconds": None,
                "bound_violations": stats.get("bound_violations", 0),
                "max_ratio": stats.get("max_ratio", 0.0),
            }

            # Branch on stats["accepted"], NEVER on a re-derived
            # tau < horizon_used using the (possibly stale) horizon_used
            # above -- stats["accepted"] is set inside grid_thinning at the
            # moment tau was drawn, before any subsequent mutation.
            event_accepted = bool(stats["accepted"])

            if event_accepted:
                row["event_type"] = "bounce"
                row["accepted"] = True
            else:
                # No bounce accepted: figure out whether this was a freeze,
                # thaw, or plain no-event, using effective_horizon (NOT the
                # raw horizon_used) -- a violation-driven early exit means
                # only [0, effective_horizon] was validly examined, so the
                # freeze/thaw check must never fire based on horizon_used
                # in that case (it would mark a coordinate frozen/thawed at
                # a crossing time the sampler never actually reached).
                eff = stats.get("effective_horizon")
                advance = eff if eff is not None else horizon_used
                tol = 1e-12

                if eff is not None and abs(eff - dt_hit) < tol and i_hit is not None:
                    row["event_type"] = "freeze"
                    with torch.no_grad():
                        pos_now, vel_now = self.trajectory_sticky(
                            torch.as_tensor(time_passed + advance, dtype=self.dtype, device=self.device),
                            x_prev, v_prev,
                        )
                    self._freeze(i_hit, vel_now, current_time + advance)
                    pos_now = pos_now.clone()
                    vel_now = vel_now.clone()
                    pos_now[i_hit] = 0.0
                    vel_now[i_hit] = 0.0

                    time_passed += advance
                    current_time += advance
                    dt_refresh -= advance

                    positions[n] = pos_now
                    velocities[n] = vel_now
                    times[n] = times[n - 1] + time_passed

                    iteration += 1
                    time_passed = 0.0
                    pbar.update(1)

                elif eff is not None and abs(eff - dt_thaw) < tol and i_thaw is not None:
                    row["event_type"] = "thaw"
                    with torch.no_grad():
                        pos_now, vel_now = self.trajectory_sticky(
                            torch.as_tensor(time_passed + advance, dtype=self.dtype, device=self.device),
                            x_prev, v_prev,
                        )
                    vel_now = vel_now.clone()
                    vel_now[i_thaw] = self.frozen_velocity[i_thaw]
                    self._thaw(i_thaw)

                    time_passed += advance
                    current_time += advance
                    dt_refresh -= advance

                    positions[n] = pos_now
                    velocities[n] = vel_now
                    times[n] = times[n - 1] + time_passed

                    iteration += 1
                    time_passed = 0.0
                    pbar.update(1)

                else:
                    row["event_type"] = "no_event"
                    time_passed += advance
                    current_time += advance
                    dt_refresh -= advance

            if event_accepted:
                current_time += tau

                pos_prop, vel_prop = self.trajectory_sticky(
                    torch.as_tensor(time_passed + tau, dtype=self.dtype, device=self.device),
                    x_prev, v_prev,
                )
                grad_at_prop = self.grad_U_excess(pos_prop)
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

            if dt_refresh <= 1e-14:
                row["wall_seconds"] = _time.perf_counter() - _t0
                diag_log.append(row)

                n_frozen_r = int(self.frozen_mask.sum())
                refresh_row = {
                    "rate_evals": 0, "horizon": 0.0, "time": current_time,
                    "event_type": "refresh", "accepted": None,
                    "n_frozen": n_frozen_r, "n_active": self.D - n_frozen_r,
                    "sparsity": n_frozen_r / self.D, "binding": None,
                    "wall_seconds": 0.0, "bound_violations": 0, "max_ratio": 0.0,
                }

                if iteration < N:
                    n = iteration
                    with torch.no_grad():
                        pos_ref, _ = self.trajectory_sticky(
                            torch.as_tensor(time_passed, dtype=self.dtype, device=self.device),
                            positions[n - 1], velocities[n - 1],
                        )
                    positions[n] = pos_ref
                    #assert abs(current_time - float(times[n - 1]) - time_passed) < 1e-5
                    velocities[n] = self._refresh_velocity_sticky(current_time)
                    times[n] = times[n - 1] + time_passed

                    iteration += 1
                    time_passed = 0.0
                    pbar.update(1)

                dt_refresh = float(np.random.exponential(1.0 / self.refresh_rate))
                diag_log.append(refresh_row)
                pbar.set_postfix_str(
                    f"t={current_time:.3f} t_max={self._grid_t_max:.4f} "
                    f"sparsity={n_frozen_r / self.D:.2f} viol={total_bound_violations}",
                    refresh=False,
                )
                continue

            row["wall_seconds"] = _time.perf_counter() - _t0
            pbar.set_postfix_str(
                f"t={current_time:.3f} t_max={self._grid_t_max:.4f} "
                f"sparsity={n_frozen / self.D:.2f} viol={total_bound_violations}",
                refresh=False,
            )
            diag_log.append(row)

        pbar.close()

        if diagnostics:
            self._print_sticky_diagnostics(diag_log, N, grad_evals, times[iteration - 1], total_bound_violations)

        return {
            "positions": positions,
            "velocities": velocities,
            "times": times,
            "diagnostics": diag_log,
            "gradient_evals": grad_evals,
            "bound_violations": total_bound_violations,
            "grid_t_max_log": grid_t_max_log,
            "frozen_mask_final": self.frozen_mask.clone(),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _print_sticky_diagnostics(diag_log, N, grad_evals, final_time, total_bound_violations):
        import pandas as pd

        df = pd.DataFrame(diag_log)
        n_accept = len(df[df["event_type"] == "bounce"])

        print("\n=== GridStickyBoomerang Diagnostics ===")
        print(f"Total gradient evals  : {grad_evals}")
        print(f"Grad evals / skeleton : {grad_evals / max(N, 1):.1f}")
        print(f"Accepted bounces      : {n_accept}")

        if "sparsity" in df.columns:
            print(f"Mean sparsity (frac frozen): {df['sparsity'].mean():.3f}")
            print(f"Max simultaneous frozen    : {int(df['n_frozen'].max())} / {N}")

        if total_bound_violations > 0:
            print(
                f"\n*** BOUND VIOLATIONS: {total_bound_violations} — the grid was too coarse "
                f"for the local curvature at least once. Samples drawn in the affected "
                f"window(s) used a bound that was not actually valid there; this is a "
                f"validity signal for the run, not a tunable knob. Consider a smaller "
                f"grid_spacing. ***\n"
            )
        else:
            print("Bound violations      : 0")

        print("\n=== Event breakdown ===")
        for etype in ["bounce", "freeze", "thaw", "no_event", "refresh"]:
            sub = df[df["event_type"] == etype]
            if len(sub) > 0:
                print(f"  {etype:10s}: {len(sub):5d} events, mean horizon={sub['horizon'].mean():.4f}")

        if "binding" in df.columns:
            print("\n=== Binding term (which candidate set the horizon) ===")
            counts = df["binding"].value_counts()
            for name, count in counts.items():
                if name is not None:
                    print(f"  {name:12s}: {count:5d}")

        wall = df["wall_seconds"].dropna()
        if len(wall) > 0:
            print(f"\nMean wall-sec / iter : {wall.mean():.6f}")
            print(f"Total wall-sec       : {wall.sum():.2f}")

        print(f"\nSimulation time reached: {float(final_time):.4f}")
