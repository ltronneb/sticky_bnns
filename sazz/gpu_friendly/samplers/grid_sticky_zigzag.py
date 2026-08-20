"""
Sticky ZigZag sampler using the grid-based upper bound (Andral & Kamatani
2024) for its bounce Poisson process. Subclasses GridZigZagSampler --
freeze/thaw are deterministic scheduling events (closed-form hitting time,
pre-drawn thaw deadline), never Poisson-thinned, so grid_bound.py needs no
change: only the bounce rate is thinned, exactly as in the base class, just
evaluated on a trajectory with frozen coordinates pinned to zero (see
GridStickyBoomerangSampler's module docstring for the same argument applied
to Boomerang).

Genuinely simpler than sticky Boomerang in two ways:
  - No reference measure, no velocity refresh clock -- ZigZag has neither,
    so the horizon here is min(grid_t_max, dt_hit, dt_thaw), three
    candidates, not GridStickyBoomerangSampler's four.
  - The hitting-time equation is linear (x_i + t*v_i = 0 => t_i = -x_i/v_i),
    not the periodic a*cos(t)+b*sin(t)=c Boomerang solves -- no atan2/acos,
    no 2*pi wraparound, no zero-root "bump" (a just-thawed coordinate sits
    at x_i=0 exactly and is simply excluded by the "already at zero" guard).

Load-bearing invariant, stated once here and relied on throughout this file:
EVERY SKELETON POINT CARRIES EXACTLY (x_i, v_i) = (0.0, 0.0) ON EVERY
COORDINATE FROZEN OVER THAT INTERVAL. This is what makes the curvature term
of the rate self-mask through the closure-captured anchor velocity (v_i=0
=> g_j(t)=grad_j U(x_t)*v_j=0 automatically, no post-hoc row-masking of
y_full/d_full needed) -- but it does NOT make the gamma floor self-mask
(gamma is an unconditional additive constant), which is why _rate_scalar_
sticky/_per_coord_rates_sticky explicitly multiply by (~frozen_mask), and
why the offset fed to build_grid_bound_vectorized/grid_thinning must be the
DYNAMIC n_active*gamma, not the base class's fixed D*gamma. The invariant
is upheld by hand-zeroing the newly-frozen coordinate's position/velocity
immediately after a freeze event in sample() (see that method) -- dropping
this step would silently corrupt resample_zigzag_path_sticky_torch's
frozen-interval detection (which reads exactly this left-endpoint
invariant), not just leave a harmless ~1e-17 residue.
"""

import math
import time as _time
from functools import partial
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .grid_zigzag import GridZigZagSampler
# from ..utils.grid_bound import grid_thinning, build_grid_bound_vectorized
from ..utils.fast_grid_bound import grid_thinning, build_grid_bound_vectorized


class GridStickyZigZagSampler(GridZigZagSampler):
    """
    Grid-bound ZigZag with freeze/thaw sparsity (see
    sazz.samplers.StickyAutomaticZigZagSampler for the Brent/PLI original
    this ports, and GridStickyBoomerangSampler for the analogous grid-bound
    sticky Boomerang).

    kappa: thaw-rate multiplier, scalar or per-coordinate Tensor[D]. Larger
    kappa = faster thawing = less sticky. kappa_i = 0 means coordinate i is
    permanently frozen once it freezes (see _grid_bound's terminal-state
    guard -- if every coordinate ends up with kappa_i=0, the sampler raises
    rather than hanging or advancing time to infinity).
    can_freeze: bool Tensor[D], coordinates eligible to freeze (default: all).
    cold_start_threshold: if a float, coordinates with |x0_i| below it (and
    finite thaw rate) are frozen at init with a synthetic thaw deadline --
    matches CPU StickyAutomaticZigZagSampler's convention of thresholding
    against the initial position directly (ZigZag has no x_ref to threshold
    against, unlike sticky Boomerang). If a bool Tensor[D], used directly as
    the freeze-at-init mask instead of thresholding |x0_i| -- lets a caller
    freeze exactly the coordinates it already pruned from x0 (e.g. per-layer,
    prior-std-relative), rather than one uniform absolute threshold applied
    indiscriminately across layers of very different scale.
    (all other parameters inherited from GridZigZagSampler)
    """

    def __init__(
        self,
        grad_target,
        D: int,
        kappa: float | Tensor = 1.0,
        can_freeze: Optional[Tensor] = None,
        cold_start_threshold: Optional[float | Tensor] = None,
        gamma: float = 0.01,
        grid_t_max_init: float = 0.1,
        n_segments: int = 20,
        grid_spacing: float = 0.01,
        alpha_plus: float = 1.01,
        alpha_minus: float = 1.04,
        alpha_violation: float = 2.0,
        strategy: str = "vectorized_signed",
        chunk_size: Optional[int] = None,
        grid_kwargs: Optional[dict] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ):
        super().__init__(
            grad_target=grad_target,
            D=D,
            gamma=gamma,
            grid_t_max_init=grid_t_max_init,
            n_segments=n_segments,
            grid_spacing=grid_spacing,
            alpha_plus=alpha_plus,
            alpha_minus=alpha_minus,
            alpha_violation=alpha_violation,
            strategy=strategy,
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
        ZigZag trajectory with frozen coordinates zeroed via torch.where
        (NOT boolean-mask in-place assignment: x_t[mask] = 0.0 is an
        index_put_ on a tensor that may be simultaneously batched (vmap)
        and dual (jvp) when this is called from the rate closures below,
        and functorch handles in-place mutation under those transforms
        inconsistently -- torch.where is a pure functional op, produces
        the identical result, and is unconditionally safe under both (same
        reasoning as GridStickyBoomerangSampler.trajectory_sticky).
        """
        x_t, v_t = self.trajectory(t, x, v)
        zeros = torch.zeros_like(x_t)
        x_t = torch.where(self.frozen_mask, zeros, x_t)
        v_t = torch.where(self.frozen_mask, zeros, v_t)
        return x_t, v_t

    # ------------------------------------------------------------------
    # Freeze / thaw bookkeeping — OFF graph, ported near-verbatim from
    # GridStickyBoomerangSampler (generic sticky-state bookkeeping, no
    # PDMP-specific content).
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
    # over coordinates, matching GridStickyBoomerangSampler's convention,
    # not the CPU StickyAutomaticZigZagSampler's looped version). Linear
    # root only -- no feasibility gate beyond the eps guards below, no
    # zero-root bump (see module docstring).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _next_hitting_event(self, x: Tensor, v: Tensor, eps: float = 1e-14):
        """
        Earliest time an active, freezable coordinate's linear trajectory
        x_i + t*v_i hits x_i = 0. t_i = -x_i / v_i, kept only if t_i > 0
        (moving toward zero) and the coordinate is eligible.

        Returns (dt_hit, i_hit) as plain Python float/int|None -- returns
        (math.inf, None) explicitly when nothing is eligible (do not rely
        on torch.min's arbitrary index-0 tie-break on an all-inf input).
        """
        active = ~self.frozen_mask & self.can_freeze

        safe_v = torch.where(v.abs() < eps, torch.ones_like(v), v)
        t_i = -x / safe_v

        degenerate = (v.abs() < eps) | (x.abs() < eps)
        t_i = torch.where(degenerate, torch.full_like(t_i, math.inf), t_i)
        t_i = torch.where(t_i > 0.0, t_i, torch.full_like(t_i, math.inf))
        t_i = torch.where(active, t_i, torch.full_like(t_i, math.inf))

        dt_hit, i_hit = torch.min(t_i, dim=0)
        dt_hit = float(dt_hit)
        if not math.isfinite(dt_hit):
            return math.inf, None
        return dt_hit, int(i_hit)

    # ------------------------------------------------------------------
    # Cold start — vectorized: freeze all coordinates within threshold of
    # zero (and freezable, and with a nonzero thaw rate) in one shot.
    # frozen_velocity MUST be stored before positions/velocities are
    # zeroed -- see module docstring / plan for the ZigZag-specific
    # silent-failure mode if this order is reversed (a coordinate that
    # thaws to v=0 contributes zero rate forever and can never re-freeze,
    # since _next_hitting_event's |v_i|<eps guard excludes it).
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

        deadlines = torch.distributions.Exponential(rate[idx]).sample()
        self.thaw_deadline[idx] = deadlines

    # ------------------------------------------------------------------
    # Rate closures for the grid bound — sticky-aware overrides. gamma is
    # masked to active coordinates only (per_coord * (~frozen_mask)),
    # matching StickyAutomaticZigZagSampler._rate_numpy_sticky exactly --
    # the curvature term would already be zero on frozen coordinates via
    # trajectory_sticky's v_t=0, but gamma is an unconditional additive
    # constant and does NOT self-mask, so it needs this explicit step.
    # ------------------------------------------------------------------
    def _rate_scalar_sticky(self, t: float, x: Tensor, v: Tensor, want_per_coord: bool = False):
        with torch.no_grad():
            t_t = torch.as_tensor(t, dtype=self.dtype, device=self.device)
            x_t, v_t = self.trajectory_sticky(t_t, x, v)
            grad = self.grad_target(x_t)
            per_coord = torch.clamp(v_t * grad, min=0.0) + self.gamma
            per_coord = per_coord * (~self.frozen_mask).to(self.dtype)
            rate = float(per_coord.sum())
        if want_per_coord:
            return rate, per_coord
        return rate

    def _per_coord_rates_sticky(self, t: float, x: Tensor, v: Tensor) -> Tensor:
        with torch.no_grad():
            t_t = torch.as_tensor(t, dtype=self.dtype, device=self.device)
            x_t, v_t = self.trajectory_sticky(t_t, x, v)
            grad = self.grad_target(x_t)
            per_coord = torch.clamp(v_t * grad, min=0.0) + self.gamma
            per_coord = per_coord * (~self.frozen_mask).to(self.dtype)
            return per_coord

    def _make_rate_and_grad_fn_sticky(self, x: Tensor, v: Tensor):
        """
        Built on trajectory_sticky instead of trajectory. No explicit
        post-hoc row-masking of y_full/d_full for frozen coordinates: the
        closure's anchor velocity v already has v_i=0 on frozen i (per the
        module-level invariant), which alone forces y_full=(g*v).T and
        d_full=(dgdt*v).T to be exactly zero on those rows -- verified
        directly in this file's test suite rather than assumed.
        """
        x = x.detach()
        v = v.detach()

        def gradU_at_t(t: Tensor) -> Tensor:
            x_t, _ = self.trajectory_sticky(t, x, v)
            return self.grad_target(x_t)

        def rate_and_grad_fn(t_batch: Tensor):
            vmap_fn = torch.func.vmap(
                lambda ti: torch.func.jvp(gradU_at_t, (ti,), (torch.ones_like(ti),)),
                chunk_size=self.chunk_size,
            )
            g, dgdt = vmap_fn(t_batch)  # each [K, D]
            y_full = (g * v).transpose(0, 1)      # [D, K]
            d_full = (dgdt * v).transpose(0, 1)   # [D, K]

            if self.strategy == "vectorized_not_signed":
                d_full = torch.where(y_full > 0, d_full, torch.zeros_like(d_full))
                y_full = torch.clamp(y_full, min=0.0)

            return y_full, d_full

        return rate_and_grad_fn

    # ------------------------------------------------------------------
    # Grid bound dispatch — horizon = min(grid_t_max, dt_hit, dt_thaw),
    # three candidates (no dt_refresh -- ZigZag has no refresh clock).
    # ------------------------------------------------------------------
    def _grid_bound(self, pos: Tensor, vel: Tensor, dt_hit: float, dt_thaw: float) -> tuple[float, dict]:
        n_active = int((~self.frozen_mask).sum())

        if n_active == 0:
            # Nothing active: true rate and bound are both identically
            # zero, so grid_thinning's n_segments+1 gradient evaluations
            # would be pure waste to conclude "no event". Route horizon
            # through dt_thaw ALONE (not min(grid_t_max, dt_thaw)) --
            # with nothing to thin, grid_t_max has no meaning here, and
            # keeping it in the min would spin through ~dt_thaw/grid_t_max
            # wasted no-op iterations to reach the thaw.
            #
            # Terminal-state guard: kappa_i=0 makes _freeze set
            # thaw_deadline[i]=inf ("permanently frozen"). If EVERY
            # coordinate ends up permanently frozen, n_active==0 and
            # dt_thaw==inf hold simultaneously and forever -- no event of
            # any kind can ever occur again. Fail loudly rather than let
            # horizon=inf propagate (ceil(inf/grid_spacing) raises on its
            # own anyway, but with a useless traceback; and routing
            # through dt_thaw alone -- necessary for the no-op-spin fix
            # above -- makes this state reachable rather than merely a
            # slow spin, so it needs its own explicit guard).
            if not math.isfinite(dt_thaw):
                raise RuntimeError(
                    "GridStickyZigZagSampler: all coordinates are permanently "
                    "frozen (kappa=0 on every currently-frozen coordinate, "
                    "n_active=0) -- no further freeze, thaw, or bounce event "
                    "is possible from this state."
                )
            horizon = dt_thaw
            n_segments = int(min(max(math.ceil(horizon / self.grid_spacing), 2), self.n_segments))
            stats = {
                "rate_evals": 0,
                "bound_violations": 0,
                "violated": False,
                "rejected_in_window": False,
                "effective_horizon": horizon,
                "accepted": False,
                "horizon": horizon,
                "binding": "dt_thaw",
                "max_ratio": 0.0,
                "curvature_ratio": 0.0,
                "n_segments": n_segments,
                "effective_spacing": horizon / max(n_segments, 1),
            }
            # No Algorithm-4 adaptation while n_active==0 -- nothing to
            # adapt grid_t_max against; it holds its last value until a
            # thaw brings a coordinate back active.
            return math.inf, stats

        candidates = {
            "grid_t_max": self._grid_t_max,
            "dt_hit": dt_hit,
            "dt_thaw": dt_thaw,
        }
        binding = min(candidates, key=candidates.get)
        horizon = candidates[binding]

        n_segments = int(min(max(math.ceil(horizon / self.grid_spacing), 2), self.n_segments))
        effective_spacing = horizon / n_segments

        # This assertion pins the invariant that no freeze or thaw occurs
        # STRICTLY INSIDE the window being thinned -- true by construction
        # only because dt_hit/dt_thaw are themselves horizon candidates
        # above. It's what makes closure-captured frozen_mask/offset valid
        # to hold fixed for the whole grid_thinning call below, including
        # any internal Section 4.7 rebuild.
        assert horizon <= min(dt_hit, dt_thaw) + 1e-9, (
            f"horizon={horizon} exceeds min(dt_hit={dt_hit}, dt_thaw={dt_thaw}) -- "
            "a freeze/thaw could occur inside the thinning window, invalidating "
            "the fixed frozen_mask/offset assumption."
        )

        offset = n_active * self.gamma

        rate_and_grad_fn = self._make_rate_and_grad_fn_sticky(pos, vel)
        rate_scalar_fn = partial(self._rate_scalar_sticky, x=pos.detach(), v=vel.detach())
        bound_fn = partial(
            build_grid_bound_vectorized,
            signed=(self.strategy == "vectorized_signed"),
            offset=offset,
        )

        tau, stats = grid_thinning(
            rate_and_grad_fn, rate_scalar_fn, horizon,
            n_segments=n_segments, device=self.device, dtype=self.dtype,
            diagnostics=True, bound_fn=bound_fn, rate_offset=offset,
            **self.grid_kwargs,
        )
        stats["binding"] = binding
        stats["n_segments"] = n_segments
        stats["effective_spacing"] = effective_spacing

        # Capture the horizon THIS call actually used to draw tau BEFORE
        # the Algorithm-4 adaptation below mutates self._grid_t_max in
        # place -- same mutation-ordering contract fixed in grid_boomerang.py,
        # grid_sticky_boomerang.py, and grid_zigzag.py's _grid_bound.
        stats["horizon"] = horizon

        # --- t_max adaptation (Algorithm 4), extended to 3 candidates ---
        if stats["violated"]:
            self._grid_t_max /= self.alpha_violation
        elif stats["rejected_in_window"]:
            self._grid_t_max /= self.alpha_minus
        elif tau == math.inf and binding == "grid_t_max":
            eff = stats["effective_horizon"]
            if eff is not None and eff >= horizon - 1e-12:
                self._grid_t_max *= self.alpha_plus

        return tau, stats

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------
    def sample(self, N: int, x0: Optional[Tensor] = None, diagnostics: bool = True) -> dict:
        positions = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        velocities = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        times = torch.zeros(N, dtype=self.dtype, device=self.device)

        self._reset_sticky_state()

        if x0 is None:
            positions[0] = torch.randn(self.D, dtype=self.dtype, device=self.device)
        else:
            positions[0] = x0.to(dtype=self.dtype, device=self.device)
        velocities[0] = self._initial_velocity()
        times[0] = 0.0

        self._apply_cold_start(positions, velocities)

        time_passed = 0.0
        current_time = 0.0
        grad_evals = 0
        total_bound_violations = 0
        diag_log = []
        grid_t_max_log = []

        pbar = tqdm(total=N, desc="GridStickyZigZag", unit="skel")
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

            tau, stats = self._grid_bound(pos.detach(), vel.detach(), dt_hit, dt_thaw)
            grad_evals += stats.get("rate_evals", 0)
            total_bound_violations += stats.get("bound_violations", 0)
            grid_t_max_log.append(self._grid_t_max)

            n_frozen = int(self.frozen_mask.sum())
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
                "curvature_ratio": stats.get("curvature_ratio", 0.0),
                "n_segments": stats.get("n_segments"),
                "effective_spacing": stats.get("effective_spacing"),
            }

            # Branch on stats["accepted"], NEVER on a re-derived
            # tau < horizon_used -- self._grid_t_max may already be mutated
            # by _grid_bound's Algorithm-4 step by the time we get here
            # (same contract as every sibling sampler in this package).
            event_accepted = bool(stats["accepted"])

            if event_accepted:
                row["event_type"] = "bounce"
                row["accepted"] = True
            else:
                # No bounce: freeze, thaw, or plain no-event, keyed off
                # effective_horizon (NOT horizon_used) -- a violation-driven
                # early exit means only [0, effective_horizon] was validly
                # examined, so a freeze/thaw check must never fire based on
                # the raw horizon in that case.
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
                    # current_time has NOT yet been incremented below at
                    # this point in the loop, so the true event time is
                    # current_time + advance -- _next_thaw_event computes
                    # dt_thaw = thaw_deadline - current_time, so the
                    # deadline _freeze draws here must be anchored at that
                    # true (post-advance) time, not the stale current_time.
                    self._freeze(i_hit, vel_now, current_time + advance)
                    pos_now = pos_now.clone()
                    vel_now = vel_now.clone()
                    # Hand-zero the newly-frozen coordinate: frozen_mask[i_hit]
                    # was still False when trajectory_sticky ran above, so
                    # it did NOT zero this coordinate -- x_i lands near (but
                    # not exactly) zero from crossing arithmetic, v_i is
                    # still +-1. This upholds the module-level invariant
                    # that every stored skeleton point has (0,0) on frozen
                    # coordinates; skipping it silently corrupts
                    # resample_zigzag_path_sticky_torch's frozen-interval
                    # detection downstream.
                    pos_now[i_hit] = 0.0
                    vel_now[i_hit] = 0.0

                    time_passed += advance
                    current_time += advance

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

                    positions[n] = pos_now
                    velocities[n] = vel_now
                    times[n] = times[n - 1] + time_passed

                    iteration += 1
                    time_passed = 0.0
                    pbar.update(1)

                else:
                    row["event_type"] = "no_event"
                    row["accepted"] = False
                    time_passed += advance
                    current_time += advance

            if event_accepted:
                current_time += tau

                pos_prop, vel_prop = self.trajectory_sticky(
                    torch.as_tensor(time_passed + tau, dtype=self.dtype, device=self.device),
                    x_prev, v_prev,
                )

                pos_np = pos_prop.detach()
                vel_np = vel_prop.detach()
                rates_i = self._per_coord_rates_sticky(0.0, pos_np, vel_np)
                total = rates_i.sum()
                probs = rates_i / total
                i_flip = int(torch.multinomial(probs, 1).item())

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

            n_frozen_now = int(self.frozen_mask.sum())
            row["wall_seconds"] = _time.perf_counter() - _t0
            pbar.set_postfix_str(
                f"t={current_time:.3f} t_max={self._grid_t_max:.4f} "
                f"sparsity={n_frozen_now / self.D:.2f} viol={total_bound_violations}",
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
            "frozen_mask_final": self.frozen_mask.clone(),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _print_diagnostics(diag_log, N, grad_evals, final_time, total_bound_violations):
        import pandas as pd

        df = pd.DataFrame(diag_log)
        n_accept = len(df[df["event_type"] == "bounce"])

        print("\n=== GridStickyZigZag Diagnostics ===")
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
                f"validity signal for the run, not a tunable knob. Consider a finer "
                f"grid_spacing. ***\n"
            )
        else:
            print("Bound violations      : 0")

        if "curvature_ratio" in df.columns:
            print(f"Mean curvature ratio  : {df['curvature_ratio'].mean():.4f} "
                  f"(isolates the target-curvature part of the bound from the "
                  f"n_active*gamma floor)")

        if "flipped_coord" in df.columns:
            flips = df["flipped_coord"].dropna()
            if len(flips) > 0:
                n_unique = flips.nunique()
                print(f"Flipped-coordinate coverage: {n_unique} distinct coordinates flipped "
                      f"out of {n_accept} accepted bounces")

        print("\n=== Event breakdown ===")
        for etype in ["bounce", "freeze", "thaw", "no_event"]:
            sub = df[df["event_type"] == etype]
            if len(sub) > 0:
                print(f"  {etype:10s}: {len(sub):5d} events, mean horizon={sub['horizon'].mean():.4f}")

        if "binding" in df.columns:
            print("\n=== Binding term (which candidate set the horizon) ===")
            counts = df["binding"].value_counts()
            for name, count in counts.items():
                if name is not None:
                    print(f"  {name:12s}: {count:5d}")

        if "effective_spacing" in df.columns:
            print(f"\nMean effective grid spacing: {df['effective_spacing'].mean():.4f}")

        wall = df["wall_seconds"].dropna()
        if len(wall) > 0:
            print(f"\nMean wall-sec / iter : {wall.mean():.6f}")
            print(f"Total wall-sec       : {wall.sum():.2f}")

        print(f"\nSimulation time reached: {float(final_time):.4f}")
