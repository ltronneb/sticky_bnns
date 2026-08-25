"""
FastGridStickyZigZagSampler_Cheap -- a duplicate of fast_grid_sticky_zigzag.py
(FastGridStickyZigZagSampler) that ADDITIONALLY supports resuming sample()
across separate calls, so a very large N (e.g. 1M skeleton points) can be
run as several smaller staged calls instead of one call whose transient
chunk_*.pt disk footprint (O(N/chunk_size) files, each O(chunk_size * D))
must all fit on disk simultaneously before the final resample step consumes
them. Between stages, the caller can delete the previous stage's chunk_*.pt
files (NOT diag_*.pt -- those are cheap and still needed for analysis),
recovering disk space that a single N=1,000,000 call could never free
until the very end.

This file exists as a full duplicate, not a parameter added to the
original, specifically so fast_grid_sticky_zigzag.py -- already used by
completed, working runs -- is never touched. See fast_cheap_mnist_cnn.py
for the staging wrapper that drives this.

RESUME CONTRACT: sample() below accepts an optional `resume_state` dict
(the "resume_state" key of a previous call's return dict, from either a
chunked or non-chunked call). When given, sample() does NOT call
_reset_sticky_state(), does NOT draw a fresh x0/initial velocity, and does
NOT call _apply_cold_start() -- all sticky/position/velocity/time state is
restored from resume_state instead, exactly continuing the trajectory
(same frozen_mask, thaw_deadline, frozen_velocity, position, velocity,
elapsed simulation time). This is NOT equivalent to running N_total in one
call unless every stage's resume_state is threaded into the next stage's
call -- an ordinary x0-only call always cold-starts (fresh random velocity,
frozen_mask reset to all-False, time reset to 0), which silently discards
everything the run has learned about which coordinates are frozen/moving.

self._grid_t_max is NOT part of resume_state: it lives on the sampler
INSTANCE (set once in __init__, mutated in place by _grid_bound's Algorithm
4 adaptation), not in sample()'s return dict, so it already persists
correctly across staged calls as long as the SAME sampler object is reused
for every stage (never reconstruct a fresh FastGridStickyZigZagSampler_Cheap
between stages). RNG state (np.random / torch's default generator, used by
_freeze's exponential draws and the initial-velocity draw) is also
process-global and needs no explicit carry-over as long as all stages run
in the same Python process.

---

Original module docstring (fast_grid_sticky_zigzag.py) below, describing
the sticky-ZigZag grid-bound design this file inherits unchanged except for
the resume additions in sample() itself:

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

---

FastGridStickyZigZagSampler (this file) -- a copy of GridStickyZigZagSampler
with two kinds of change, both scoped to this file and fast_grid_bound.py
(see radiant-finding-church.md for the full plan this implements):

1. Chunked skeleton flushing: sample() can bound peak memory to
   O(chunk_size * D) instead of O(N * D) by writing skeleton rows to disk
   in chunks instead of holding the whole [N, D] trajectory in memory.
   Triggered by passing chunk_size/chunk_dir. chunk_size=None (default)
   preserves exactly the original's behavior -- same in-memory [N, D]
   tensors, same return dict shape.

2. Sync/overhead reduction: a handful of GPU-hostile patterns (scalar
   single-index writes in _freeze/_thaw, duplicate .sum() reads of
   frozen_mask, fresh torch.as_tensor(time_passed, ...) allocations,
   multiple independent .item()-style host reads per event) are merged
   or vectorized. All of these are pure refactors -- bit-identical output
   for the same inputs -- EXCEPT where explicitly noted (see the
   n_frozen-staleness note in _grid_bound's docstring... n/a here, that's
   the Boomerang file; ZigZag has no such staleness to preserve).
"""

import math
import os
import time as _time
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .grid_zigzag import GridZigZagSampler
from ..utils.fast_grid_bound import grid_thinning, build_grid_bound_vectorized

# Set True to assert-verify the n_frozen +-1 arithmetic merge in sample()
# against a fresh int(self.frozen_mask.sum()) each iteration -- off by
# default since the assert itself forces the sync it exists to avoid.
_DEBUG_SYNC_MERGE = False


class FastGridStickyZigZagSampler_Cheap(GridZigZagSampler):
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
        resample_grad_batch: Optional[Callable[[], None]] = None,
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

        # Optional zero-arg hook, called once per sample() loop iteration
        # (immediately before _grid_bound) -- lets a caller using a
        # subsampled-gradient grad_target pick a fresh minibatch exactly
        # once per Poisson-thinning proposal. grad_target itself must stay
        # a fixed function of x for the ENTIRE _grid_bound call: it's
        # evaluated both eagerly (_rate_scalar_sticky) and inside
        # torch.func.vmap/jvp over a batch of candidate grid times
        # (_make_rate_and_grad_fn_sticky), and the grid-thinning bound is
        # only valid if every one of those evaluations sees the same rate
        # function. None (default): no-op, exactly today's behavior.
        self._resample_grad_batch = resample_grad_batch

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

        # Reused scalar tensor for torch.as_tensor(time_passed, ...) call
        # sites -- avoids reallocating a fresh 0-dim tensor 2-3x per
        # iteration. Filled via .fill_() at each use site instead.
        self._time_scalar = torch.zeros((), dtype=dtype, device=self.device)

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
    # PDMP-specific content). Runs eagerly under @torch.no_grad(), never
    # inside a vmap/jvp closure (only call sites are in sample()'s plain
    # event-dispatch branches) -- so, unlike trajectory_sticky above,
    # there is no vmap/jvp-safety reason to avoid in-place indexed writes
    # here. The vectorization below is purely a kernel-launch-count
    # reduction: one single-element write (the one-hot mask) instead of
    # three, plus vectorized torch.where updates instead of three more
    # single-element writes.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _freeze(self, i: int, v: Tensor, current_time: float):
        v_i = float(v[i])
        rate_i = float(self.kappa[i]) * abs(v_i)
        deadline_i = current_time + float(np.random.exponential(1.0 / rate_i)) if rate_i > 1e-14 else float("inf")

        idx_mask = torch.zeros(self.D, dtype=torch.bool, device=self.device)
        idx_mask[i] = True

        self.frozen_mask = torch.where(idx_mask, torch.ones_like(self.frozen_mask), self.frozen_mask)
        self.frozen_velocity = torch.where(
            idx_mask, torch.full_like(self.frozen_velocity, v_i), self.frozen_velocity
        )
        self.thaw_deadline = torch.where(
            idx_mask, torch.full_like(self.thaw_deadline, deadline_i), self.thaw_deadline
        )

    @torch.no_grad()
    def _thaw(self, i: int):
        idx_mask = torch.zeros(self.D, dtype=torch.bool, device=self.device)
        idx_mask[i] = True

        self.frozen_mask = torch.where(idx_mask, torch.zeros_like(self.frozen_mask), self.frozen_mask)
        self.thaw_deadline = torch.where(
            idx_mask, torch.full_like(self.thaw_deadline, float("inf")), self.thaw_deadline
        )

    @torch.no_grad()
    def _reset_sticky_state(self):
        self.frozen_mask.zero_()
        self.frozen_velocity.zero_()
        self.thaw_deadline.fill_(float("inf"))

    @torch.no_grad()
    def _next_thaw_event(self, current_time: float):
        """
        Return (dt_thaw, i_thaw): time until (and index of) the next thaw.

        Merges the three independent host reads (len(), int(), float())
        of the original into one combined .tolist()-style read where a
        frozen coordinate exists -- len(frozen_idx) alone doesn't sync
        (Tensor.__len__ on a 1-D tensor's shape is metadata, no device
        read), so only the argmin index and the deadline value need
        merging.
        """
        frozen_idx = torch.where(self.frozen_mask)[0]
        if len(frozen_idx) == 0:
            return float("inf"), None
        deadlines = self.thaw_deadline[frozen_idx]
        local_min = torch.argmin(deadlines)
        # index round-trips through the deadline's dtype (float64/float32)
        # for a single combined host read -- exact for D < 2**24 in
        # float32 (D here is at most ~61706, LeNet5 scale), so safe, but
        # would silently corrupt a much larger index range at lower
        # precision if this pattern is ever reused elsewhere.
        i_thaw_f, deadline_i = torch.stack(
            [frozen_idx[local_min].to(deadlines.dtype), self.thaw_deadline[frozen_idx[local_min]]]
        ).tolist()
        i_thaw = int(i_thaw_f)
        dt_thaw = max(deadline_i - current_time, 0.0)
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

        The float(dt_hit)/int(i_hit) pair from torch.min is merged into
        one combined host read -- the index round-trips through dt_hit's
        dtype for the single .tolist() call, exact for D < 2**24 in
        float32 (see _next_thaw_event's identical note).
        """
        active = ~self.frozen_mask & self.can_freeze

        safe_v = torch.where(v.abs() < eps, torch.ones_like(v), v)
        t_i = -x / safe_v

        degenerate = (v.abs() < eps) | (x.abs() < eps)
        t_i = torch.where(degenerate, torch.full_like(t_i, math.inf), t_i)
        t_i = torch.where(t_i > 0.0, t_i, torch.full_like(t_i, math.inf))
        t_i = torch.where(active, t_i, torch.full_like(t_i, math.inf))

        dt_hit_t, i_hit_t = torch.min(t_i, dim=0)
        dt_hit, i_hit_f = torch.stack([dt_hit_t, i_hit_t.to(dt_hit_t.dtype)]).tolist()
        if not math.isfinite(dt_hit):
            return math.inf, None
        return dt_hit, int(i_hit_f)

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
    #
    # n_active is now an optional parameter: sample() computes n_frozen
    # once per iteration (see the merged-sync note there) and threads
    # D - n_frozen through here instead of letting this method
    # independently re-derive it via its own int((~self.frozen_mask).sum())
    # sync -- both computed the same value from the same frozen_mask, so
    # this is a pure duplicate-read elimination, not a behavior change.
    # ------------------------------------------------------------------
    def _grid_bound(
        self, pos: Tensor, vel: Tensor, dt_hit: float, dt_thaw: float, n_active: Optional[int] = None,
    ) -> tuple[float, dict]:
        if n_active is None:
            n_active = int((~self.frozen_mask).sum())

        if n_active == 0:
            if not math.isfinite(dt_thaw):
                raise RuntimeError(
                    "FastGridStickyZigZagSampler: all coordinates are permanently "
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
    # Chunked-flush bookkeeping — all new, no analog in the original.
    # ------------------------------------------------------------------
    def _flush_chunk(
        self, chunk_dir: Path, chunk_idx: int, positions: Tensor, velocities: Tensor,
        times: Tensor, local_idx: int, global_row_start: int,
    ) -> dict:
        """
        Writes positions[:local_idx]/velocities[:local_idx]/times[:local_idx]
        (always whole rows -- local_idx only ever advances immediately
        after a complete row write, see _write_row) to
        chunk_dir/chunk_{idx:05d}.pt, on CPU (portable; downstream
        resampling/analysis is a separate, later step from sampling, so
        the one-time D2H copy per flush is preferred over pinning
        analysis to a GPU host). Returns this chunk's manifest entry.
        """
        chunk_path = chunk_dir / f"chunk_{chunk_idx:05d}.pt"
        t_start = float(times[0])
        t_end = float(times[local_idx - 1])
        torch.save({
            "positions": positions[:local_idx].detach().cpu(),
            "velocities": velocities[:local_idx].detach().cpu(),
            "times": times[:local_idx].detach().cpu(),
            "chunk_idx": chunk_idx,
            "row_count": local_idx,
            "t_start": t_start,
            "t_end": t_end,
            "global_row_start": global_row_start,
        }, chunk_path)
        return {
            "chunk_idx": chunk_idx,
            "row_count": local_idx,
            "t_start": t_start,
            "t_end": t_end,
            "global_row_start": global_row_start,
            "path": str(chunk_path),
        }

    @staticmethod
    def _write_manifest(chunk_dir: Path, chunk_entries: list[dict], dtype: torch.dtype) -> Path:
        """
        Atomic write: manifest.pt.tmp then os.replace -- a crash mid-write
        must not leave a truncated manifest.pt, since the manifest is the
        sole index into a potentially long-running chunked sample() call's
        output (see plan). Rewritten (not appended) after every flush --
        this is a full manifest, not a delta.
        """
        manifest_path = chunk_dir / "manifest.pt"
        tmp_path = chunk_dir / "manifest.pt.tmp"
        torch.save({"chunks": chunk_entries, "dtype": dtype}, tmp_path)
        os.replace(tmp_path, manifest_path)
        return manifest_path

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------
    def sample(
        self, N: int, x0: Optional[Tensor] = None, diagnostics: bool = True,
        chunk_size: Optional[int] = None, chunk_dir: Optional[Path | str] = None,
        flush_diag_every: Optional[int] = None, resume_state: Optional[dict] = None,
    ) -> dict:
        """
        chunk_size=None (default): identical behavior to the original --
        one full [N, D] in-memory allocation, full diag_log retained,
        return dict has "positions"/"velocities"/"times"/"diagnostics"
        (a list) keys exactly as before.

        chunk_size=<int>: bounds peak memory to O(chunk_size * D). Requires
        chunk_dir. Buffers are sized min(chunk_size, N); a chunk is flushed
        to chunk_dir/chunk_{idx:05d}.pt immediately after any row write
        that fills the buffer (not once per while-loop iteration -- see
        _write_row; ZigZag writes at most 1 row per iteration so this
        distinction doesn't bite here the way it does in the Boomerang
        file's refresh branch, but the same helper is used for
        consistency and because sample()'s carry-over-state handling is
        identical either way). flush_diag_every (default: chunk_size when
        chunking is active) additionally flushes diag_log to
        chunk_dir/diag_{idx:05d}.pt at that cadence and clears it from
        memory -- diag_log grows with LOOP ITERATION count, not N (a
        no_event iteration appends a diag row without writing a skeleton
        row), so it is not bounded by chunk_size alone. Return dict in
        this mode has "chunk_dir"/"chunk_files"/"manifest_path"/
        "diag_summary" keys instead of "positions"/"velocities"/"times";
        "diagnostics" is None unless flush_diag_every was never set (i.e.
        diag_log was never cleared, so returning it is safe).

        resume_state=<dict> (new in this _Cheap file, no analog upstream):
        continues a PREVIOUS sample() call's trajectory instead of cold-
        starting. Must be exactly the "resume_state" dict from a prior
        call's return value (see the end of this method for its exact
        keys). When given: x0/diagnostics-cold-start machinery is bypassed
        entirely -- _reset_sticky_state() is NOT called, frozen_mask/
        frozen_velocity/thaw_deadline are restored via .copy_() from
        resume_state instead of starting from all-unfrozen, position[0]/
        velocity[0]/time[0] are the exact final carry-over state of the
        previous call (not a fresh x0/_initial_velocity() draw), and
        _apply_cold_start() is NOT called (cold-start freezing is a t=0-only
        concept -- re-running it mid-trajectory on the CURRENT position
        would incorrectly re-freeze coordinates using x0's semantics on a
        position that has already moved). `x0` is ignored when resume_state
        is given (both cannot apply at once). self._grid_t_max is NOT part
        of resume_state -- it must already be correct on this sampler
        instance (i.e. the caller reused the SAME instance across stages;
        see this file's module docstring).
        """
        if resume_state is not None and x0 is not None:
            raise ValueError(
                "sample() got both x0 and resume_state -- these are mutually exclusive "
                "(resume_state already carries the continuing position; x0 only makes "
                "sense for a fresh cold start). Pass exactly one."
            )

        chunking_active = chunk_size is not None
        if chunking_active:
            if chunk_dir is None:
                raise ValueError("chunk_dir is required when chunk_size is set.")
            chunk_dir = Path(chunk_dir)
            chunk_dir.mkdir(parents=True, exist_ok=True)
            if flush_diag_every is None:
                flush_diag_every = chunk_size

        buffer_len = min(chunk_size, N) if chunking_active else N
        positions = torch.zeros(buffer_len, self.D, dtype=self.dtype, device=self.device)
        velocities = torch.zeros(buffer_len, self.D, dtype=self.dtype, device=self.device)
        times = torch.zeros(buffer_len, dtype=self.dtype, device=self.device)

        if resume_state is not None:
            self.frozen_mask = resume_state["frozen_mask"].to(dtype=torch.bool, device=self.device).clone()
            self.frozen_velocity = resume_state["frozen_velocity"].to(dtype=self.dtype, device=self.device).clone()
            self.thaw_deadline = resume_state["thaw_deadline"].to(dtype=self.dtype, device=self.device).clone()
            positions[0] = resume_state["position"].to(dtype=self.dtype, device=self.device)
            velocities[0] = resume_state["velocity"].to(dtype=self.dtype, device=self.device)
            times[0] = 0.0
            resume_time_offset = float(resume_state["current_time"])
        else:
            self._reset_sticky_state()

            if x0 is None:
                positions[0] = torch.randn(self.D, dtype=self.dtype, device=self.device)
            else:
                positions[0] = x0.to(dtype=self.dtype, device=self.device)
            velocities[0] = self._initial_velocity()
            times[0] = 0.0

            self._apply_cold_start(positions, velocities)
            resume_time_offset = 0.0

        # Carry-over state -- seeded AFTER _apply_cold_start (which
        # mutates positions[0]/velocities[0] in place to zero frozen
        # coordinates) and stored via .copy_() into preallocated [D]
        # tensors, not as a view into the buffer: positions[n-1] is a
        # view today, and a view becomes stale/wrong the instant that
        # buffer slot is overwritten or the buffer is reset after a
        # flush. Replaces EVERY direct positions[n-1]/velocities[n-1]/
        # times[n-1]-style read in the loop below, not just the ones at
        # the top of the loop.
        x_prev_state = torch.zeros(self.D, dtype=self.dtype, device=self.device)
        v_prev_state = torch.zeros(self.D, dtype=self.dtype, device=self.device)
        x_prev_state.copy_(positions[0])
        v_prev_state.copy_(velocities[0])
        # Boxed so the write_row closure can rebind it. Seeded from
        # resume_time_offset (0.0 on a cold start) so times[] written by
        # this stage continue the previous stage's global simulation clock
        # instead of restarting at 0 -- required for thaw_deadline
        # (absolute times drawn by a prior stage's _freeze/_apply_cold_start)
        # to compare correctly against current_time in THIS stage.
        t_prev_holder = [resume_time_offset]
        times[0] = resume_time_offset

        local_idx = 1  # row 0 already written above
        chunk_idx = 0
        global_row_start = 0
        chunk_entries: list[dict] = []
        manifest_path: Optional[Path] = None

        time_passed = 0.0
        current_time = resume_time_offset
        grad_evals = 0
        total_bound_violations = 0
        diag_log = []
        diag_flushed = False
        grid_t_max_log = []

        # Running diagnostics accumulator -- mirrors exactly what
        # _print_diagnostics/_print_sticky_diagnostics_chunked compute
        # from a full pd.DataFrame(diag_log), but incrementally, so
        # diag_log itself can be safely cleared under chunking without
        # losing the ability to print an equivalent summary. Uses .get()
        # semantics throughout (see _update_diag_accumulator) since not
        # every row dict has every key (e.g. Boomerang's refresh row
        # omits several fields -- ZigZag rows are more uniform but the
        # same accumulator code is shared in spirit with fast_grid_
        # sticky_boomerang.py, so the defensive .get() stays here too).
        diag_acc = _DiagAccumulator()

        pbar = tqdm(total=N, desc="FastGridStickyZigZag", unit="skel")
        pbar.update(1)
        iteration = 1

        def write_row(pos_row: Tensor, vel_row: Tensor, t_row: float, event_type: str) -> None:
            """
            Writes one complete skeleton row into the current buffer,
            updates carry-over state, and flushes (if chunking) the
            instant the buffer fills -- checked after EVERY row write,
            not once per while-loop body, so a hypothetical future
            multi-row-per-iteration branch (none exist in ZigZag today,
            unlike Boomerang's refresh branch) would still flush
            correctly by construction.
            """
            nonlocal local_idx, chunk_idx, global_row_start, iteration, time_passed
            positions[local_idx] = pos_row
            velocities[local_idx] = vel_row
            times[local_idx] = t_row

            x_prev_state.copy_(pos_row)
            v_prev_state.copy_(vel_row)
            t_prev_holder[0] = t_row
            local_idx += 1
            iteration += 1
            time_passed = 0.0
            pbar.update(1)

            if chunking_active and local_idx == buffer_len and iteration < N:
                entry = self._flush_chunk(
                    chunk_dir, chunk_idx, positions, velocities, times, local_idx, global_row_start,
                )
                chunk_entries.append(entry)
                nonlocal manifest_path
                manifest_path = self._write_manifest(chunk_dir, chunk_entries, self.dtype)
                global_row_start += local_idx
                chunk_idx += 1
                local_idx = 0
                positions.zero_()
                velocities.zero_()
                times.zero_()

        while iteration < N:
            _t0 = _time.perf_counter()

            x_prev = x_prev_state
            v_prev = v_prev_state
            t_prev = t_prev_holder[0]

            with torch.no_grad():
                self._time_scalar.fill_(time_passed)
                pos, vel = self.trajectory_sticky(self._time_scalar, x_prev, v_prev)

            dt_hit, i_hit = self._next_hitting_event(pos, vel)
            dt_thaw, i_thaw = self._next_thaw_event(current_time)

            n_frozen = int(self.frozen_mask.sum())
            n_active = self.D - n_frozen

            if self._resample_grad_batch is not None:
                self._resample_grad_batch()

            # Bound-time vs loop-time split, gating whether the sync
            # reduction in fast_grid_bound.py is a meaningful fraction of
            # per-iteration wall time (see plan sec 4a) -- on CUDA,
            # torch.cuda.synchronize() before/after brackets the true
            # device-side cost; on CPU (this dev environment) it's a
            # no-op and the split is diagnostic only, not a real signal.
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            _t_bound0 = _time.perf_counter()
            tau, stats = self._grid_bound(pos.detach(), vel.detach(), dt_hit, dt_thaw, n_active=n_active)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            bound_seconds = _time.perf_counter() - _t_bound0
            grad_evals += stats.get("rate_evals", 0)
            total_bound_violations += stats.get("bound_violations", 0)
            grid_t_max_log.append(self._grid_t_max)

            horizon_used = stats["horizon"]

            row = {
                "rate_evals": stats.get("rate_evals", 0),
                "horizon": horizon_used,
                "time": current_time,
                "event_type": None,
                "accepted": None,
                "n_frozen": n_frozen,
                "n_active": n_active,
                "sparsity": n_frozen / self.D,
                "binding": stats.get("binding"),
                "wall_seconds": None,
                "bound_seconds": bound_seconds,
                "bound_violations": stats.get("bound_violations", 0),
                "rejected_in_window": stats.get("rejected_in_window", False),
                "max_ratio": stats.get("max_ratio", 0.0),
                "curvature_ratio": stats.get("curvature_ratio", 0.0),
                "n_segments": stats.get("n_segments"),
                "effective_spacing": stats.get("effective_spacing"),
            }

            event_accepted = bool(stats["accepted"])

            if event_accepted:
                row["event_type"] = "bounce"
                row["accepted"] = True
            else:
                # No bounce: freeze, thaw, or plain no-event
                eff = stats.get("effective_horizon")
                advance = eff if eff is not None else horizon_used
                tol = 1e-12

                if eff is not None and abs(eff - dt_hit) < tol and i_hit is not None:
                    row["event_type"] = "freeze"
                    with torch.no_grad():
                        self._time_scalar.fill_(time_passed + advance)
                        pos_now, vel_now = self.trajectory_sticky(self._time_scalar, x_prev, v_prev)
                    # current_time has NOT yet been incremented below
                    self._freeze(i_hit, vel_now, current_time + advance)
                    pos_now = pos_now.clone()
                    vel_now = vel_now.clone()
                    # Hand-zero the newly-frozen coordinate
                    pos_now[i_hit] = 0.0
                    vel_now[i_hit] = 0.0

                    time_passed += advance
                    current_time += advance

                    write_row(pos_now, vel_now, t_prev + time_passed, "freeze")

                elif eff is not None and abs(eff - dt_thaw) < tol and i_thaw is not None:
                    row["event_type"] = "thaw"
                    with torch.no_grad():
                        self._time_scalar.fill_(time_passed + advance)
                        pos_now, vel_now = self.trajectory_sticky(self._time_scalar, x_prev, v_prev)
                    vel_now = vel_now.clone()
                    vel_now[i_thaw] = self.frozen_velocity[i_thaw]
                    self._thaw(i_thaw)

                    time_passed += advance
                    current_time += advance

                    write_row(pos_now, vel_now, t_prev + time_passed, "thaw")

                else:
                    row["event_type"] = "no_event"
                    row["accepted"] = False
                    time_passed += advance
                    current_time += advance

            if event_accepted:
                current_time += tau

                self._time_scalar.fill_(time_passed + tau)
                pos_prop, vel_prop = self.trajectory_sticky(self._time_scalar, x_prev, v_prev)

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

                write_row(pos_prop.detach(), vel_flipped.detach(), t_prev + time_passed + tau, "bounce")

            # n_frozen_now derived via +-1 arithmetic instead of a second
            # int(self.frozen_mask.sum()) sync -- freeze/thaw change
            # exactly one coordinate's frozen state per event (never
            # batched), so n_frozen +- 1 is exact. Guarded by an assert
            # (only under _DEBUG_SYNC_MERGE, since the assert itself
            # forces the sync it's meant to avoid) so a future change that
            # ever batches multiple freezes per event fails loudly instead
            # of silently drifting.
            if row["event_type"] == "freeze":
                n_frozen_now = n_frozen + 1
            elif row["event_type"] == "thaw":
                n_frozen_now = n_frozen - 1
            else:
                n_frozen_now = n_frozen
            if _DEBUG_SYNC_MERGE:
                assert n_frozen_now == int(self.frozen_mask.sum()), (
                    f"n_frozen +-1 arithmetic drifted: derived {n_frozen_now}, "
                    f"actual {int(self.frozen_mask.sum())} -- a freeze/thaw event "
                    f"changed more than one coordinate's frozen state."
                )
            row["wall_seconds"] = _time.perf_counter() - _t0
            pbar.set_postfix_str(
                f"t={current_time:.3f} t_max={self._grid_t_max:.4f} "
                f"sparsity={n_frozen_now / self.D:.2f} viol={total_bound_violations}",
                refresh=False,
            )
            diag_log.append(row)
            diag_acc.update(row)

            if chunking_active and flush_diag_every and len(diag_log) >= flush_diag_every:
                diag_path = chunk_dir / f"diag_{chunk_idx:05d}.pt"
                torch.save(diag_log, diag_path)
                diag_log = []
                diag_flushed = True

        pbar.close()

        # Final partial chunk -- flushed unconditionally, regardless of
        # whether it hit buffer_len.
        if chunking_active and local_idx > 0:
            entry = self._flush_chunk(
                chunk_dir, chunk_idx, positions, velocities, times, local_idx, global_row_start,
            )
            chunk_entries.append(entry)
            manifest_path = self._write_manifest(chunk_dir, chunk_entries, self.dtype)

        if chunking_active and flush_diag_every and diag_log:
            diag_path = chunk_dir / f"diag_{chunk_idx:05d}.pt"
            torch.save(diag_log, diag_path)
            diag_log = []
            diag_flushed = True

        if diagnostics:
            if chunking_active:
                _print_diagnostics_chunked(diag_acc, N, grad_evals, t_prev_holder[0], total_bound_violations)
            else:
                self._print_diagnostics(diag_log, N, grad_evals, times[iteration - 1], total_bound_violations)

        # State needed to CONTINUE this exact trajectory in a later sample()
        # call (see this file's module docstring / sample()'s resume_state
        # param doc). Built from x_prev_state/v_prev_state/t_prev_holder --
        # the true final carry-over state, correct regardless of chunking
        # (unlike positions[local_idx-1]/velocities[local_idx-1], which are
        # only valid in the non-chunked path and get zeroed out on a flush
        # in the chunked one). self._grid_t_max deliberately excluded --
        # lives on the instance, see module docstring.
        resume_state = {
            "position": x_prev_state.detach().cpu().clone(),
            "velocity": v_prev_state.detach().cpu().clone(),
            "current_time": float(t_prev_holder[0]),
            "frozen_mask": self.frozen_mask.detach().cpu().clone(),
            "frozen_velocity": self.frozen_velocity.detach().cpu().clone(),
            "thaw_deadline": self.thaw_deadline.detach().cpu().clone(),
        }

        if chunking_active:
            return {
                "chunk_dir": str(chunk_dir),
                "chunk_files": [e["path"] for e in chunk_entries],
                "n_chunks": len(chunk_entries),
                "manifest_path": str(manifest_path) if manifest_path else None,
                "diag_summary": diag_acc.summary(),
                "diagnostics": None if diag_flushed else diag_log,
                "gradient_evals": grad_evals,
                "bound_violations": total_bound_violations,
                "grid_t_max_log": grid_t_max_log,
                "frozen_mask_final": self.frozen_mask.clone(),
                "resume_state": resume_state,
                "N": N,
            }

        return {
            "positions": positions,
            "velocities": velocities,
            "times": times,
            "diagnostics": diag_log,
            "gradient_evals": grad_evals,
            "bound_violations": total_bound_violations,
            "grid_t_max_log": grid_t_max_log,
            "frozen_mask_final": self.frozen_mask.clone(),
            "resume_state": resume_state,
        }

    # ------------------------------------------------------------------
    # Diagnostics (non-chunked path — unchanged from the original)
    # ------------------------------------------------------------------
    @staticmethod
    def _print_diagnostics(diag_log, N, grad_evals, final_time, total_bound_violations):
        import pandas as pd

        df = pd.DataFrame(diag_log)
        n_accept = len(df[df["event_type"] == "bounce"])

        print("\n=== FastGridStickyZigZag Diagnostics ===")
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

        if "bound_seconds" in df.columns:
            bound = df["bound_seconds"].dropna()
            if len(bound) > 0 and len(wall) > 0:
                frac = bound.sum() / max(wall.sum(), 1e-12)
                print(f"Mean _grid_bound sec / iter : {bound.mean():.6f}  "
                      f"({100 * frac:.1f}% of total wall time -- see plan sec 4a: "
                      f"a small fraction here means fast_grid_bound.py's sync reduction "
                      f"is noise relative to rate evaluation cost)")

        print(f"\nSimulation time reached: {float(final_time):.4f}")


class _DiagAccumulator:
    """
    Running aggregate of exactly the statistics _print_diagnostics prints
    from a full pd.DataFrame(diag_log) -- built incrementally so diag_log
    itself can be cleared under chunking without losing the ability to
    print an equivalent summary at the end of sample(). Every row-key
    access uses .get(key, default), not direct indexing: row dicts are
    not guaranteed to carry every key (mirrors the same defensive
    convention in fast_grid_sticky_boomerang.py, whose refresh-event rows
    genuinely do omit several fields -- kept here too for a single shared
    accumulator contract across both sampler files).
    """

    def __init__(self):
        self.n_rows = 0
        self.n_accept = 0
        self.sparsity_sum = 0.0
        self.n_frozen_max = 0
        self.curvature_ratio_sum = 0.0
        self.curvature_ratio_n = 0
        self.wall_seconds_sum = 0.0
        self.wall_seconds_n = 0
        self.bound_seconds_sum = 0.0
        self.bound_seconds_n = 0
        self.event_counts: dict[str, int] = {}
        self.event_horizon_sum: dict[str, float] = {}
        self.binding_counts: dict[str, int] = {}
        self.flipped_coords: set[int] = set()

    def update(self, row: dict) -> None:
        self.n_rows += 1
        if row.get("event_type") == "bounce":
            self.n_accept += 1
        self.sparsity_sum += row.get("sparsity", 0.0)
        self.n_frozen_max = max(self.n_frozen_max, row.get("n_frozen", 0))

        cr = row.get("curvature_ratio")
        if cr is not None:
            self.curvature_ratio_sum += cr
            self.curvature_ratio_n += 1

        ws = row.get("wall_seconds")
        if ws is not None:
            self.wall_seconds_sum += ws
            self.wall_seconds_n += 1

        bs = row.get("bound_seconds")
        if bs is not None:
            self.bound_seconds_sum += bs
            self.bound_seconds_n += 1

        etype = row.get("event_type")
        if etype is not None:
            self.event_counts[etype] = self.event_counts.get(etype, 0) + 1
            self.event_horizon_sum[etype] = self.event_horizon_sum.get(etype, 0.0) + row.get("horizon", 0.0)

        binding = row.get("binding")
        if binding is not None:
            self.binding_counts[binding] = self.binding_counts.get(binding, 0) + 1

        flipped = row.get("flipped_coord")
        if flipped is not None:
            self.flipped_coords.add(flipped)

    def summary(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_accept": self.n_accept,
            "mean_sparsity": self.sparsity_sum / max(self.n_rows, 1),
            "n_frozen_max": self.n_frozen_max,
            "mean_curvature_ratio": self.curvature_ratio_sum / max(self.curvature_ratio_n, 1),
            "mean_wall_seconds": self.wall_seconds_sum / max(self.wall_seconds_n, 1),
            "total_wall_seconds": self.wall_seconds_sum,
            "mean_bound_seconds": self.bound_seconds_sum / max(self.bound_seconds_n, 1),
            "total_bound_seconds": self.bound_seconds_sum,
            "event_counts": dict(self.event_counts),
            "event_mean_horizon": {
                k: self.event_horizon_sum[k] / max(self.event_counts[k], 1) for k in self.event_counts
            },
            "binding_counts": dict(self.binding_counts),
            "n_unique_flipped_coords": len(self.flipped_coords),
        }


def _print_diagnostics_chunked(diag_acc: _DiagAccumulator, N, grad_evals, final_time, total_bound_violations):
    """
    Chunked-path equivalent of FastGridStickyZigZagSampler._print_diagnostics,
    printing from the running accumulator instead of a full
    pd.DataFrame(diag_log) -- the non-chunked path's diagnostics method is
    left completely unchanged; this is an additional function, not a
    modification.
    """
    s = diag_acc.summary()

    print("\n=== FastGridStickyZigZag Diagnostics (chunked) ===")
    print(f"Total gradient evals  : {grad_evals}")
    print(f"Grad evals / skeleton : {grad_evals / max(N, 1):.1f}")
    print(f"Accepted bounces      : {s['n_accept']}")
    print(f"Mean sparsity (frac frozen): {s['mean_sparsity']:.3f}")
    print(f"Max simultaneous frozen    : {s['n_frozen_max']} / {N}")

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

    print(f"Mean curvature ratio  : {s['mean_curvature_ratio']:.4f} "
          f"(isolates the target-curvature part of the bound from the "
          f"n_active*gamma floor)")
    print(f"Flipped-coordinate coverage: {s['n_unique_flipped_coords']} distinct coordinates flipped "
          f"out of {s['n_accept']} accepted bounces")

    print("\n=== Event breakdown ===")
    for etype in ["bounce", "freeze", "thaw", "no_event"]:
        count = s["event_counts"].get(etype, 0)
        if count > 0:
            print(f"  {etype:10s}: {count:5d} events, mean horizon={s['event_mean_horizon'][etype]:.4f}")

    if s["binding_counts"]:
        print("\n=== Binding term (which candidate set the horizon) ===")
        for name, count in s["binding_counts"].items():
            print(f"  {name:12s}: {count:5d}")

    print(f"\nMean wall-sec / iter : {s['mean_wall_seconds']:.6f}")
    print(f"Total wall-sec       : {s['total_wall_seconds']:.2f}")

    if s.get("total_bound_seconds", 0.0) > 0 and s["total_wall_seconds"] > 0:
        frac = s["total_bound_seconds"] / s["total_wall_seconds"]
        print(f"Mean _grid_bound sec / iter : {s['mean_bound_seconds']:.6f}  "
              f"({100 * frac:.1f}% of total wall time -- see plan sec 4a)")

    print(f"\nSimulation time reached: {float(final_time):.4f}")
