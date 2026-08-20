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

---

FastGridStickyBoomerangSampler (this file) -- a copy of
GridStickyBoomerangSampler with chunked skeleton flushing and sync/overhead
reduction (see radiant-finding-church.md for the full plan this implements,
and fast_grid_sticky_zigzag.py's module docstring for the shared design).

The refresh branch (below, in sample()) is the one structural wrinkle not
present in ZigZag: a single pass through the outer while loop can write UP
TO TWO skeleton rows (a regular bounce/freeze/thaw/no-event row, then
possibly a refresh row) before `continue`-ing back to the top. Both rows
are routed through the same write_row() closure used everywhere else, so
the chunk-flush check (which fires after EVERY row write, not once per
loop body) handles this correctly by construction -- see write_row's
docstring in sample().

Also structurally distinct from ZigZag: the refresh branch reads
positions[n-1]/velocities[n-1] directly from the buffer in the ORIGINAL
file (grid_sticky_boomerang.py:590), which is a SECOND, easily-missed
carry-over read site beyond the "top of the loop" one -- it fires AFTER
the bounce/freeze/thaw/no-event branch immediately above it may already
have written the row that triggers a flush-and-reset. This file routes
that read through the same x_prev_state/v_prev_state/t_prev carry-over
state as every other read, not just the top-of-loop one.

Boomerang's postfix-string n_frozen display has a pre-existing staleness
(the original never recomputes n_frozen after a freeze/thaw event for the
non-refresh postfix string -- see sample()'s n_frozen_now assignment
below) that this file deliberately preserves rather than "fixes", since
correcting it would change printed diagnostics output relative to the
original for reasons unrelated to this file's actual goal.
"""

import math
import os
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .grid_boomerang import GridBoomerangSampler
from ..utils.fast_grid_bound import grid_thinning


class FastGridStickyBoomerangSampler(GridBoomerangSampler):
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

        # Reused scalar tensor for torch.as_tensor(time_passed, ...) call
        # sites -- avoids reallocating a fresh 0-dim tensor several times
        # per iteration. Filled via .fill_() at each use site instead.
        self._time_scalar = torch.zeros((), dtype=dtype, device=self.device)

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
    # bottleneck: fires once per freeze/thaw, not once per grid-bound
    # call). Runs eagerly under @torch.no_grad(), never inside a vmap/jvp
    # closure -- same reasoning as fast_grid_sticky_zigzag.py's _freeze/
    # _thaw: the vectorized torch.where updates here are a kernel-launch-
    # count reduction only, not a vmap/jvp-safety requirement.
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
        Merges the argmin-index and deadline-value host reads into one
        combined read where a frozen coordinate exists -- see
        fast_grid_sticky_zigzag.py's identical helper for the exact-below-
        2**24-in-float32 caveat on the index round-trip.
        """
        frozen_idx = torch.where(self.frozen_mask)[0]
        if len(frozen_idx) == 0:
            return float("inf"), None
        deadlines = self.thaw_deadline[frozen_idx]
        local_min = torch.argmin(deadlines)
        i_thaw_f, deadline_i = torch.stack(
            [frozen_idx[local_min].to(deadlines.dtype), self.thaw_deadline[frozen_idx[local_min]]]
        ).tolist()
        i_thaw = int(i_thaw_f)
        dt_thaw = max(deadline_i - current_time, 0.0)
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

        Returns (dt_hit, i_hit) as plain Python float/int|None. The
        float(dt_hit)/int(i_hit) pair from torch.min is merged into one
        combined host read (see fast_grid_sticky_zigzag.py's identical
        helper for the index-round-trip caveat).
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

        dt_hit_t, i_hit_t = torch.min(t_i, dim=0)
        dt_hit, i_hit_f = torch.stack([dt_hit_t, i_hit_t.to(dt_hit_t.dtype)]).tolist()
        if not math.isfinite(dt_hit):
            return math.inf, None
        return dt_hit, int(i_hit_f)

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
    def _refresh_velocity_sticky(self) -> Tensor:
        v = torch.zeros(self.D, dtype=self.dtype, device=self.device)
        active = ~self.frozen_mask

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
    # Chunked-flush bookkeeping — all new, no analog in the original.
    # Identical to fast_grid_sticky_zigzag.py's helpers of the same name.
    # ------------------------------------------------------------------
    def _flush_chunk(
        self, chunk_dir: Path, chunk_idx: int, positions: Tensor, velocities: Tensor,
        times: Tensor, local_idx: int, global_row_start: int,
    ) -> dict:
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
        manifest_path = chunk_dir / "manifest.pt"
        tmp_path = chunk_dir / "manifest.pt.tmp"
        torch.save({"chunks": chunk_entries, "dtype": dtype}, tmp_path)
        os.replace(tmp_path, manifest_path)
        return manifest_path

    # ------------------------------------------------------------------
    # Main sampling loop — overrides the base class: adds freeze/thaw
    # branches to the event dispatch, keyed off effective_horizon (not the
    # raw horizon) so a bound-violation-truncated window can never be
    # mistaken for a fully-examined one when deciding whether a freeze or
    # thaw actually fired (see plan decision #9).
    # ------------------------------------------------------------------
    def sample(
        self, N: int, x0: Optional[Tensor] = None, diagnostics: bool = True,
        chunk_size: Optional[int] = None, chunk_dir: Optional[Path | str] = None,
        flush_diag_every: Optional[int] = None,
    ) -> dict:
        """
        See fast_grid_sticky_zigzag.py's sample() docstring for the full
        chunk_size/chunk_dir/flush_diag_every contract -- identical here.
        The one Boomerang-specific wrinkle: the refresh branch can write a
        SECOND skeleton row within the same while-loop iteration (after
        the regular bounce/freeze/thaw/no-event branch above it may have
        already written one) -- both routed through the same write_row()
        closure, whose flush check fires after every individual row write,
        not once per loop body, so this is handled correctly by
        construction.
        """
        assert self.x_ref is not None, "Call preprocess() first."

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

        self._reset_sticky_state()

        if x0 is None:
            positions[0] = self.x_ref + self._refresh_velocity()
        else:
            positions[0] = x0.to(dtype=self.dtype, device=self.device)
        velocities[0] = self._refresh_velocity()
        times[0] = 0.0

        self._apply_cold_start(positions, velocities)

        # Carry-over state -- seeded AFTER _apply_cold_start, stored via
        # .copy_() into preallocated [D] tensors (not a view into the
        # buffer). Replaces EVERY direct positions[n-1]/velocities[n-1]/
        # times[n-1]-style read in the loop below, including the refresh
        # branch's own read (grid_sticky_boomerang.py:590 in the
        # original) -- not just the top-of-loop one, since that read can
        # fire after a flush-and-reset has already happened this same
        # iteration.
        x_prev_state = torch.zeros(self.D, dtype=self.dtype, device=self.device)
        v_prev_state = torch.zeros(self.D, dtype=self.dtype, device=self.device)
        x_prev_state.copy_(positions[0])
        v_prev_state.copy_(velocities[0])
        t_prev_holder = [0.0]  # boxed so the write_row closure can rebind it

        local_idx = 1  # row 0 already written above
        chunk_idx = 0
        global_row_start = 0
        chunk_entries: list[dict] = []
        manifest_path: Optional[Path] = None

        dt_refresh = float(np.random.exponential(1.0 / self.refresh_rate))
        time_passed = 0.0
        current_time = 0.0
        grad_evals = 0
        total_bound_violations = 0
        diag_log = []
        diag_flushed = False
        grid_t_max_log = []

        # Running diagnostics accumulator -- see fast_grid_sticky_zigzag.py's
        # _DiagAccumulator (reused here, imported below to avoid duplicating
        # the class). Uses .get()-style defensive access throughout since
        # the refresh row genuinely omits several fields (curvature_ratio,
        # n_segments, effective_spacing) that the regular event row carries.
        diag_acc = _DiagAccumulator()

        pbar = tqdm(total=N, desc="FastGridStickyBoomerang", unit="skel")
        pbar.update(1)
        iteration = 1

        def write_row(pos_row: Tensor, vel_row: Tensor, t_row: float) -> None:
            """
            Writes one complete skeleton row into the current buffer,
            updates carry-over state, and flushes (if chunking) the
            instant the buffer fills -- checked after EVERY row write,
            not once per while-loop body. This is what makes the refresh
            branch's potential second row-write-per-iteration safe: it
            calls this same closure, so the flush check runs again,
            correctly, for that second row too.
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

        def flush_diag_if_due() -> None:
            nonlocal diag_log, diag_flushed
            if chunking_active and flush_diag_every and len(diag_log) >= flush_diag_every:
                diag_path = chunk_dir / f"diag_{chunk_idx:05d}.pt"
                torch.save(diag_log, diag_path)
                diag_log = []
                diag_flushed = True

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

            # Bound-time vs loop-time split, gating whether the sync
            # reduction in fast_grid_bound.py is a meaningful fraction of
            # per-iteration wall time (see plan sec 4a) -- no-op sync
            # bracket on CPU (this dev environment), real on CUDA.
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            _t_bound0 = _time.perf_counter()
            tau, stats = self._grid_bound(pos.detach(), vel.detach(), dt_refresh, dt_hit, dt_thaw)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            bound_seconds = _time.perf_counter() - _t_bound0
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
                "bound_seconds": bound_seconds,
                "bound_violations": stats.get("bound_violations", 0),
                "rejected_in_window": stats.get("rejected_in_window", False),
                "max_ratio": stats.get("max_ratio", 0.0),
                "curvature_ratio": stats.get("curvature_ratio", 0.0),
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
                        self._time_scalar.fill_(time_passed + advance)
                        pos_now, vel_now = self.trajectory_sticky(self._time_scalar, x_prev, v_prev)
                    self._freeze(i_hit, vel_now, current_time + advance)
                    pos_now = pos_now.clone()
                    vel_now = vel_now.clone()
                    pos_now[i_hit] = 0.0
                    vel_now[i_hit] = 0.0

                    time_passed += advance
                    current_time += advance
                    dt_refresh -= advance

                    write_row(pos_now, vel_now, t_prev + time_passed)

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
                    dt_refresh -= advance

                    write_row(pos_now, vel_now, t_prev + time_passed)

                else:
                    row["event_type"] = "no_event"
                    time_passed += advance
                    current_time += advance
                    dt_refresh -= advance

            if event_accepted:
                current_time += tau

                self._time_scalar.fill_(time_passed + tau)
                pos_prop, vel_prop = self.trajectory_sticky(self._time_scalar, x_prev, v_prev)
                grad_at_prop = self.grad_U_excess(pos_prop)
                vel_reflected = self.reflect_velocity_sticky(vel_prop, grad_at_prop)

                grad_evals += 1
                row["rate_evals"] += 1

                dt_refresh -= tau

                write_row(pos_prop.detach(), vel_reflected.detach(), t_prev + time_passed + tau)

            # n_frozen_now: preserves the ORIGINAL's pre-existing staleness
            # for the non-refresh postfix string -- the original never
            # recomputes n_frozen after a freeze/thaw event for this
            # specific print, reusing the top-of-loop value unchanged
            # (grid_sticky_boomerang.py:612 reads the same `n_frozen` from
            # line 449, not a post-event value). This is deliberately NOT
            # "fixed" to n_frozen +-1 the way ZigZag's genuinely-duplicated
            # read is merged -- doing so would change printed diagnostics
            # output relative to the original for reasons unrelated to
            # this file's actual goal (see module docstring).
            if dt_refresh <= 1e-14:
                row["wall_seconds"] = _time.perf_counter() - _t0
                diag_log.append(row)
                diag_acc.update(row)
                flush_diag_if_due()

                n_frozen_r = int(self.frozen_mask.sum())
                refresh_row = {
                    "rate_evals": 0, "horizon": 0.0, "time": current_time,
                    "event_type": "refresh", "accepted": None,
                    "n_frozen": n_frozen_r, "n_active": self.D - n_frozen_r,
                    "sparsity": n_frozen_r / self.D, "binding": None,
                    "wall_seconds": 0.0, "bound_violations": 0,
                    "rejected_in_window": False, "max_ratio": 0.0, "curvature_ratio": 0.0,
                }

                if iteration < N:
                    x_prev_refresh = x_prev_state
                    v_prev_refresh = v_prev_state
                    t_prev_refresh = t_prev_holder[0]
                    with torch.no_grad():
                        self._time_scalar.fill_(time_passed)
                        pos_ref, _ = self.trajectory_sticky(self._time_scalar, x_prev_refresh, v_prev_refresh)
                    vel_ref = self._refresh_velocity_sticky()

                    write_row(pos_ref, vel_ref, t_prev_refresh + time_passed)

                dt_refresh = float(np.random.exponential(1.0 / self.refresh_rate))
                diag_log.append(refresh_row)
                diag_acc.update(refresh_row)
                flush_diag_if_due()
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
            diag_acc.update(row)
            flush_diag_if_due()

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
                _print_sticky_diagnostics_chunked(diag_acc, N, grad_evals, t_prev_holder[0], total_bound_violations)
            else:
                self._print_sticky_diagnostics(diag_log, N, grad_evals, times[iteration - 1], total_bound_violations)

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
        }

    # ------------------------------------------------------------------
    # Diagnostics (non-chunked path — unchanged from the original)
    # ------------------------------------------------------------------
    @staticmethod
    def _print_sticky_diagnostics(diag_log, N, grad_evals, final_time, total_bound_violations):
        import pandas as pd

        df = pd.DataFrame(diag_log)
        n_accept = len(df[df["event_type"] == "bounce"])

        print("\n=== FastGridStickyBoomerang Diagnostics ===")
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
    Running aggregate mirroring _print_sticky_diagnostics's summary stats
    -- see fast_grid_sticky_zigzag.py's identical class for the full
    rationale (this is a separate copy, not an import, to keep the two
    sampler files independently self-contained per the plan's file-scope
    constraint). All row-key access uses .get(key, default): the refresh
    row genuinely omits curvature_ratio/n_segments/effective_spacing.
    """

    def __init__(self):
        self.n_rows = 0
        self.n_accept = 0
        self.sparsity_sum = 0.0
        self.n_frozen_max = 0
        self.wall_seconds_sum = 0.0
        self.wall_seconds_n = 0
        self.bound_seconds_sum = 0.0
        self.bound_seconds_n = 0
        self.event_counts: dict[str, int] = {}
        self.event_horizon_sum: dict[str, float] = {}
        self.binding_counts: dict[str, int] = {}

    def update(self, row: dict) -> None:
        self.n_rows += 1
        if row.get("event_type") == "bounce":
            self.n_accept += 1
        self.sparsity_sum += row.get("sparsity", 0.0)
        self.n_frozen_max = max(self.n_frozen_max, row.get("n_frozen", 0))

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

    def summary(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_accept": self.n_accept,
            "mean_sparsity": self.sparsity_sum / max(self.n_rows, 1),
            "n_frozen_max": self.n_frozen_max,
            "mean_wall_seconds": self.wall_seconds_sum / max(self.wall_seconds_n, 1),
            "total_wall_seconds": self.wall_seconds_sum,
            "mean_bound_seconds": self.bound_seconds_sum / max(self.bound_seconds_n, 1),
            "total_bound_seconds": self.bound_seconds_sum,
            "event_counts": dict(self.event_counts),
            "event_mean_horizon": {
                k: self.event_horizon_sum[k] / max(self.event_counts[k], 1) for k in self.event_counts
            },
            "binding_counts": dict(self.binding_counts),
        }


def _print_sticky_diagnostics_chunked(diag_acc: _DiagAccumulator, N, grad_evals, final_time, total_bound_violations):
    """
    Chunked-path equivalent of
    FastGridStickyBoomerangSampler._print_sticky_diagnostics, printing
    from the running accumulator instead of a full pd.DataFrame(diag_log)
    -- the non-chunked path's diagnostics method is left completely
    unchanged; this is an additional function, not a modification.
    """
    s = diag_acc.summary()

    print("\n=== FastGridStickyBoomerang Diagnostics (chunked) ===")
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
            f"validity signal for the run, not a tunable knob. Consider a smaller "
            f"grid_spacing. ***\n"
        )
    else:
        print("Bound violations      : 0")

    print("\n=== Event breakdown ===")
    for etype in ["bounce", "freeze", "thaw", "no_event", "refresh"]:
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
