"""
ZigZag sampler using ONLY the grid-based upper bound (Andral & Kamatani
2024) for Poisson thinning. Not a subclass of
sazz.samplers.AutomaticZigZagSampler -- independent copy, kept isolated
from sazz/samplers/, mirroring grid_boomerang.py's isolation convention.

Differs fundamentally from grid_boomerang.py in how the bound is built:
ZigZag's rate lambda_ZZ(t) = sum_j (grad_j U(x_t) * v_j)_+ + D*gamma is a SUM
of D independently-kinked per-coordinate terms, not Boomerang's single
smooth signed inner product (Andral & Kamatani Section 4.4.2). Naively
bounding the aggregate signed sum fails (coordinates can cancel and hide
curvature), so this sampler builds a per-coordinate Algorithm-2 bound for
every coordinate (batched via one shared vmap(jvp(...)) call across all D
coordinates and every grid time simultaneously -- see
_make_rate_and_grad_fn), then sums the per-coordinate bounds via
grid_bound.build_grid_bound_vectorized. The resulting scalar
piecewise-constant bound feeds the SAME single shared Poisson thinning
process (grid_bound.grid_thinning) that grid_boomerang.py already uses,
via its bound_fn/rate_offset hooks -- only the bound CONSTRUCTION differs
per PDMP; the thinning/Section-4.7/Algorithm-4 machinery is shared exactly.


Critical correctness note carried over from grid_boomerang.py, but load-
bearing here specifically because it fires on essentially every call:
_grid_bound MUST capture stats["horizon"] (the horizon actually used to
draw tau) BEFORE applying the Algorithm-4 adaptation to self._grid_t_max --
the adaptation mutates self._grid_t_max in place, so sample() must never
re-derive "the horizon this tau was drawn against" from self._grid_t_max
after _grid_bound returns (that value has already moved). sample() branches
on stats["accepted"] and reads stats["horizon"], never a fresh
min(self._grid_t_max, ...) recomputation.
"""

import math
import time as _time
from functools import partial
from typing import Callable, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

# from ..utils.grid_bound import grid_thinning, build_grid_bound_vectorized
from ..utils.fast_grid_bound import grid_thinning, build_grid_bound_vectorized


class GridZigZagSampler(nn.Module):
    """
    ZigZag sampler with the grid-based (Andral & Kamatani 2024) upper bound
    as its only thinning strategy, using the per-coordinate vectorized
    bound construction their Section 4.4.2 describes for a ZigZag-shaped
    rate (see grid_bound.build_grid_bound_vectorized).

    grad_target: PLAIN FUNCTION gradient of the (negative) log-target,
    built via torch.func.grad(energy) -- must not detach its input, so
    grid times can be differentiated through it via jvp (same contract as
    grid_boomerang.py's grad_target).
    gamma: refreshment floor added per-coordinate before summing (matches
    sazz.samplers.AutomaticZigZagSampler's default of 0.01). D*gamma
    supplies genuine irreducibility slack to the bound -- see grid_spacing
    below for a caveat about scale.
    grid_t_max_init: initial adaptive horizon; ZigZag's rate has no
    periodic structure to anchor a default to (unlike Boomerang's ~pi
    period), so this defaults to 0.1, matching the CPU ZigZag sampler's own
    t_max default -- Algorithm 4 will move it from there regardless.
    grid_spacing: target grid-node spacing (NOT segment count, same
    convention as grid_boomerang.py). Unlike Boomerang's pi/16 (anchored to
    the known pi/4 first-harmonic curvature scale), ZigZag has no principled
    anchor for this value -- it depends on the target's Hessian curvature
    along v, which varies per-target and per-region of parameter space.
    Default 0.01 is a STARTING point only (roughly grid_t_max_init/10,
    independent of n_segments); tune per-target by watching
    bound_violations/max_ratio/curvature_ratio/effective_spacing in the
    diagnostics output, do not trust the default blindly.
    n_segments: CAP on the per-call segment count derived from grid_spacing
    (same "spacing is the tunable, n_segments is the ceiling" convention as
    grid_boomerang.py). Note self._grid_t_max's Algorithm-4 equilibrium is
    itself D-dependent here (roughly 1/(D*gamma + curvature-driven rate)),
    unlike Boomerang's D-independent pi/4-anchored horizon -- at large D
    this can pin n_segments at its floor of 2, making grid_spacing
    effectively inert; effective_spacing is logged per-iteration so this is
    visible rather than a silent drift.
    strategy: "vectorized_signed" (default) bounds each coordinate's SMOOTH
    signed rate separately (no kink, robust, matches the paper's own
    recommendation -- fewest bound violations in their Figure 6), clamping
    each coordinate's bound to >=0 individually before summing.
    "vectorized_not_signed" bounds each coordinate's already-CLAMPED
    (kinked) rate directly -- tighter, but can silently under-bound if a
    coordinate's signed rate crosses zero strictly inside a segment (the
    missed mass is undetectable by the accept/reject ratio check, since no
    proposal is ever generated there). Use vectorized_not_signed only for
    reproducing Andral & Kamatani's Figure 6 comparison, not as an
    unattended production default.
    chunk_size: passed to the vmap composition in _make_rate_and_grad_fn.
    [K, D] with K = n_segments+1 concurrent forward+backward graphs through
    the target can OOM on larger BNN/CNN targets; default None keeps the
    current unchunked behavior.
    alpha_plus/alpha_minus/alpha_violation: Algorithm 4 horizon growth/
    shrink constants, identical role and defaults to grid_boomerang.py.
    grid_kwargs: forwarded to grid_thinning (max_iter, min_window,
    max_violations).
    """

    def __init__(
        self,
        grad_target: Callable[[Tensor], Tensor],
        D: int,
        gamma: float = 0.01,
        grid_t_max_init: float = 0.1,
        n_segments: int = 20,
        grid_spacing: float = 0.01,
        alpha_plus: float = 1.01,
        alpha_minus: float = 1.04,
        alpha_violation: float = 2.0,
        strategy: Literal["vectorized_signed", "vectorized_not_signed"] = "vectorized_signed",
        chunk_size: Optional[int] = None,
        grid_kwargs: Optional[dict] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.D = D
        self.grad_target = grad_target
        self.gamma = float(gamma)
        self.n_segments = n_segments
        self.grid_spacing = grid_spacing
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.alpha_violation = alpha_violation
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.grid_kwargs = grid_kwargs or {
            "max_iter": 200,
            "min_window": 1e-8,
            "max_violations": 10,
        }
        self.dtype = dtype
        self.device = torch.device(device)

        # Adaptive horizon -- persists across _grid_bound calls (sampler
        # state; grid_bound.py itself stays stateless). No x_ref/Sigma_inv/
        # preprocess() -- ZigZag has no reference measure.
        self._grid_t_max = float(grid_t_max_init)

    # ------------------------------------------------------------------
    # Dynamics -- ON the computational graph. `t` must be a differentiable
    # tensor op, not a Python float, so torch.func.jvp can trace a tangent
    # through it -- same discipline as grid_boomerang.py's trajectory, even
    # though `+`/`*` don't themselves require it (keeps the convention
    # consistent so a future edit can't accidentally detach t).
    # ------------------------------------------------------------------
    def trajectory(self, t: Tensor, x: Tensor, v: Tensor):
        x_t = x + t * v
        v_t = v
        return x_t, v_t

    def flip_velocity(self, v: Tensor, i: int) -> Tensor:
        """
        Flip the i-th coordinate of the velocity. Off-graph only -- called
        post-accept in sample(), never inside the vmapped/jvp'd rate
        closures (same as grid_boomerang.py's reflect_velocity), so no
        functional/no-in-place-mutation requirement applies here.
        """
        v_new = v.clone()
        v_new[i] = -v_new[i]
        return v_new

    # ------------------------------------------------------------------
    # Rate closures for the grid bound
    # ------------------------------------------------------------------
    def _rate_scalar(self, t: float, x: Tensor, v: Tensor, want_per_coord: bool = False):
        """
        Off-graph, true rate lambda_ZZ(t) = sum_j clamp(v_j*grad_j U(x_t), 0)
        + D*gamma -- for the accept/reject check inside grid_thinning. Same
        regardless of `strategy` (only the BOUND construction differs; the
        true rate used to accept/reject is always this one function).

        want_per_coord: if True, also returns the per-coordinate clamped
        rate vector (before summing) so a post-accept flip draw can reuse
        it instead of a second gradient evaluation at the same point.
        """
        with torch.no_grad():
            t_t = torch.as_tensor(t, dtype=self.dtype, device=self.device)
            x_t, v_t = self.trajectory(t_t, x, v)
            grad = self.grad_target(x_t)
            per_coord = torch.clamp(v_t * grad, min=0.0) + self.gamma
            rate = float(per_coord.sum())
        if want_per_coord:
            return rate, per_coord
        return rate

    def _per_coord_rates(self, t: float, x: Tensor, v: Tensor) -> Tensor:
        """
        Per-coordinate rates at time t. Fallback path when a stashed vector
        from _rate_scalar isn't available (e.g. called independently of an
        accept-check) -- feeds the post-accept categorical flip draw.
        """
        with torch.no_grad():
            t_t = torch.as_tensor(t, dtype=self.dtype, device=self.device)
            x_t, v_t = self.trajectory(t_t, x, v)
            grad = self.grad_target(x_t)
            return torch.clamp(v_t * grad, min=0.0) + self.gamma

    def _make_rate_and_grad_fn(self, x: Tensor, v: Tensor):
        """
        Batched (value, derivative) closure for the per-coordinate signed
        rate g_j(t) = grad_j U(x_t) * v_j, anchored at (x, v). x, v are
        detached; only t is differentiable, per grid_bound.py's contract.

        ONE torch.func.jvp of the FULL gradient vector x -> grad_target(x_t)
        (composed with x_t = x + t*v via the chain rule inside jvp) gives
        every coordinate's (value, derivative) pair simultaneously -- v is
        constant in t for ZigZag's linear trajectory, so the product rule
        collapses to d/dt[grad_j U(x_t) * v_j] = v_j * d/dt[grad_j U(x_t)],
        and d/dt[grad U(x_t)] (the full vector) is exactly the tangent jvp
        computes. This means rate_evals for one call is n_segments+1
        (shared across all D coordinates), NOT D*(n_segments+1) -- do not
        restructure this as D independent per-coordinate closures each
        redoing a full-D gradient call, which would be D-times wasteful.

        Returns rate_and_grad_fn(t_batch: Tensor[K]) -> (y_full, d_full),
        both [D, K] -- exactly the shape build_grid_bound_vectorized
        expects.
        """
        x = x.detach()
        v = v.detach()

        def gradU_at_t(t: Tensor) -> Tensor:
            x_t, _ = self.trajectory(t, x, v)
            return self.grad_target(x_t)

        def rate_and_grad_fn(t_batch: Tensor):
            vmap_fn = torch.func.vmap(
                lambda ti: torch.func.jvp(gradU_at_t, (ti,), (torch.ones_like(ti),)),
                chunk_size=self.chunk_size,
            )
            g, dgdt = vmap_fn(t_batch)  # each [K, D]
            y_full = (g * v).transpose(0, 1)      # [D, K] signed per-coord rate
            d_full = (dgdt * v).transpose(0, 1)   # [D, K] its time-derivative

            if self.strategy == "vectorized_not_signed":
                # Bound the already-clamped (kinked) rate directly: clamp
                # the value and zero the derivative where the signed rate
                # is negative -- both derived from the SAME underlying
                # (y_full, d_full), no second autodiff pass needed.
                d_full = torch.where(y_full > 0, d_full, torch.zeros_like(d_full))
                y_full = torch.clamp(y_full, min=0.0)

            return y_full, d_full

        return rate_and_grad_fn

    # ------------------------------------------------------------------
    # Grid bound dispatch -- single window per call.
    # ------------------------------------------------------------------
    def _grid_bound(self, pos: Tensor, vel: Tensor) -> tuple[float, dict]:
        """
        No dt_refresh/dt_hit/dt_thaw -- plain (non-sticky) ZigZag has no
        refresh clock and no sticky candidates in this pass, so horizon is
        simply self._grid_t_max with nothing to min() against.
        """
        horizon = self._grid_t_max

        # n_segments computed ONCE from the incoming horizon and held fixed
        # for the lifetime of this grid_thinning call -- including across
        # any internal Section 4.7 shrink/rebuild on a bound violation
        # (same convention as grid_boomerang.py's _grid_bound).
        n_segments = int(min(max(math.ceil(horizon / self.grid_spacing), 2), self.n_segments))
        effective_spacing = horizon / n_segments

        rate_and_grad_fn = self._make_rate_and_grad_fn(pos, vel)
        rate_scalar_fn = partial(self._rate_scalar, x=pos.detach(), v=vel.detach())
        bound_fn = partial(
            build_grid_bound_vectorized,
            signed=(self.strategy == "vectorized_signed"),
            offset=self.D * self.gamma,
        )

        tau, stats = grid_thinning(
            rate_and_grad_fn, rate_scalar_fn, horizon,
            n_segments=n_segments, device=self.device, dtype=self.dtype,
            diagnostics=True, bound_fn=bound_fn, rate_offset=self.D * self.gamma,
            **self.grid_kwargs,
        )
        stats["n_segments"] = n_segments
        stats["effective_spacing"] = effective_spacing

        # Capture the horizon THIS call actually used to draw tau BEFORE
        # applying the Algorithm-4 adaptation below -- the adaptation
        # mutates self._grid_t_max in place, so sample() must read
        # stats["horizon"], never re-derive it from self._grid_t_max after
        # this method returns (that value has already moved). See module
        # docstring's "critical correctness note."
        stats["horizon"] = horizon

        # --- t_max adaptation (Algorithm 4), same logic as grid_boomerang.py ---
        if stats["violated"]:
            self._grid_t_max /= self.alpha_violation
        elif stats["rejected_in_window"]:
            self._grid_t_max /= self.alpha_minus
        elif tau == math.inf:
            eff = stats["effective_horizon"]
            if eff is not None and eff >= horizon - 1e-12:
                self._grid_t_max *= self.alpha_plus

        return tau, stats

    # ------------------------------------------------------------------
    # Initial velocity -- Rademacher +-1 per coordinate, NOT a Gaussian
    # refresh draw (ZigZag has no reference measure to refresh toward).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _initial_velocity(self) -> Tensor:
        signs = torch.randint(0, 2, (self.D,), device=self.device, dtype=torch.int64)
        return (2 * signs - 1).to(dtype=self.dtype)

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------
    def sample(self, N: int, x0: Optional[Tensor] = None, diagnostics: bool = True) -> dict:
        """
        x0=None defaults to N(0, I) (matching the CPU AutomaticZigZagSampler's
        default -- ZigZag has no reference measure, so Boomerang's
        x_ref + refresh convention doesn't apply here). Pass x0=x_ref
        explicitly to warm-start from the same point used for a Boomerang
        comparison run on the same target -- ZigZag's trajectory has no
        structural dependency on a reference point, so x_ref can be used
        purely as an ordinary initial position, with no preprocess() call
        needed.
        """
        positions = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        velocities = torch.zeros(N, self.D, dtype=self.dtype, device=self.device)
        times = torch.zeros(N, dtype=self.dtype, device=self.device)

        if x0 is None:
            positions[0] = torch.randn(self.D, dtype=self.dtype, device=self.device)
        else:
            positions[0] = x0.to(dtype=self.dtype, device=self.device)
        velocities[0] = self._initial_velocity()
        times[0] = 0.0

        time_passed = 0.0
        current_time = 0.0
        grad_evals = 0
        total_bound_violations = 0
        diag_log = []
        grid_t_max_log = []

        pbar = tqdm(total=N, desc="GridZigZag", unit="skel")
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

            tau, stats = self._grid_bound(pos.detach(), vel.detach())
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
                "wall_seconds": None,
                "bound_violations": stats.get("bound_violations", 0),
                "max_ratio": stats.get("max_ratio", 0.0),
                "curvature_ratio": stats.get("curvature_ratio", 0.0),
                "n_segments": stats.get("n_segments"),
                "effective_spacing": stats.get("effective_spacing"),
            }

            # Branch on stats["accepted"] (equivalently tau < math.inf, what
            # grid_thinning's own _make_stats already sets), NEVER on a
            # re-derived min(self._grid_t_max, ...) -- self._grid_t_max has
            # already been mutated by _grid_bound's Algorithm-4 step by the
            # time we get here, so a fresh recomputation would silently
            # discard a genuinely accepted event whenever the adaptation
            # shrank the horizon below tau (see module docstring).
            event_accepted = bool(stats["accepted"])

            if event_accepted:
                row["event_type"] = "bounce"
                row["accepted"] = True
            else:
                row["event_type"] = "no_event"
                row["accepted"] = False
                eff = stats.get("effective_horizon")
                advance = eff if eff is not None else horizon_used
                time_passed += advance
                current_time += advance

            if event_accepted:
                current_time += tau

                pos_prop, vel_prop = self.trajectory(
                    torch.as_tensor(time_passed + tau, dtype=self.dtype, device=self.device),
                    x_prev, v_prev,
                )

                # Pick which coordinate flips: categorical over true
                # per-coordinate rates at the accepted time. Off graph.
                pos_np = pos_prop.detach()
                vel_np = vel_prop.detach()
                rates_i = self._per_coord_rates(0.0, pos_np, vel_np)
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

        print("\n=== GridZigZag Diagnostics ===")
        print(f"Total gradient evals  : {grad_evals}")
        print(f"Grad evals / skeleton : {grad_evals / max(N, 1):.1f}")
        print(f"Accepted bounces      : {n_accept}")

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
                  f"(isolates the target-curvature part of the bound from the D*gamma "
                  f"floor -- a weaker signal than max_ratio for how well grid_spacing "
                  f"is tuned; see class docstring)")

        if "flipped_coord" in df.columns:
            flips = df["flipped_coord"].dropna()
            if len(flips) > 0:
                n_unique = flips.nunique()
                print(f"Flipped-coordinate coverage: {n_unique} distinct coordinates flipped "
                      f"out of {n_accept} accepted bounces")

        print("\n=== Event breakdown ===")
        for etype in ["bounce", "no_event"]:
            sub = df[df["event_type"] == etype]
            if len(sub) > 0:
                print(f"  {etype:10s}: {len(sub):5d} events, mean horizon={sub['horizon'].mean():.4f}")

        if "effective_spacing" in df.columns:
            print(f"\nMean effective grid spacing: {df['effective_spacing'].mean():.4f} "
                  f"(vs. requested grid_spacing -- large drift suggests n_segments is "
                  f"capping resolution rather than grid_spacing controlling it)")

        wall = df["wall_seconds"].dropna()
        if len(wall) > 0:
            print(f"\nMean wall-sec / iter : {wall.mean():.6f}")
            print(f"Total wall-sec       : {wall.sum():.2f}")

        print(f"\nSimulation time reached: {float(final_time):.4f}")
