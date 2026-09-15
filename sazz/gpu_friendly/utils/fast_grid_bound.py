"""
Grid-based piecewise-constant upper bound for Poisson thinning (Andral &
Kamatani 2024, Algorithms 2/3). Horizon adaptation (Algorithm 4) is the
caller's job -- see grid_boomerang.py's `_grid_bound`.

`rate_and_grad_fn(t: Tensor[K]) -> (y, d)` must be forward-mode (jvp), not
reverse-over-reverse, e.g.:
    def rate_and_grad_fn(t_batch):
        return torch.func.vmap(
            lambda ti: torch.func.jvp(g_scalar, (ti,), (torch.ones_like(ti),))
        )(t_batch)

Single-window contract: every function here handles exactly one window
[0, horizon]; horizon adaptation across calls is the caller's job.

`stats["effective_horizon"]`: None on accept (check `tau < horizon`
instead); == horizon if no event was found; on a bound-violation exit,
the last violation's proposal time (only that much was validly examined).
`stats["bound_violations"] > 0` is a validity signal, not a tuning knob,
and should always be surfaced.

---

fast_grid_bound.py -- rewrites sample_from_grid_bound and grid_thinning's
per-proposal reads to use plain Python lists instead of re-touching
knot_times/seg_bounds/cum_bound tensors on every proposal (built once per
window, ~7 host syncs/proposal down to 3 per window build).

Two things this deliberately does NOT do:
1. Reuse sample_from_grid_bound's segment index for grid_thinning's own
   seg_idx lookup -- they're different searches (see
   _sample_from_grid_bound_list's docstring); conflating them would force
   lam_bound ~= 0 on any proposal crossing a zero-height segment.
2. Batch the torch.rand(()) draws -- both are already CPU tensors
   (no device= anywhere in this codebase), so there's no sync to save.

PRECISION NOTE: the original rounds the query to `dtype` (float32 on
GPU) only for the searchsorted/bisect comparison, not for the
surrounding arithmetic. `_round_query` reproduces exactly that cast --
skipping it would disagree with the original by one index whenever a
query sits within half a float32 ULP of a knot.
"""


import bisect
import math
from typing import Callable, Optional

import torch
from torch import Tensor


def _make_stats(tau: float, proposals: int, rate_evals: int, max_ratio: float,
                 bound_violations: int, rejected_in_window: bool,
                 violated: bool, effective_horizon: Optional[float],
                 curvature_ratio: float = 0.0) -> dict:
    return {
        "accepted": tau < math.inf,
        "rejected": tau == math.inf,
        "proposals": proposals,
        "rate_evals": rate_evals,
        "max_ratio": max_ratio,
        "curvature_ratio": curvature_ratio,
        "bound_violations": bound_violations,
        "rejected_in_window": rejected_in_window,
        "violated": violated,
        "effective_horizon": effective_horizon,
        "tau": tau,
    }


def build_grid_bound(
    rate_and_grad_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
    horizon: float,
    n_segments: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: Optional[float] = None,
    t_offset: float = 0.0,
    bound_inflation: float = 0.01,
):
    """
    Algorithm 2 (Andral & Kamatani): piecewise-constant upper bound of a
    signed rate g(t) over [0, horizon], from tangent lines at n_segments+1
    grid nodes.

    eps: degenerate-slope tolerance for Eq. 2's tangent-intersection formula
    (default 1e-4).
    
    t_offset: evaluates rate_and_grad_fn shifted by t_offset while knot_times
    stay local to [0, horizon] (needed for Section 4.7 window rebuilds).
    bound_inflation: multiplicative safety margin on seg_bounds (default 0.01).

    Returns knot_times[n_segments+1], seg_bounds[n_segments] (unclamped
    Lambda_i), cum_bound[n_segments+1] (cumulative integral of
    max(Lambda, 0)), rate_evals (== n_segments+1).

    """
    if eps is None:
        eps = 1e-4

    # t_local is what's returned as knot_times and used for all downstream
    # width/clamp arithmetic; t_eval is ONLY used to evaluate rate_and_grad_fn
    t_local = torch.linspace(0.0, horizon, n_segments + 1, device=device, dtype=dtype)
    t_eval = t_local + t_offset if t_offset != 0.0 else t_local
    y, d = rate_and_grad_fn(t_eval)

    t0, t1 = t_local[:-1], t_local[1:]
    y0, y1 = y[:-1], y[1:]
    d0, d1 = d[:-1], d[1:]

    denom = d0 - d1
    degenerate = denom.abs() < eps

    # Safe denom to avoid NaN/Inf propagating through the non-degenerate
    # branch's arithmetic even where we'll discard the result via `where`.
    safe_denom = torch.where(degenerate, torch.ones_like(denom), denom)
    x_i_raw = (y1 - y0 + d0 * t0 - d1 * t1) / safe_denom
    # Degenerate fallback (Eq. 2's d_i == d_{i+1} case): x_i = t_i, m_i = y_i.
    x_i = torch.where(degenerate, t0, x_i_raw)
    x_i = torch.clamp(x_i, min=t0, max=t1)

    m_i = d0 * x_i + y0 - d0 * t0
    m_i = torch.where(torch.isfinite(m_i), m_i, torch.minimum(y0, y1))

    seg_bounds = torch.maximum(torch.maximum(y0, y1), m_i) * (1.0 + bound_inflation)  # unclamped (design decision #4)

    seg_widths = t1 - t0
    seg_integrals = torch.clamp(seg_bounds, min=0.0) * seg_widths
    cum_bound = torch.cat([
        torch.zeros(1, device=device, dtype=dtype),
        torch.cumsum(seg_integrals, dim=0),
    ])

    return t_local, seg_bounds, cum_bound, n_segments + 1


def build_grid_bound_vectorized(
    rate_and_grad_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
    horizon: float,
    n_segments: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: Optional[float] = None,
    *,
    signed: bool = True,
    offset: float = 0.0,
    t_offset: float = 0.0,
    bound_inflation: float = 0.01,
):
    """
    Per-coordinate Algorithm 2 (Andral & Kamatani): ZigZag's
    rate is a sum of D independently-kinked per-coordinate terms, so a
    single aggregate tangent-line bound can under-bound when coordinates cancel. 
    Instead, bound each coordinate's tangent line separately, then sum.

    rate_and_grad_fn(t_batch: Tensor[K]) -> (y_full, d_full): Tensor[D, K]
    each, evaluated for all D coordinates in one batched call
    (rate_evals == n_segments + 1, not D * (n_segments + 1)).

    signed: True (default, "vectorized_signed") -- clamp each coordinate's
    bound to >= 0 individually before summing over D. False
    ("vectorized_not_signed") -- caller already clamped the per-coordinate
    rate; can silently under-bound where a coordinate's rate crosses
    positive strictly inside a segment.

    offset: added to the summed scalar seg_bounds before forming the
    integral (e.g. ZigZag's D*gamma refreshment floor). Must be added here,
    not by the caller, so the accept/reject check and the sampled integral
    stay consistent.

    t_offset, bound_inflation: same contract as build_grid_bound

    eps: RELATIVE tolerance (default 1e-8), unlike build_grid_bound's
    ABSOLUTE 1e-4 -- per-coordinate slopes are ~1/D the aggregate scale

    Returns knot_times[n_segments+1] (local), seg_bounds[n_segments]
    (scalar, summed over D, offset included), cum_bound[n_segments+1],
    rate_evals (== n_segments+1).
    """
    if eps is None:
        eps = 1e-8

    t_local = torch.linspace(0.0, horizon, n_segments + 1, device=device, dtype=dtype)
    t_eval = t_local + t_offset if t_offset != 0.0 else t_local
    y_full, d_full = rate_and_grad_fn(t_eval)  # each [D, K]

    t0, t1 = t_local[:-1], t_local[1:]                    # [K-1]
    y0, y1 = y_full[:, :-1], y_full[:, 1:]                 # [D, K-1]
    d0, d1 = d_full[:, :-1], d_full[:, 1:]                 # [D, K-1]

    denom = d0 - d1
    scale = torch.maximum(torch.maximum(d0.abs(), d1.abs()), torch.full_like(denom, 1e-30))
    degenerate = denom.abs() < eps * scale

    safe_denom = torch.where(degenerate, torch.ones_like(denom), denom)
    x_i_raw = (y1 - y0 + d0 * t0 - d1 * t1) / safe_denom
    x_i = torch.where(degenerate, t0, x_i_raw)
    x_i = torch.clamp(x_i, min=t0, max=t1)

    m_i = d0 * x_i + y0 - d0 * t0
    m_i = torch.where(torch.isfinite(m_i), m_i, torch.minimum(y0, y1))

    seg_bounds_percoord = torch.maximum(torch.maximum(y0, y1), m_i)  # [D, K-1]
    if signed:
        seg_bounds_percoord = torch.clamp(seg_bounds_percoord, min=0.0)

    seg_bounds = (seg_bounds_percoord.sum(dim=0) + offset) * (1.0 + bound_inflation)  # [K-1] scalar per segment

    seg_widths = t1 - t0
    seg_integrals = torch.clamp(seg_bounds, min=0.0) * seg_widths
    cum_bound = torch.cat([
        torch.zeros(1, device=device, dtype=dtype),
        torch.cumsum(seg_integrals, dim=0),
    ])

    return t_local, seg_bounds, cum_bound, n_segments + 1


def _round_query(q: float, dtype: torch.dtype) -> float:
    """Rounds q through dtype before a list search, matching the
    original tensor-backed search's implicit cast precision (float32
    on GPU) instead of comparing at full double precision."""
    return float(torch.tensor(q, dtype=dtype))


def _sample_from_grid_bound_list(
    knot_times_list: list[float], seg_bounds_list: list[float], cum_bound_list: list[float],
    e: float, horizon: float, dtype: torch.dtype,
) -> tuple[float, Optional[int]]:
    """
    Algorithm 3 (Andral & Kamatani): given accumulated Exp(1) budget `e`, find the
    proposed event time under the piecewise-constant bound. Fixes the
    paper's off-by-one (the segment containing the solution starts at
    t_{i-1}, not t_i) via bisect_right, the list analog of the original's
    torch.searchsorted(..., side="right").

    Returns (tau, seg_idx); seg_idx is None on both "no event this window"
    exits (budget exceeds the total integral, or a trailing zero-height
    segment) -- callers needing seg_idx (grid_thinning) must re-locate it
    themselves rather than reuse the zero-height segment's index.

    Precision: matches the tensor version exactly -- `e` is rounded to
    `dtype` only for the bisect comparison, never for the `e >= total` or
    `remaining` arithmetic, mirroring the original's searchsorted cast.
    """
    total = cum_bound_list[-1]
    if e >= total:
        return horizon, None

    # side="right": first index i such that cum_bound[i] > e. The segment
    # containing e is [i-1, i]; clamp defensively for the e==0 edge case.
    e_q = _round_query(e, dtype)
    idx = bisect.bisect_right(cum_bound_list, e_q)
    idx = max(1, min(idx, len(seg_bounds_list)))
    i = idx - 1

    t_i = knot_times_list[i]
    lam_i = seg_bounds_list[i]
    remaining = e - cum_bound_list[i]

    if lam_i <= 1e-14:
        # Zero-height segment: cannot supply any more integral, defer to
        # the next segment's start (mirrors falling through to i+1).
        if i + 1 < len(knot_times_list):
            return knot_times_list[i + 1], None
        return horizon, None

    # Within-segment rate is constant, solve linearly: remaining = lam_i * (tau - t_i).
    tau = t_i + remaining / lam_i
    return tau, i


def grid_thinning(
    rate_and_grad_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
    rate_scalar_fn: Callable[[float], float],
    horizon: float,
    n_segments: int = 20,
    max_iter: int = 200,
    min_window: float = 1e-8,
    max_violations: int = 10,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    eps: Optional[float] = None,
    diagnostics: bool = True,
    *,
    bound_fn: Optional[Callable] = None,
    rate_offset: float = 0.0,
):
    """
    Single-window grid-based Poisson thinning (Algorithms 2+3 + Section
    4.7's bound-violation response). See module docstring for the
    `effective_horizon` / violation semantics contract.

    rate_scalar_fn: single-point signed rate for the accept/reject check.
    min_window: stop subdividing if a violation-driven shrink would take
    the remaining sub-window below this width.
    max_violations: cap on violations handled per call.

    bound_fn: bound-construction callable to use in place of
    build_grid_bound, e.g. a functools.partial(build_grid_bound_vectorized,
    signed=..., offset=...) for ZigZag's per-coordinate rate. Defaults to
    build_grid_bound (Boomerang's case). Must accept `t_offset` as a
    keyword arg.

    rate_offset: the additive constant (e.g. ZigZag's D*gamma refreshment
    floor) the active bound_fn folds in, used only to report a
    curvature-only ratio, `stats["curvature_ratio"] = (lam_true -
    rate_offset) / (lam_bound - rate_offset)`, alongside `max_ratio`.
    Default 0.0 makes curvature_ratio equal max_ratio (Boomerang's case).

    Returns (tau, stats) if diagnostics else tau.

    Per-proposal reads against knot_times/seg_bounds/cum_bound use plain
    Python lists (one .tolist() per window build, not per proposal) rather
    than re-touching tensors every iteration -- see module docstring and
    _round_query for the precision-matching details this depends on.
    """

    if device is None:
        device = torch.device("cpu")
    build_fn = bound_fn if bound_fn is not None else build_grid_bound

    window_start = 0.0
    window_end = horizon
    rate_evals = 0
    proposals = 0
    max_ratio = 0.0
    curvature_ratio = 0.0
    bound_violations = 0
    rejected_in_window = False
    violated = False

    knot_times, seg_bounds, cum_bound, n_evals = build_fn(
        rate_and_grad_fn, window_end - window_start, n_segments, device, dtype, eps,
        t_offset=window_start,
    )
    rate_evals += n_evals
    knot_times_list = knot_times.tolist()
    seg_bounds_list = seg_bounds.tolist()
    cum_bound_list = cum_bound.tolist()

    e_budget = 0.0

    for _ in range(max_iter):
        # CPU tensor, stays untouched from the original.
        u = torch.rand(()).item()
        e_budget += -math.log(1.0 - u)

        tau_local, _unused_seg_idx = _sample_from_grid_bound_list(
            knot_times_list, seg_bounds_list, cum_bound_list, e_budget,
            window_end - window_start, dtype,
        )

        if tau_local >= window_end - window_start:
            # No event in the remaining window.
            effective_horizon = window_end if window_start == 0.0 else window_end
            stats = _make_stats(
                math.inf, proposals, rate_evals, max_ratio, bound_violations,
                rejected_in_window, violated, effective_horizon,
                curvature_ratio=curvature_ratio,
            )
            return (math.inf, stats) if diagnostics else math.inf

        tau_global = window_start + tau_local

        proposals += 1
        rate_evals += 1
        lam_true = max(rate_scalar_fn(tau_global), 0.0)

        # Lambda at tau_local within its segment: locate the segment INDEPENDENTLY
        tau_local_q = _round_query(tau_local, dtype)
        seg_idx = bisect.bisect_right(knot_times_list, tau_local_q) - 1
        seg_idx = max(0, min(seg_idx, len(seg_bounds_list) - 1))
        lam_bound = max(seg_bounds_list[seg_idx], 0.0)

        ratio = lam_true / lam_bound if lam_bound > 1e-14 else 0.0

        # Curvature-only ratio, isolating the target-curvature-driven part
        # of the bound from a coordinate-uniform additive offset (e.g. ZigZag's D*gamma refreshment floor)
        curvature_denom = lam_bound - rate_offset
        if curvature_denom > 1e-10:
            curvature_ratio = max(curvature_ratio, (lam_true - rate_offset) / curvature_denom)

        if ratio > 1.0:
            bound_violations += 1
            violated = True
            max_ratio = max(max_ratio, ratio)

            remaining_width = (window_end - window_start) - tau_local
            new_width = remaining_width / 2.0

            if bound_violations >= max_violations or new_width < min_window:
                # Give up subdividing
                stats = _make_stats(
                    math.inf, proposals, rate_evals, max_ratio, bound_violations,
                    rejected_in_window, violated, tau_global,
                    curvature_ratio=curvature_ratio,
                )
                return (math.inf, stats) if diagnostics else math.inf

            # Rebuild over the shrunk remainder [tau_global, tau_global + new_width], NOT over [0, new_width].
            # t_offset=window_start (== tau_global, the new window_start) 
            window_start = tau_global
            window_end = tau_global + new_width
            knot_times, seg_bounds, cum_bound, n_evals = build_fn(
                rate_and_grad_fn, window_end - window_start, n_segments, device, dtype, eps,
                t_offset=window_start,
            )
            rate_evals += n_evals
            knot_times_list = knot_times.tolist()
            seg_bounds_list = seg_bounds.tolist()
            cum_bound_list = cum_bound.tolist()
            # Fresh Exp(1) budget for the shrunk window 
            e_budget = 0.0
            continue

        max_ratio = max(max_ratio, ratio)

        if torch.rand(()).item() < ratio:
            stats = _make_stats(
                tau_global, proposals, rate_evals, max_ratio, bound_violations,
                rejected_in_window, violated, None,
                curvature_ratio=curvature_ratio,
            )
            return (tau_global, stats) if diagnostics else tau_global

        # Rejected, continue drawing from the same window without rebuilding.
        rejected_in_window = True

    # max_iter exhausted without resolving 
    import warnings
    warnings.warn(
        f"grid_thinning: max_iter={max_iter} reached without resolving "
        f"(window=[{window_start:.6g}, {window_end:.6g}]).",
        RuntimeWarning,
    )
    stats = _make_stats(
        math.inf, proposals, rate_evals, max_ratio, bound_violations,
        rejected_in_window, violated, window_end,
        curvature_ratio=curvature_ratio,
    )
    return (math.inf, stats) if diagnostics else math.inf
