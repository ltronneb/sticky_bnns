"""
Grid-based piecewise-constant upper bound for Poisson thinning (Andral &
Kamatani 2024, Algorithms 2/3; horizon adaptation, Algorithm 4, is the
caller's job -- see grid_boomerang.py's `_grid_bound`).

Batched/GPU-shaped: `build_grid_bound` takes one vectorized
`rate_and_grad_fn` closure, no Python looping over grid points. Only
`grid_thinning`'s accept/reject loop is inherently sequential.

`rate_and_grad_fn(t: Tensor[K]) -> (y, d)` must be forward-mode (jvp), not
reverse-over-reverse, e.g.:
    def rate_and_grad_fn(t_batch):
        return torch.func.vmap(
            lambda ti: torch.func.jvp(g_scalar, (ti,), (torch.ones_like(ti),))
        )(t_batch)
Forward-mode is exact and traces as one batched call; ~2x a forward pass.

Single-window contract: every function here handles exactly one window
[0, horizon] -- no internal multi-window loop, so a truncating safety cap
can never let a caller believe time was examined that wasn't. Horizon
adaptation across calls is the caller's job.

`stats["effective_horizon"]`:
- ACCEPT (tau < horizon): None. Swept interval is [0, tau]; check
  `stats["tau"] < horizon`, don't treat None as "full horizon swept".
- NO EVENT, full window examined: == horizon.
- Bound-violation early exit: the LAST VIOLATION's proposal time, not the
  original horizon or the partial sub-horizon reached -- only [0, that
  time] was examined against a bound not already known to be wrong.

Bound violations (ratio > 1) are a validity signal, not a tuning knob:
samples drawn in that window used a bound that wasn't actually valid
there. Shrinking the horizon prevents future violations, not past ones --
`stats["bound_violations"] > 0` should always be surfaced, not just logged.

---

fast_grid_bound.py -- a copy of grid_bound.py that rewrites
sample_from_grid_bound and grid_thinning's per-proposal scalar reads to
operate on plain Python lists instead of re-touching knot_times/seg_bounds/
cum_bound (small, ~n_segments+1 element tensors) once per proposal. Those
three tensors are built once per window (build_grid_bound[_vectorized],
unchanged below) and never mutated between builds, so every downstream
scalar read against them (float(cum_bound[-1]), searchsorted, indexing)
was pure Python arithmetic once the values were known -- there was never a
reason to keep touching a tensor for it after the window build. This takes
roughly 7 host syncs per proposal down to 0, amortized into 3 syncs
(.tolist() x3) per window build instead (far less frequent -- a window
build happens once per grid_thinning() call, or again per violation-driven
rebuild, not once per accept/reject proposal).

Two prior approaches to this same idea were considered and are DELIBERATELY
NOT what's implemented here -- both are correctness traps, not just
missed optimizations:

1. Reusing sample_from_grid_bound's internal segment index for
   grid_thinning's own seg_idx lookup. These are NOT the same search:
   sample_from_grid_bound's zero-height-segment early return (see its
   docstring below) returns a value belonging to the NEXT segment, and
   grid_thinning's seg_idx search must re-locate the segment for THAT
   returned value, independently, by time -- not by re-deriving it from
   the integral-budget search's index. Reusing the index would silently
   force lam_bound ~= 0 (the zero-height segment being stepped past),
   forcing a guaranteed rejection on every proposal that crosses one.
   This file's rewrite keeps both searches, independently, exactly as the
   original -- just against Python lists instead of tensors.

2. Batching the torch.rand(()) draws (grid_thinning's Exp(1) budget draw
   and its accept/reject draw). Both calls omit device=, and this whole
   codebase never calls torch.set_default_device (confirmed by grep) --
   so these are already CPU tensors, and .item() on a CPU scalar tensor
   is Python-level overhead, not a device sync. There was nothing here
   for batching to save; not implemented.

PRECISION NOTE, the reason the rewrite below is bit-identical to the
original and not just "close": the original's sample_from_grid_bound
(line ~equivalent below) and grid_thinning's seg_idx search do NOT compare
the query (e / tau_local, both Python doubles) at double precision --
`torch.tensor(e, dtype=cum_bound.dtype, ...)` rounds the query DOWN to
cum_bound's working dtype (float32 on the GPU path this whole gpu_friendly
tree targets: DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64,
see scripts/mnist_cnn_grid.py) before the comparison. A naive .tolist()
rewrite that compares a full-double-precision query against the (still
float32-exact) stored values would silently disagree with the original by
one index whenever a query sits within half a float32 ULP of a knot. The
fix, applied at every list-search site below: round the query through
`dtype` via a CPU-only `float(torch.tensor(q, dtype=dtype))` cast BEFORE
the bisect call -- exactly mirroring the original's implicit cast, at
essentially zero cost (no device sync; a CPU-side float32 round-trip).
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
):
    """
    Algorithm 2: piecewise-constant upper bound of signed rate g(t) over
    [0, horizon], from tangent lines at n_segments+1 grid nodes.

    eps: tolerance for the d_i == d_{i+1} degenerate case in Eq. 2's
    tangent-intersection formula. The paper uses exact equality, which
    near-never triggers in float and lets near-equal slopes blow the
    intersection up (~1e8 for a 1e-9 slope gap); default 1e-4 (float32-sized,
    since GPU use implies float32 is the common working dtype here).

    t_offset: shifts ONLY the times rate_and_grad_fn is evaluated at
    (t_offset .. t_offset + horizon) -- the RETURNED knot_times stay local
    ([0, horizon]) so every downstream consumer (sample_from_grid_bound,
    grid_thinning's tau_local/tau_global bookkeeping) is unchanged. Needed
    because grid_thinning's Section 4.7 rebuild anchors a fresh window at a
    global time (window_start) but must still evaluate the caller's closure
    at that global time, not at a re-zeroed local time -- default 0.0 is a
    no-op, preserving the original (pre-fix) behavior when omitted.

    Returns knot_times[n_segments+1], seg_bounds[n_segments] (unclamped
    Lambda_i per segment), cum_bound[n_segments+1] (cumulative integral of
    max(Lambda, 0), cum_bound[0] == 0), rate_evals (== n_segments+1).
    """
    if eps is None:
        eps = 1e-4

    # t_local is what's returned as knot_times and used for all downstream
    # width/clamp arithmetic; t_eval (shifted by t_offset) is ONLY used to
    # evaluate rate_and_grad_fn, so the caller's window bookkeeping (which
    # reasons entirely in the local [0, horizon] frame) never has to know
    # about the shift.
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

    seg_bounds = torch.maximum(torch.maximum(y0, y1), m_i)  # unclamped (design decision #4)

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
):
    """
    Per-coordinate Algorithm 2 (Andral & Kamatani Section 4.4.2), batched
    over BOTH grid segments (build_grid_bound's existing elementwise
    vectorization) AND an added leading coordinate axis D, summed into one
    scalar piecewise-constant bound -- consumed by the EXISTING, unmodified
    sample_from_grid_bound/grid_thinning exactly as build_grid_bound's
    scalar output already is. Ports the Boomerang/BPS-shaped scalar bound
    to ZigZag's rate, which is a SUM of D independently-kinked per-
    coordinate terms rather than one smooth signed inner product -- naively
    bounding the aggregate signed sum fails (coordinates can cancel and
    hide curvature), so each coordinate gets its own tangent-line bound,
    summed after an optional per-coordinate clamp (see `signed` below).

    rate_and_grad_fn(t_batch: Tensor[K]) -> (y_full, d_full): Tensor[D, K]
    each -- the per-coordinate rate and its time-derivative, at every grid
    time simultaneously. One call covers all D coordinates (a single
    torch.func.vmap(jvp(...)) composition sharing one gradient evaluation
    per grid time across every coordinate -- NOT a per-coordinate loop), so
    rate_evals == n_segments + 1, not D * (n_segments + 1).

    signed: True (default, "vectorized_signed") -- caller feeds the raw
    SIGNED per-coordinate rate/derivative; each coordinate's segment bound
    is clamped to >= 0 individually, BEFORE summing over D. Smooth (no
    kink), so the tangent-line construction is robust, at the cost of a
    looser bound. False ("vectorized_not_signed") -- caller has already
    clamped the per-coordinate rate (and zeroed its derivative where the
    rate is negative) before calling, so no additional per-coordinate clamp
    is applied here, only the existing defensive final clamp(min=0) before
    forming the integral. This variant is tighter but can silently
    under-bound: if a coordinate's signed rate is negative at both grid
    knots but crosses positive strictly inside the segment, the clamped
    input has y0=y1=0, d0=d1=0 at the knots, giving a segment bound of 0 --
    the missed positive mass is undetectable by grid_thinning's ratio
    check, since no proposal is ever generated there to trigger it. Use for
    reproducing Andral & Kamatani's Figure 6 comparison, not as an
    unattended production default.

    offset: added to the SCALAR seg_bounds (after summing over D) BEFORE
    seg_widths/seg_integrals/cum_bound are formed -- e.g. ZigZag's D*gamma
    refreshment floor. Must be added HERE, inside this function, not by the
    caller after grid_thinning returns: seg_bounds/cum_bound are built and
    fully consumed inside grid_thinning (including any Section 4.7
    rebuild), so a post-hoc addition would never reach the integral
    actually used for sampling, while the true-rate accept/reject check
    (which does include the offset) would then be checked against a bound
    that's missing it -- producing a ratio > 1 on essentially every
    proposal. Because `offset` is coordinate-uniform and conceptually
    added AFTER each coordinate's own clamp (matching the true rate's
    clamp-then-add-gamma convention), "clamp each coordinate then add
    offset then sum" and "sum the clamped per-coordinate bounds then add
    offset once" coincide -- so a single post-sum add is exact, not an
    approximation.

    t_offset: shifts ONLY the times rate_and_grad_fn is evaluated at
    (t_offset .. t_offset + horizon); returned knot_times stay LOCAL
    ([0, horizon]) -- identical contract to build_grid_bound's own
    t_offset (see its docstring), needed for a correct Section 4.7 rebuild.

    eps: RELATIVE tolerance here (unlike build_grid_bound's ABSOLUTE
    tolerance in the same-named slot) -- if None, defaults to 1e-8, NOT
    build_grid_bound's 1e-4. Per-coordinate slopes d_j = v_j*(Hessian @ v)_j
    are typically ~1/D of the aggregate rate scale build_grid_bound's
    absolute default was calibrated for, so reusing 1e-4 here would falsely
    flag a large fraction of genuinely-distinct coordinate slopes as
    degenerate on a wide target, silently under-bounding via the m_i=y0
    fallback. The degeneracy test itself is also RELATIVE with no unit
    floor: |d0-d1| < eps * max(|d0|, |d1|, 1e-30) -- a floor of 1.0 (as in
    an earlier draft) would collapse the threshold back to eps*1.0 at
    small-magnitude slopes, silently reproducing the absolute-scale
    failure this relative test exists to avoid. The 1e-30 floor only
    guards a literal 0/0 in the threshold itself; it is not a meaningful
    physical scale (the same safe_denom construction below already
    prevents any actual division by a degenerate denom).

    Returns knot_times[n_segments+1] (local), seg_bounds[n_segments]
    (scalar, summed over D, offset already added), cum_bound[n_segments+1],
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

    seg_bounds = seg_bounds_percoord.sum(dim=0) + offset  # [K-1] scalar per segment

    seg_widths = t1 - t0
    seg_integrals = torch.clamp(seg_bounds, min=0.0) * seg_widths
    cum_bound = torch.cat([
        torch.zeros(1, device=device, dtype=dtype),
        torch.cumsum(seg_integrals, dim=0),
    ])

    return t_local, seg_bounds, cum_bound, n_segments + 1


def _round_query(q: float, dtype: torch.dtype) -> float:
    """
    Rounds a Python double through `dtype` (a CPU-only, no-device-sync
    round-trip) before a list search -- reproduces the original's implicit
    cast (torch.tensor(q, dtype=cum_bound.dtype/knot_times.dtype, ...))
    ahead of its torch.searchsorted call, so the .tolist()-backed list
    search below compares at the SAME precision the original tensor-backed
    search did (float32 on the GPU path this whole tree targets), not at
    full Python-double precision. See module docstring's PRECISION NOTE.
    """
    return float(torch.tensor(q, dtype=dtype))


def _sample_from_grid_bound_list(
    knot_times_list: list[float], seg_bounds_list: list[float], cum_bound_list: list[float],
    e: float, horizon: float, dtype: torch.dtype,
) -> tuple[float, Optional[int]]:
    """
    Algorithm 3 (corrected): given accumulated Exp(1) budget `e`, find the
    proposed event time under the piecewise-constant bound.

    The paper's pseudocode exits at the first knot t_i with integral(0,t_i)
    > e and uses t_i/Lambda(t_i) directly -- an off-by-one; the segment
    containing the solution starts at t_{i-1}. bisect.bisect_right below
    already applies that correction (list analog of the original's
    torch.searchsorted(..., side="right")).

    Returns (tau, seg_idx) where seg_idx is the segment index the returned
    tau falls in -- `None` on the two "no further event this window"
    returns (total-exceeded, or the horizon fallback of a trailing
    zero-height segment), matching the tau==horizon sentinel's "nothing to
    look up" semantics. Callers that need seg_idx (grid_thinning) resolve
    it with their OWN independent search over the returned tau -- see
    module docstring point 1 for why this must not be conflated with the
    zero-height-segment's `i`.

    Precision: matches the original EXACTLY, not just approximately --
    the original rounds the query to `dtype` ONLY for the
    torch.searchsorted comparison (torch.tensor(e, dtype=cum_bound.dtype)),
    while `e >= total` and `remaining = e - cum_bound[i]` both use the
    FULL-PRECISION `e` unrounded. This function reproduces that exactly:
    `e` itself is never touched; only the bisect query is rounded.
    """
    total = cum_bound_list[-1]
    if e >= total:
        return horizon, None

    # side="right": first index i such that cum_bound[i] > e. The segment
    # containing e is [i-1, i]; clamp defensively for the e==0 edge case.
    # Query rounded to `dtype` for this comparison ONLY -- see docstring.
    e_q = _round_query(e, dtype)
    idx = bisect.bisect_right(cum_bound_list, e_q)
    idx = max(1, min(idx, len(seg_bounds_list)))
    i = idx - 1

    t_i = knot_times_list[i]
    lam_i = seg_bounds_list[i]
    remaining = e - cum_bound_list[i]

    if lam_i <= 1e-14:
        # Zero-height segment: cannot supply any more integral, defer to
        # the next segment's start (mirrors falling through to i+1). This
        # tau belongs to segment i+1, NOT i -- see module docstring point
        # 1 for why a caller must never reuse `i` here as this tau's
        # segment index.
        if i + 1 < len(knot_times_list):
            return knot_times_list[i + 1], None
        return horizon, None

    # Within-segment rate is constant (Lambda_i is a single height, not a
    # slope) -- solve linearly: remaining = lam_i * (tau - t_i).
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

    rate_scalar_fn: single-point signed rate, for the accept/reject check
    (cheaper than batching one point through vmap/jvp).
    min_window: stop subdividing and return early if a violation-driven
    shrink would take the remaining sub-window below this width.
    max_violations: cap on violations handled per call before returning.

    bound_fn: the bound-construction callable to use in place of
    build_grid_bound, e.g. a functools.partial(build_grid_bound_vectorized,
    signed=..., offset=...) for a ZigZag-shaped per-coordinate rate.
    Defaults to None, in which case build_grid_bound is called exactly as
    before -- byte-identical default behavior for any existing caller
    (e.g. the Boomerang sampler). Must accept a `t_offset` KEYWORD
    argument, since it is always passed as one (never positionally) at
    both call sites below -- see build_grid_bound's own `t_offset` for the
    contract this satisfies (evaluate the closure at globally-shifted
    times while returning locally-indexed knot_times/seg_bounds/cum_bound).

    rate_offset: the SAME additive constant the active bound_fn folds into
    its bound (e.g. ZigZag's D*gamma refreshment floor), used here ONLY to
    report a curvature-only accept-check ratio, `stats["curvature_ratio"]
    = (lam_true - rate_offset) / (lam_bound - rate_offset)`, alongside the
    existing `max_ratio`. This must be computed HERE (not re-derived by
    the caller from returned stats), since `lam_true`/`lam_bound` are local
    to this loop and never otherwise escape. Default 0.0 makes
    curvature_ratio degenerate into max_ratio exactly (harmless for
    Boomerang, which passes neither bound_fn nor rate_offset). Guarded
    against a near-zero `lam_bound - rate_offset` denominator (e.g. a
    segment where all curvature-driven bound slack is consumed and only
    the offset remains) -- left at its previous value rather than dividing.

    Returns (tau, stats) if diagnostics else tau.

    Per-proposal reads against knot_times/seg_bounds/cum_bound operate on
    plain Python lists (materialized via one .tolist() each, immediately
    after every build_fn call -- 3 syncs per window build, not per
    proposal) rather than re-touching the underlying tensors on every
    accept/reject iteration -- see module docstring for the full rationale
    and the precision-matching fix (_round_query) this rewrite depends on
    for bit-identical output.
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
        # CPU tensor (no device= given -- see module docstring point 2):
        # .item() here is Python-level overhead, not a host sync, so this
        # stays untouched from the original.
        u = torch.rand(()).item()
        e_budget += -math.log(1.0 - u)

        tau_local, _unused_seg_idx = _sample_from_grid_bound_list(
            knot_times_list, seg_bounds_list, cum_bound_list, e_budget,
            window_end - window_start, dtype,
        )

        if tau_local >= window_end - window_start:
            # No event in the (possibly shrunk) remaining window.
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

        # Lambda at tau_local within its segment: locate the segment
        # INDEPENDENTLY, by time, exactly as the original did -- this is
        # NOT the same search as the integral-budget search above (see
        # module docstring point 1: the zero-height-segment early return
        # makes them genuinely different lookups, not a redundant pair).
        tau_local_q = _round_query(tau_local, dtype)
        seg_idx = bisect.bisect_right(knot_times_list, tau_local_q) - 1
        seg_idx = max(0, min(seg_idx, len(seg_bounds_list) - 1))
        lam_bound = max(seg_bounds_list[seg_idx], 0.0)

        ratio = lam_true / lam_bound if lam_bound > 1e-14 else 0.0

        # Curvature-only ratio, isolating the target-curvature-driven part
        # of the bound from a coordinate-uniform additive offset (e.g.
        # ZigZag's D*gamma refreshment floor) -- see this function's
        # `rate_offset` docstring. Only defined (and only updates the
        # running value) when the curvature-driven bound headroom is
        # itself non-negligible; otherwise left at its previous value
        # rather than dividing by a near-zero denominator.
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
                # Give up subdividing -- only [0, tau_global] was ever
                # examined against a bound that wasn't already known to
                # be wrong. Report that, not the original horizon.
                stats = _make_stats(
                    math.inf, proposals, rate_evals, max_ratio, bound_violations,
                    rejected_in_window, violated, tau_global,
                    curvature_ratio=curvature_ratio,
                )
                return (math.inf, stats) if diagnostics else math.inf

            # Rebuild over the shrunk remainder [tau_global, tau_global + new_width].
            # t_offset=window_start (== tau_global, the new window_start) is
            # REQUIRED here, not cosmetic: rate_and_grad_fn/rate_scalar_fn
            # are anchored at this call's original (x, v), so without
            # shifting the evaluation grid to the new window's GLOBAL start,
            # the rebuilt bound would silently be evaluated over
            # [0, new_width] (measured from the very start of this whole
            # grid_thinning call) instead of the window it's actually meant
            # to cover, [tau_global, tau_global + new_width].
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
            # Fresh Exp(1) budget for the shrunk window -- valid by the
            # memorylessness of the exponential distribution.
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

        # Rejected (ordinary thinning rejection, not a bound violation):
        # continue drawing from the same window without rebuilding.
        rejected_in_window = True

    # max_iter exhausted without resolving -- treat as no-event over
    # whatever window was live at the end, being explicit that this is a
    # (rare) fallback distinct from a clean horizon exhaustion.
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
