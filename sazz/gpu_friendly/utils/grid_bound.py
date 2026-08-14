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
"""

import math
from typing import Callable, Optional

import torch
from torch import Tensor


def _make_stats(tau: float, proposals: int, rate_evals: int, max_ratio: float,
                 bound_violations: int, rejected_in_window: bool,
                 violated: bool, effective_horizon: Optional[float]) -> dict:
    return {
        "accepted": tau < math.inf,
        "rejected": tau == math.inf,
        "proposals": proposals,
        "rate_evals": rate_evals,
        "max_ratio": max_ratio,
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
):
    """
    Algorithm 2: piecewise-constant upper bound of signed rate g(t) over
    [0, horizon], from tangent lines at n_segments+1 grid nodes.

    eps: tolerance for the d_i == d_{i+1} degenerate case in Eq. 2's
    tangent-intersection formula. The paper uses exact equality, which
    near-never triggers in float and lets near-equal slopes blow the
    intersection up (~1e8 for a 1e-9 slope gap); default 1e-4 (float32-sized,
    since GPU use implies float32 is the common working dtype here).

    Returns knot_times[n_segments+1], seg_bounds[n_segments] (unclamped
    Lambda_i per segment), cum_bound[n_segments+1] (cumulative integral of
    max(Lambda, 0), cum_bound[0] == 0), rate_evals (== n_segments+1).
    """
    if eps is None:
        eps = 1e-4

    t = torch.linspace(0.0, horizon, n_segments + 1, device=device, dtype=dtype)
    y, d = rate_and_grad_fn(t)

    t0, t1 = t[:-1], t[1:]
    y0, y1 = y[:-1], y[1:]
    d0, d1 = d[:-1], d[1:]

    denom = d0 - d1
    degenerate = denom.abs() < eps

    # Safe denom to avoid NaN/Inf propagating through the non-degenerate
    # branch's arithmetic even where we'll discard the result via `where`.
    safe_denom = torch.where(degenerate, torch.ones_like(denom), denom)
    x_i_raw = (y1 - y0 + d0 * t0 - d1 * t1) / safe_denom
    m_i_raw = d0 * x_i_raw + y0 - d0 * t0

    # Degenerate fallback (Eq. 2's d_i == d_{i+1} case): x_i = t_i, m_i = y_i.
    x_i = torch.where(degenerate, t0, x_i_raw)
    m_i = torch.where(degenerate, y0, m_i_raw)

    # Guard against non-finite m_i even outside the degenerate branch
    # (can occur if denom is small but not below eps) -- clip x_i to the
    # segment BEFORE trusting m_i, and if m_i is still non-finite, fall
    # back to the y0/y1 endpoints only.
    x_i = torch.clamp(x_i, min=t0, max=t1)
    m_i = torch.where(torch.isfinite(m_i), m_i, torch.minimum(y0, y1))

    seg_bounds = torch.maximum(torch.maximum(y0, y1), m_i)  # unclamped (design decision #4)

    seg_widths = t1 - t0
    seg_integrals = torch.clamp(seg_bounds, min=0.0) * seg_widths
    cum_bound = torch.cat([
        torch.zeros(1, device=device, dtype=dtype),
        torch.cumsum(seg_integrals, dim=0),
    ])

    return t, seg_bounds, cum_bound, n_segments + 1


def sample_from_grid_bound(
    knot_times: Tensor, seg_bounds: Tensor, cum_bound: Tensor, e: float, horizon: float,
    device: Optional[torch.device] = None,
) -> float:
    """
    Algorithm 3 (corrected): given accumulated Exp(1) budget `e`, find the
    proposed event time under the piecewise-constant bound.

    The paper's pseudocode exits at the first knot t_i with integral(0,t_i)
    > e and uses t_i/Lambda(t_i) directly -- an off-by-one; the segment
    containing the solution starts at t_{i-1}. searchsorted below already
    applies that correction.

    Returns tau, or `horizon` as a "no event in this window" sentinel if
    `e` exceeds the total cumulative bound.
    """
    total = float(cum_bound[-1])
    if e >= total:
        return horizon

    if device is None:
        device = cum_bound.device

    # side="right": first index i such that cum_bound[i] > e. The segment
    # containing e is [i-1, i]; clamp defensively for the e==0 edge case.
    idx = int(torch.searchsorted(cum_bound, torch.tensor(e, dtype=cum_bound.dtype, device=device), side="right"))
    idx = max(1, min(idx, len(seg_bounds)))
    i = idx - 1

    t_i = float(knot_times[i])
    lam_i = float(seg_bounds[i])
    remaining = e - float(cum_bound[i])

    if lam_i <= 1e-14:
        # Zero-height segment: cannot supply any more integral, defer to
        # the next segment's start (mirrors falling through to i+1).
        return float(knot_times[i + 1]) if i + 1 < len(knot_times) else horizon

    # Within-segment rate is constant (Lambda_i is a single height, not a
    # slope) -- solve linearly: remaining = lam_i * (tau - t_i).
    tau = t_i + remaining / lam_i
    return tau


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

    Returns (tau, stats) if diagnostics else tau.
    """
    if device is None:
        device = torch.device("cpu")

    window_start = 0.0
    window_end = horizon
    rate_evals = 0
    proposals = 0
    max_ratio = 0.0
    bound_violations = 0
    rejected_in_window = False
    violated = False

    knot_times, seg_bounds, cum_bound, n_evals = build_grid_bound(
        rate_and_grad_fn, window_end - window_start, n_segments, device, dtype, eps,
    )
    rate_evals += n_evals

    e_budget = 0.0

    for _ in range(max_iter):
        u = torch.rand(()).item()
        e_budget += -math.log(1.0 - u)

        tau_local = sample_from_grid_bound(
            knot_times, seg_bounds, cum_bound, e_budget, window_end - window_start,
            device=device,
        )

        if tau_local >= window_end - window_start:
            # No event in the (possibly shrunk) remaining window.
            effective_horizon = window_end if window_start == 0.0 else window_end
            stats = _make_stats(
                math.inf, proposals, rate_evals, max_ratio, bound_violations,
                rejected_in_window, violated, effective_horizon,
            )
            return (math.inf, stats) if diagnostics else math.inf

        tau_global = window_start + tau_local

        proposals += 1
        rate_evals += 1
        lam_true = max(rate_scalar_fn(tau_global), 0.0)

        # Lambda at tau_local within its segment: locate the segment again
        # (cheap; n_segments is small) rather than threading an index out
        # of sample_from_grid_bound.
        seg_idx = int(torch.searchsorted(knot_times, torch.tensor(tau_local, dtype=knot_times.dtype, device=device), side="right")) - 1
        seg_idx = max(0, min(seg_idx, len(seg_bounds) - 1))
        lam_bound = max(float(seg_bounds[seg_idx]), 0.0)

        ratio = lam_true / lam_bound if lam_bound > 1e-14 else 0.0

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
                )
                return (math.inf, stats) if diagnostics else math.inf

            # Rebuild over the shrunk remainder [tau_global, tau_global + new_width].
            window_start = tau_global
            window_end = tau_global + new_width
            knot_times, seg_bounds, cum_bound, n_evals = build_grid_bound(
                rate_and_grad_fn, window_end - window_start, n_segments, device, dtype, eps,
            )
            rate_evals += n_evals
            # Fresh Exp(1) budget for the shrunk window -- valid by the
            # memorylessness of the exponential distribution.
            e_budget = 0.0
            continue

        max_ratio = max(max_ratio, ratio)

        if torch.rand(()).item() < ratio:
            stats = _make_stats(
                tau_global, proposals, rate_evals, max_ratio, bound_violations,
                rejected_in_window, violated, None,
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
    )
    return (math.inf, stats) if diagnostics else math.inf
