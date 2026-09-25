"""
Single-segment Poisson thinning: the window [0, horizon] IS the segment.

Motivation: grid_thinning (fast_grid_bound.py) evaluates the rate and its
time-derivative at all n_segments+1 nodes of a window upfront, before it
knows where the event falls. With D >> n_segments the batch over n buys
little parallelism, so every node right of the event is plausibly wasted.
Walking segments of width delta lazily (evaluate segment i only while the
accumulated bound integral is still below the Exp(1) budget) and setting
delta = t_max makes every call a single segment -- the sampler's own outer
loop already supplies "the next segment" after a no-event call, and
Algorithm 4 then adapts the segment width directly.

Equivalent to grid_thinning(n_segments=1), plus one thing grid_thinning
cannot do: reuse the right node. After a clean no-event window the
trajectory continues unchanged, so the next window's left node (t=0) is
exactly this window's right node (t=horizon). Passing it back in via
`left_node` brings an empty window's cost from 2 evaluations to 1.

`stats["right_node"]` is set ONLY when the call ended with no event after
examining the full original window with no violation; otherwise None. The
CALLER decides whether the state actually continues (no bounce, freeze,
thaw, refresh, or minibatch change at the window end) -- a stale node
bounds the wrong function silently.

With left_node=None, random draws happen in the same order and the bound
is built with the same tensor ops as grid_thinning(n_segments=1), so both
return the same tau under the same seed.
"""


import math
import warnings
from typing import Callable, Optional

import torch
from torch import Tensor

from .fast_grid_bound import _make_stats


def segment_bound_scalar(
    t_local: Tensor, y: Tensor, d: Tensor,
    eps: Optional[float] = None, bound_inflation: float = 0.01,
) -> float:
    """
    Algorithm 2's tangent-line bound for ONE segment of a signed scalar
    rate (Boomerang). t_local, y, d: Tensor[2] (left/right node). Same ops
    as build_grid_bound with n_segments=1. Returns the unclamped Lambda.
    """
    if eps is None:
        eps = 1e-4

    t0, t1 = t_local[:-1], t_local[1:]
    y0, y1 = y[:-1], y[1:]
    d0, d1 = d[:-1], d[1:]

    denom = d0 - d1
    degenerate = denom.abs() < eps
    safe_denom = torch.where(degenerate, torch.ones_like(denom), denom)
    x_i_raw = (y1 - y0 + d0 * t0 - d1 * t1) / safe_denom
    x_i = torch.where(degenerate, t0, x_i_raw)
    x_i = torch.clamp(x_i, min=t0, max=t1)

    m_i = d0 * x_i + y0 - d0 * t0
    m_i = torch.where(torch.isfinite(m_i), m_i, torch.minimum(y0, y1))

    seg_bound = torch.maximum(torch.maximum(y0, y1), m_i) * (1.0 + bound_inflation)
    return float(seg_bound[0])


def segment_bound_vectorized(
    t_local: Tensor, y_full: Tensor, d_full: Tensor,
    eps: Optional[float] = None, *,
    signed: bool = True, offset: float = 0.0, bound_inflation: float = 0.01,
) -> float:
    """
    Per-coordinate tangent-line bound for ONE segment of a ZigZag rate,
    summed over D. t_local: Tensor[2]; y_full, d_full: Tensor[D, 2]. Same
    ops as build_grid_bound_vectorized with n_segments=1 (eps RELATIVE,
    default 1e-8; offset added before inflation).
    """
    if eps is None:
        eps = 1e-8

    t0, t1 = t_local[:-1], t_local[1:]
    y0, y1 = y_full[:, :-1], y_full[:, 1:]
    d0, d1 = d_full[:, :-1], d_full[:, 1:]

    denom = d0 - d1
    scale = torch.maximum(torch.maximum(d0.abs(), d1.abs()), torch.full_like(denom, 1e-30))
    degenerate = denom.abs() < eps * scale

    safe_denom = torch.where(degenerate, torch.ones_like(denom), denom)
    x_i_raw = (y1 - y0 + d0 * t0 - d1 * t1) / safe_denom
    x_i = torch.where(degenerate, t0, x_i_raw)
    x_i = torch.clamp(x_i, min=t0, max=t1)

    m_i = d0 * x_i + y0 - d0 * t0
    m_i = torch.where(torch.isfinite(m_i), m_i, torch.minimum(y0, y1))

    seg_bounds_percoord = torch.maximum(torch.maximum(y0, y1), m_i)
    if signed:
        seg_bounds_percoord = torch.clamp(seg_bounds_percoord, min=0.0)

    seg_bound = (seg_bounds_percoord.sum(dim=0) + offset) * (1.0 + bound_inflation)
    return float(seg_bound[0])


def _eval_window(
    rate_and_grad_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
    width: float, t_offset: float, left_node: Optional[tuple[Tensor, Tensor]],
    device: torch.device, dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """
    (t_local[2], y[..., 2], d[..., 2], rate_evals). Evaluates both nodes in
    one call when left_node is None (identical t tensor to
    build_grid_bound's linspace), else only the right node. The node axis
    is the LAST axis for both the scalar ([K]) and vectorized ([D, K]) rate.
    """
    t_local = torch.linspace(0.0, width, 2, device=device, dtype=dtype)
    t_eval = t_local + t_offset if t_offset != 0.0 else t_local

    if left_node is None:
        y, d = rate_and_grad_fn(t_eval)
        return t_local, y, d, 2

    y_r, d_r = rate_and_grad_fn(t_eval[1:])
    y0, d0 = left_node
    y = torch.cat([y0.unsqueeze(-1), y_r], dim=-1)
    d = torch.cat([d0.unsqueeze(-1), d_r], dim=-1)
    return t_local, y, d, 1


def single_segment_thinning(
    rate_and_grad_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
    rate_scalar_fn: Callable[[float], float],
    horizon: float,
    left_node: Optional[tuple[Tensor, Tensor]] = None,
    max_iter: int = 200,
    min_window: float = 1e-8,
    max_violations: int = 10,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    eps: Optional[float] = None,
    diagnostics: bool = True,
    *,
    segment_bound_fn: Optional[Callable] = None,
    rate_offset: float = 0.0,
):
    """
    grid_thinning with the whole window as one segment, plus right-node
    reuse. Same rate_and_grad_fn / rate_scalar_fn / stats contract as
    grid_thinning (see fast_grid_bound.py's module docstring for the
    effective_horizon and violation semantics).

    left_node: (y, d) at t=0 cached from the previous call's
    stats["right_node"], or None to evaluate it here.
    segment_bound_fn: segment_bound_scalar (default, Boomerang) or a
    functools.partial(segment_bound_vectorized, signed=..., offset=...)
    for ZigZag. Called as fn(t_local, y, d, eps).

    Extra stats: "rejections" (count) and "right_node".

    Returns (tau, stats) if diagnostics else tau.
    """
    if device is None:
        device = torch.device("cpu")
    bound_fn = segment_bound_fn if segment_bound_fn is not None else segment_bound_scalar

    window_start = 0.0
    window_end = horizon
    rate_evals = 0
    proposals = 0
    rejections = 0
    max_ratio = 0.0
    curvature_ratio = 0.0
    bound_violations = 0
    rejected_in_window = False
    violated = False

    def finish(tau, effective_horizon, right_node=None):
        stats = _make_stats(
            tau, proposals, rate_evals, max_ratio, bound_violations,
            rejected_in_window, violated, effective_horizon,
            curvature_ratio=curvature_ratio,
        )
        stats["rejections"] = rejections
        stats["right_node"] = right_node
        return (tau, stats) if diagnostics else tau

    t_local, y, d, n_evals = _eval_window(
        rate_and_grad_fn, window_end - window_start, window_start, left_node, device, dtype,
    )
    rate_evals += n_evals
    lam = bound_fn(t_local, y, d, eps)
    width = float(t_local[1])
    total = max(lam, 0.0) * width

    e_budget = 0.0

    for _ in range(max_iter):
        u = torch.rand(()).item()
        e_budget += -math.log(1.0 - u)

        # Exhausted budget, zero-height segment, or tau rounding onto the
        # window end: no event in the remaining window
        tau_local = e_budget / lam if (e_budget < total and lam > 1e-14) else math.inf
        if tau_local >= window_end - window_start:
            right_node = (y[..., 1], d[..., 1]) if bound_violations == 0 else None
            return finish(math.inf, window_end, right_node)

        tau_global = window_start + tau_local

        proposals += 1
        rate_evals += 1
        lam_true = max(rate_scalar_fn(tau_global), 0.0)
        lam_bound = max(lam, 0.0)

        ratio = lam_true / lam_bound if lam_bound > 1e-14 else 0.0

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
                return finish(math.inf, tau_global)

            # Section 4.7: rebuild over [tau_global, tau_global + new_width]
            # with a fresh left node (tau is not a cached point) and budget
            window_start = tau_global
            window_end = tau_global + new_width
            t_local, y, d, n_evals = _eval_window(
                rate_and_grad_fn, window_end - window_start, window_start, None, device, dtype,
            )
            rate_evals += n_evals
            lam = bound_fn(t_local, y, d, eps)
            width = float(t_local[1])
            total = max(lam, 0.0) * width
            e_budget = 0.0
            continue

        max_ratio = max(max_ratio, ratio)

        if torch.rand(()).item() < ratio:
            return finish(tau_global, None)

        rejections += 1
        rejected_in_window = True

    warnings.warn(
        f"single_segment_thinning: max_iter={max_iter} reached without resolving "
        f"(window=[{window_start:.6g}, {window_end:.6g}]).",
        RuntimeWarning,
    )
    return finish(math.inf, window_end)


def adapt_t_max(
    t_max: float, stats: dict, tau: float, horizon: float, t_max_binding: bool,
    rule: str, alpha_plus: float, alpha_minus: float, alpha_violation: float,
) -> float:
    """
    Horizon adaptation after one single_segment_thinning call.

    rule="alg4": Algorithm 4 exactly as the grid samplers apply it (shrink
    by alpha_violation on a violation, by alpha_minus if ANY rejection,
    grow by alpha_plus after a clean, fully examined empty window where
    t_max was the binding horizon candidate).

    rule="balanced": one alpha (alpha_plus) for both directions, log t_max
    += log(alpha) * (1[clean empty window] - n_rejections). Its fixed point
    is E[empty windows] = E[rejections] per call. With the node cache both
    cost one evaluation, and for a cost ~ 1/w + a*w in the window width w
    the optimum is where those two terms are equal. Violations still
    shrink by alpha_violation.
    """
    if stats["violated"]:
        return t_max / alpha_violation

    clean_empty = False
    if tau == math.inf and t_max_binding:
        eff = stats["effective_horizon"]
        clean_empty = eff is not None and eff >= horizon - 1e-12

    if rule == "alg4":
        if stats["rejected_in_window"]:
            return t_max / alpha_minus
        return t_max * alpha_plus if clean_empty else t_max
    if rule == "balanced":
        return t_max * alpha_plus ** (int(clean_empty) - stats["rejections"])
    raise ValueError(f"unknown adapt_rule {rule!r}")
