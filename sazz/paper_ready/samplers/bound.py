"""Poisson thinning with one tangent-line segment per window.

For a signed rate g on [0, h] with values y and slopes d at both ends, the
tangent lines at 0 and h meet at x*, and max(y(0), y(h), tangent at x*) bounds
g on the window whenever g has no inflection point in it (Andral & Kamatani,
2024, Algorithm 2 with a single segment). ZigZag bounds every coordinate
separately and sums the positive parts. The window length t_max adapts. It
grows after an empty window and shrinks after every rejection, so empty
windows and rejections, which cost one gradient each, balance.
"""

import math
import warnings
from typing import Callable, Optional

import torch
from torch import Tensor


def tangent_bound(y: Tensor, d: Tensor, h: float, per_coord: bool, offset: float = 0.0,
                  inflation: float = 0.01) -> float:
    """Upper bound on [0, h] from values y and slopes d at t = 0 and t = h
    (last axis of size 2). With per_coord, y and d are [D, 2] and the
    per-coordinate bounds are clamped at 0 and summed, plus a constant offset."""
    y0, y1, d0, d1 = y[..., 0], y[..., 1], d[..., 0], d[..., 1]
    denom = d0 - d1
    if per_coord:
        scale = torch.maximum(torch.maximum(d0.abs(), d1.abs()), torch.full_like(denom, 1e-30))
        degenerate = denom.abs() < 1e-8 * scale
    else:
        degenerate = denom.abs() < 1e-4
    x = (y1 - y0 - d1 * h) / torch.where(degenerate, torch.ones_like(denom), denom)
    x = torch.where(degenerate, torch.zeros_like(x), x).clamp(0.0, h)
    m = d0 * x + y0
    m = torch.where(torch.isfinite(m), m, torch.minimum(y0, y1))
    bound = torch.maximum(torch.maximum(y0, y1), m)
    if per_coord:
        bound = bound.clamp(min=0.0).sum() + offset
    return float(bound * (1.0 + inflation))


def thin(rate_and_slope: Callable, rate: Callable[[float], float], horizon: float,
         bound: Callable, left: Optional[tuple] = None, max_iter: int = 200,
         max_violations: int = 10, min_window: float = 1e-8):
    """First event on [0, horizon] of a Poisson process with intensity
    max(rate, 0), by thinning a constant bound.

    rate_and_slope(t) -> (y, d) for a tensor of window times t, with the node
    axis last. bound(y, d, h) -> float. left is a cached (y, d) at t = 0.
    Returns (tau, stats), tau = inf when there is no event. On a bound
    violation the remaining window is halved from the violating point.
    """
    s = {"accepted": False, "rate_evals": 0, "violations": 0, "rejections": 0,
         "effective_horizon": horizon, "right": None}
    start, end = 0.0, horizon

    def build(left):
        t = torch.tensor([start, end], dtype=torch.float64)
        if left is None:
            y, d = rate_and_slope(t)
            s["rate_evals"] += 2
        else:
            y1, d1 = rate_and_slope(t[1:])
            s["rate_evals"] += 1
            y = torch.cat([left[0].unsqueeze(-1), y1], -1)
            d = torch.cat([left[1].unsqueeze(-1), d1], -1)
        return y, d, max(bound(y, d, end - start), 0.0)

    y, d, lam = build(left)
    e = 0.0
    for _ in range(max_iter):
        e -= math.log1p(-torch.rand(()).item())
        tau = e / lam if lam > 1e-14 else math.inf
        if tau >= end - start:
            if s["violations"] == 0:
                s["right"] = (y[..., 1], d[..., 1])
            s["effective_horizon"] = end
            return math.inf, s
        t = start + tau
        s["rate_evals"] += 1
        ratio = max(rate(t), 0.0) / lam
        if ratio > 1.0:
            s["violations"] += 1
            width = (end - t) / 2.0
            if s["violations"] >= max_violations or width < min_window:
                s["effective_horizon"] = t
                return math.inf, s
            start, end = t, t + width
            y, d, lam = build(None)
            e = 0.0
            continue
        if torch.rand(()).item() < ratio:
            s["accepted"], s["effective_horizon"] = True, None
            return t, s
        s["rejections"] += 1
    warnings.warn(f"thin: max_iter={max_iter} reached", RuntimeWarning)
    s["effective_horizon"] = end
    return math.inf, s


def adapt_t_max(t_max: float, s: dict, tau: float, horizon: float, t_max_binding: bool,
                alpha: float = 1.01, alpha_violation: float = 2.0) -> float:
    """Grow by alpha after a fully examined empty window set by t_max, shrink
    by alpha per rejection, halve (alpha_violation) after a violation."""
    if s["violations"]:
        return t_max / alpha_violation
    empty = tau == math.inf and t_max_binding and s["effective_horizon"] >= horizon - 1e-12
    return t_max * alpha ** (int(empty) - s["rejections"])
