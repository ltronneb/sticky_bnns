"""Helpers for notebooks/uci_skeletons.ipynb.

The skeletons are too large to load (up to 22 GB), so every file is read
memory-mapped in chunks of rows and reduced to a compact summary, cached as a
.npz next to the skeletons. A skeleton row is one event, (x, v, t). A frozen
coordinate is stored as x = v = 0, so freezes and thaws show up as changes in
the number of frozen coordinates between consecutive rows.
"""

import math
from pathlib import Path

import numpy as np
import torch

RESULTS = Path("results/paper_v2")
SKELETONS = RESULTS / "skeletons"
CACHE = SKELETONS / "_summaries"

# network folder of the skeletons -> result folder of the published run, per PIW
RESULT_DIR = {
    ("shallow", 0.3): RESULTS / "shallow",
    ("shallow", 0.1): RESULTS / "shallow_piw_0.1",
    ("deep_narrow", 0.3): RESULTS / "deep_narrow",
    ("deep_narrow", 0.1): RESULTS / "deep_narrow" / "piw_0.1",
}


def discover(dataset: str = "boston", split: int = 0) -> list[dict]:
    """Every saved skeleton with the path of its published result file."""
    out = []
    for net in ("shallow", "deep_narrow"):
        for w in (0.3, 0.1):
            d = SKELETONS / net / f"piw_{w:g}" / dataset / f"split_{split:02d}"
            for f in sorted(d.glob("*_skeleton.pt")):
                sampler = f.name.removesuffix("_skeleton.pt")
                res = RESULT_DIR[net, w] / dataset / f"split_{split:02d}" / f"{sampler}.pt"
                out.append({"network": net, "w": w, "sampler": sampler, "skeleton": f, "result": res})
    return out


def _weight_mask(layer_sizes) -> torch.Tensor:
    """True on weight-matrix coordinates of the flat [W0, b0, W1, b1, ..., log sigma] vector."""
    m = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        m += [True] * (n_in * n_out) + [False] * n_out
    return torch.tensor(m + [False])


@torch.no_grad()
def summarize(entry: dict, burnin: float = 0.2, chunk: int = 10_000, use_cache: bool = True) -> dict:
    """Per-row and per-coordinate summary of one skeleton, plus its run's metadata.

    Per row (event): time, number of frozen coordinates, log sigma.
    Per weight: fraction of the post-burn-in trajectory TIME it is frozen (exact,
    a coordinate is frozen on an interval iff it is frozen at its left end),
    and the fraction of the saved draws in which it is exactly zero."""
    src = entry["skeleton"]
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{entry['network']}_piw{entry['w']:g}_{entry['sampler']}.npz"
    stamp = np.array([src.stat().st_size, int(src.stat().st_mtime)])
    if use_cache and cache.exists():
        c = np.load(cache, allow_pickle=True)
        if np.array_equal(c["stamp"], stamp):
            return {**entry, **{k: c[k] for k in c.files}, **c["meta"].item()}

    sk = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    P, V, T = sk["positions"], sk["velocities"], sk["times"].double()
    n, D = P.shape
    run = torch.load(entry["result"], map_location="cpu", weights_only=False, mmap=True)
    layers = run["layer_sizes"]
    wmask = _weight_mask(layers)
    assert wmask.numel() == D, (wmask.numel(), D)

    t0, t_end = float(T[0]), float(T[-1])
    t_cut = t0 + burnin * (t_end - t0)
    overlap = (T[1:].clamp(min=t_cut) - T[:-1].clamp(min=t_cut))          # interval time after burn-in

    n_frozen = np.empty(n, np.int32)
    log_sigma = np.empty(n, np.float32)
    frozen_time = torch.zeros(D, dtype=torch.float64)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        p, v = P[a:b], V[a:b]
        fr = (p == 0) & (v == 0)
        n_frozen[a:b] = fr.sum(1).numpy()
        log_sigma[a:b] = p[:, -1].float().numpy()
        m = min(b, n - 1)                                                  # rows that start an interval
        if m > a:
            frozen_time += overlap[a:m] @ fr[: m - a].double()
    frozen_time_frac = (frozen_time / (t_end - t_cut))[wmask].numpy()

    samples = run["samples"]
    zero_draw_frac = (samples[:, wmask] == 0).double().mean(0).numpy()

    d = np.diff(n_frozen)
    counts = {"freeze": int((d > 0).sum()), "thaw": int((d < 0).sum()), "velocity": int((d == 0).sum())}
    tmax_log = np.asarray(run.get("grid_t_max_log") or [], dtype=np.float64)
    meta = {
        "n_rows": n, "D": D, "n_weights": int(wmask.sum()), "layer_sizes": list(layers),
        "t0": t0, "t_end": t_end, "t_cut": t_cut,
        "grads": int(run.get("gradient_evals") or run.get("grad_evals") or 0),
        "violations": int(run.get("bound_violations") or 0),
        "rejections": run.get("rejections"),
        "iterations": len(tmax_log) or None,
        "t_max_final": float(tmax_log[-1]) if len(tmax_log) else run.get("t_max_final"),
        "run_counts": {k: run[k] for k in ("bounce", "freeze", "thaw", "refresh") if k in run},
        "skel_counts": counts,
        "y_std": float(run["y_std"]),
    }
    out = {"times": T.numpy(), "n_frozen": n_frozen, "log_sigma": log_sigma,
           "frozen_time_frac": frozen_time_frac, "zero_draw_frac": zero_draw_frac,
           "tmax_log": tmax_log}
    np.savez(cache, stamp=stamp, meta=np.array(meta, dtype=object), **out)
    return {**entry, **out, **meta}


@torch.no_grad()
def path_on_grid(entry: dict, n_grid: int = 4000, burnin: float = 0.2) -> tuple[np.ndarray, torch.Tensor]:
    """The trajectory evaluated at n_grid EVENLY spaced times after burn-in.
    Returns (times, states [n_grid, D]). ZigZag moves in straight lines,
    Boomerang on arcs around x_ref, frozen coordinates stay at exactly 0."""
    sk = torch.load(entry["skeleton"], map_location="cpu", weights_only=False, mmap=True)
    P, V, T = sk["positions"], sk["velocities"], sk["times"].double()
    t0, t_end = float(T[0]), float(T[-1])
    tg = torch.linspace(t0 + burnin * (t_end - t0), t_end, n_grid, dtype=torch.float64)
    idx = (torch.searchsorted(T, tg, right=True) - 1).clamp(0, T.shape[0] - 2)
    x, v = P[idx].double(), V[idx].double()
    dt = (tg - T[idx]).unsqueeze(-1)
    if "boomerang" in entry["sampler"]:
        xr = sk["x_ref"].double()
        xt = xr + (x - xr) * torch.cos(dt) + v * torch.sin(dt)
    else:
        xt = x + v * dt
    xt = torch.where((x == 0) & (v == 0), torch.zeros_like(xt), xt)
    return tg.numpy(), xt


# ---------------------------------------------------------------- ZigZag rate along the path
def energy_grad(layer_sizes, data: dict, prior_sigma_scale: float = 0.3):
    """grad U of the small and medium UCI networks, built as in uci_bnn_grid.build_target
    (fan-in prior with weight and bias std 1, learned noise), without the MAP fit."""
    from sazz.gpu_friendly.models.neural_networks import FFN
    from sazz.gpu_friendly.models.model import BayesianModule
    from sazz.gpu_friendly.models.priors import build_fan_in_prior_precision
    module = FFN(list(layer_sizes), "tanh")
    prec = build_fan_in_prior_precision(module, 1.0, 1.0, True, dtype=torch.float64, device="cpu")
    bm = BayesianModule.build(module, likelihood="gaussian", X=data["X_train"], y=data["y_train"],
                              prior_precision=prec, prior_sigma_scale=prior_sigma_scale,
                              dtype=torch.float64, device="cpu")
    return torch.func.grad(bm.energy)


def zigzag_rate_terms(grad, x: torch.Tensor, v: torch.Tensor, ts: torch.Tensor):
    """Per-coordinate signed rates v_i dU/dx_i at x + v t and their time derivatives,
    both [len(ts), D]. Frozen coordinates have v_i = 0 and so contribute 0."""
    f = lambda t: grad(x + v * t)
    g, dg = torch.func.vmap(lambda t: torch.func.jvp(f, (t,), (torch.ones_like(t),)))(ts)
    return g * v, dg * v


def tangent_bound_sum(y: torch.Tensor, d: torch.Tensor, h: float) -> float:
    """ZigZag single-segment bound on [0, h] before offset and inflation. y, d are [2, D]
    at the two ends, each coordinate is bounded by its tangent lines, clamped at 0 and
    summed (as segment_bound_vectorized in sazz/gpu_friendly/utils/single_segment_bound.py)."""
    y0, y1, d0, d1 = y[0], y[1], d[0], d[1]
    den = d0 - d1
    scale = torch.maximum(torch.maximum(d0.abs(), d1.abs()), torch.full_like(den, 1e-30))
    deg = den.abs() < 1e-8 * scale
    xs = (y1 - y0 - d1 * h) / torch.where(deg, torch.ones_like(den), den)
    xs = torch.where(deg, torch.zeros_like(xs), xs).clamp(0.0, h)
    m = d0 * xs + y0
    m = torch.where(torch.isfinite(m), m, torch.minimum(y0, y1))
    return float(torch.maximum(torch.maximum(y0, y1), m).clamp(min=0.0).sum())


def rebuild_windows(P, V, T, k0: int, n_windows: int, h: float, can_freeze: torch.Tensor,
                    tol: float = 1e-7) -> list[tuple]:
    """The thinning windows of a ZigZag skeleton from row k0 on, for a fixed t_max h.
    A window starts at every row and after every window without an event, and its
    horizon is min(h, next hitting time of an unfrozen freezable weight, next thaw).
    Thaw times are read from the skeleton (the first later row where a frozen weight
    moves). t_max really drifts by a factor alpha per window, so the boundaries are
    approximate. Returns [(row, start, end, kind)], kind is "empty", "bounce" or
    "freeze/thaw" (the event that ends the window)."""
    out, k = [], k0
    while len(out) < n_windows and k + 1 < len(T):
        x, v, t0, t1 = P[k].double(), V[k].double(), float(T[k]), float(T[k + 1])
        frozen = (x == 0) & (v == 0) & can_freeze
        t_thaw = math.inf
        if frozen.any():
            idx = frozen.nonzero()[:, 0]
            for j in range(k + 1, min(k + 50_000, len(T))):
                if (V[j, idx] != 0).any():
                    t_thaw = float(T[j])
                    break
        s = t0
        while s < t1 - tol and len(out) < n_windows:
            xs = x + v * (s - t0)
            hit = torch.where(can_freeze & ~frozen & (xs * v < 0) & (xs.abs() > 0), -xs / v,
                              torch.full_like(xs, math.inf))
            end = s + min(h, float(hit.min()), t_thaw - s)
            if end >= t1 - tol:
                moving = lambda row: int((V[row] != 0).sum())
                out.append((k, s, max(end, t1), "bounce" if moving(k + 1) == moving(k) else "freeze/thaw"))
                break
            out.append((k, s, end, "empty"))
            s = end
        k += 1
    return out
