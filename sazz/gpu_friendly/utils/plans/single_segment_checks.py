"""
Standalone checks for single_segment_bound.py (one tangent-line segment per
t_max window, right node reused as the next window's left node). Synthetic
rates only -- no sampler involved. Run with:

    python -m sazz.gpu_friendly.utils.plans.single_segment_checks
    python -m sazz.gpu_friendly.utils.plans.single_segment_checks --n-reps 20000

1. Same bound: segment_bound_{scalar,vectorized} == build_grid_bound{,_vectorized}
   with n_segments=1.
2. Same draws: single_segment_thinning(left_node=None) returns the same tau
   and stats as grid_thinning(n_segments=1) under the same seed.
3. Cache is correct: right_node matches a fresh evaluation at the next
   window's start.
4. Exact event law: an outer loop (t_max adaptation + node cache) run to the
   first event; Lambda(tau) KS-tested against Exp(1), for both adapt rules.
5. Adaptation study (printed, not asserted): cost per event over a fixed
   t_max sweep (gives the cost-optimal width), where each adapt rule
   settles, and grid_thinning at the samplers' default spacing for reference.
"""

import argparse
import functools
import math
import time

import numpy as np
import torch
from scipy import stats as sps

from sazz.gpu_friendly.utils.fast_grid_bound import (
    build_grid_bound, build_grid_bound_vectorized, grid_thinning,
)
from sazz.gpu_friendly.utils.single_segment_bound import (
    adapt_t_max, segment_bound_scalar, segment_bound_vectorized, single_segment_thinning,
)

DTYPE = torch.float64
GAMMA = 0.01


# ---------------------------------------------------------------------------
# Synthetic targets. closures(s) returns the (rate_and_grad_fn, rate_scalar_fn)
# pair a sampler would build at trajectory time s (both take window-local t).
# ---------------------------------------------------------------------------

class BoomerangLike:
    """Signed scalar rate g(t) = 1.5 cos(t + 0.3) + 0.4 cos(3t) + 0.3,
    lambda = max(g, 0). Period 2pi with inflection points, like a Boomerang
    rate on a non-Gaussian target."""
    name = "boomerang_like"
    kind = "scalar"
    offset = 0.0
    t_cap = 80.0
    # samplers' GridBoomerangSampler defaults, for the grid_thinning reference
    grid_t_max_init = math.pi / 4
    grid_spacing = math.pi / 16

    @staticmethod
    def g(t):
        return 1.5 * torch.cos(t + 0.3) + 0.4 * torch.cos(3.0 * t) + 0.3

    def rate(self, t_abs):
        return torch.clamp(self.g(t_abs), min=0.0)

    def closures(self, s):
        def rate_and_grad_fn(t_batch):
            return torch.func.jvp(lambda t: self.g(s + t), (t_batch,), (torch.ones_like(t_batch),))

        def rate_scalar_fn(t):
            return float(self.g(torch.tensor(s + t, dtype=DTYPE)))

        return rate_and_grad_fn, rate_scalar_fn

    def segment_bound_fn(self):
        return segment_bound_scalar

    def grid_bound_fn(self):
        return build_grid_bound


class ZigZagLike:
    """ZigZag rate on U(x) = 0.5 x'Ax + 0.25 sum_j c_j x_j^4 from a fixed
    (x, v): lambda(t) = sum_j (v_j grad_j U(x + tv))_+ + D*gamma. c=0 is the
    quadratic (affine per-coordinate) case."""
    kind = "vectorized"
    t_cap = 30.0
    # samplers' GridZigZagSampler defaults, for the grid_thinning reference
    grid_t_max_init = 0.1
    grid_spacing = 0.01

    def __init__(self, name, D=8, quartic=0.0, seed=0):
        self.name = name
        gen = torch.Generator().manual_seed(seed)
        M = torch.randn(D, D, generator=gen, dtype=DTYPE)
        self.A = M @ M.T / D + 0.5 * torch.eye(D, dtype=DTYPE)
        self.c = torch.full((D,), float(quartic), dtype=DTYPE)
        self.x = 0.3 * torch.randn(D, generator=gen, dtype=DTYPE)
        self.v = torch.where(torch.rand(D, generator=gen) < 0.5, -1.0, 1.0).to(DTYPE)
        self.D = D
        self.offset = D * GAMMA

    def per_coord(self, t):
        """Signed per-coordinate rates, [D, K] for t of shape [K]."""
        X = self.x[:, None] + self.v[:, None] * t[None, :]
        grad = self.A @ X + self.c[:, None] * X ** 3
        return self.v[:, None] * grad

    def rate(self, t_abs):
        return torch.clamp(self.per_coord(t_abs), min=0.0).sum(dim=0) + self.offset

    def closures(self, s):
        def rate_and_grad_fn(t_batch):
            return torch.func.jvp(lambda t: self.per_coord(s + t), (t_batch,), (torch.ones_like(t_batch),))

        def rate_scalar_fn(t):
            y = self.per_coord(torch.tensor([s + t], dtype=DTYPE))[:, 0]
            return float(torch.clamp(y, min=0.0).sum()) + self.offset

        return rate_and_grad_fn, rate_scalar_fn

    def segment_bound_fn(self):
        return functools.partial(segment_bound_vectorized, signed=True, offset=self.offset)

    def grid_bound_fn(self):
        return functools.partial(build_grid_bound_vectorized, signed=True, offset=self.offset)


TARGETS = [
    BoomerangLike(),
    ZigZagLike("zigzag_quadratic", quartic=0.0, seed=0),
    ZigZagLike("zigzag_quartic", quartic=0.3, seed=1),
]


def cumulative_rate(target, n_grid=400_001):
    """Lambda(t) on [0, t_cap] by trapezoid, returned as an interpolator."""
    t = torch.linspace(0.0, target.t_cap, n_grid, dtype=DTYPE)
    lam = target.rate(t)
    cum = torch.cat([torch.zeros(1, dtype=DTYPE),
                     torch.cumsum(0.5 * (lam[1:] + lam[:-1]) * (t[1:] - t[:-1]), dim=0)])
    t_np, cum_np = t.numpy(), cum.numpy()
    return lambda tau: np.interp(tau, t_np, cum_np)


# ---------------------------------------------------------------------------
# Outer loop: what a sampler's sample() does between two events
# ---------------------------------------------------------------------------

class Run:
    """t_max state that persists across events, like a sampler's _grid_t_max."""

    def __init__(self, method, t_max, adapt=None, cache=True, spacing=None, fixed=False):
        self.method = method        # "single" or "grid"
        self.t_max = t_max
        self.adapt = adapt          # "alg4" / "balanced"
        self.cache = cache
        self.spacing = spacing      # grid method only
        self.fixed = fixed          # no adaptation (t_max sweep)


def first_event(target, run: Run):
    """Simulate from t=0 until the first accepted event. Returns (T, stats)."""
    s = 0.0
    node = None
    out = {"evals": 0, "calls": 0, "violations": 0, "rejections": 0, "empty": 0}
    while True:
        rg, rs = target.closures(s)
        h = run.t_max
        if run.method == "single":
            tau, st = single_segment_thinning(
                rg, rs, h, left_node=node if run.cache else None, dtype=DTYPE,
                segment_bound_fn=target.segment_bound_fn(), rate_offset=target.offset,
            )
            node = st.pop("right_node")
            rejections = st["rejections"]
        else:
            n = int(min(max(math.ceil(h / run.spacing), 2), 60))
            tau, st = grid_thinning(
                rg, rs, h, n_segments=n, dtype=DTYPE,
                bound_fn=target.grid_bound_fn(), rate_offset=target.offset,
            )
            rejections = st["proposals"] - int(st["accepted"]) - st["bound_violations"]
            st["rejections"] = rejections

        out["evals"] += st["rate_evals"]
        out["calls"] += 1
        out["violations"] += st["bound_violations"]
        out["rejections"] += rejections
        if not st["accepted"] and not st["violated"]:
            out["empty"] += 1

        if not run.fixed:
            run.t_max = adapt_t_max(run.t_max, st, tau, h, True, run.adapt, 1.01, 1.04, 2.0)

        if st["accepted"]:
            return s + tau, out
        s += st["effective_horizon"]
        if s > target.t_cap:
            raise RuntimeError(f"{target.name}: no event before t_cap={target.t_cap}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_1_same_bound():
    print("\n[1] segment bound == build_*_grid_bound(n_segments=1)")
    gen = np.random.default_rng(0)
    for target in TARGETS:
        worst = 0.0
        for _ in range(200):
            s, h = gen.uniform(0, 10), gen.uniform(1e-3, 2.0)
            rg, _ = target.closures(s)
            _, seg, _, _ = target.grid_bound_fn()(rg, h, 1, torch.device("cpu"), DTYPE)
            t_local = torch.linspace(0.0, h, 2, dtype=DTYPE)
            y, d = rg(t_local)
            lam = target.segment_bound_fn()(t_local, y, d)
            worst = max(worst, abs(lam - float(seg[0])))
        assert worst == 0.0, f"{target.name}: max |diff| = {worst}"
        print(f"    {target.name:18s} identical over 200 windows")


def check_2_same_draws(n_calls=2000):
    print("\n[2] single_segment_thinning(left_node=None) == grid_thinning(n_segments=1), same seed")
    gen = np.random.default_rng(1)
    keys = ["accepted", "violated", "rejected_in_window", "proposals",
            "bound_violations", "rate_evals"]
    for target in TARGETS:
        n_acc = 0
        for i in range(n_calls):
            s, h = gen.uniform(0, 10), gen.uniform(1e-3, 2.0)
            rg, rs = target.closures(s)
            torch.manual_seed(i)
            tau_g, st_g = grid_thinning(rg, rs, h, n_segments=1, dtype=DTYPE,
                                        bound_fn=target.grid_bound_fn(), rate_offset=target.offset)
            torch.manual_seed(i)
            tau_s, st_s = single_segment_thinning(rg, rs, h, dtype=DTYPE,
                                                  segment_bound_fn=target.segment_bound_fn(),
                                                  rate_offset=target.offset)
            assert tau_g == tau_s, f"{target.name} call {i}: tau {tau_g} != {tau_s}"
            for k in keys:
                assert st_g[k] == st_s[k], f"{target.name} call {i}: {k} {st_g[k]} != {st_s[k]}"
            eg, es = st_g["effective_horizon"], st_s["effective_horizon"]
            assert eg == es, f"{target.name} call {i}: effective_horizon {eg} != {es}"
            n_acc += st_s["accepted"]
        print(f"    {target.name:18s} identical over {n_calls} calls ({n_acc} accepted)")


def check_3_cache(n_calls=500):
    print("\n[3] right_node == fresh evaluation at the next window's start")
    gen = np.random.default_rng(2)
    for target in TARGETS:
        worst, n_cached = 0.0, 0
        for _ in range(n_calls):
            s, h = gen.uniform(0, 10), gen.uniform(1e-3, 0.5)
            rg, rs = target.closures(s)
            _, st = single_segment_thinning(rg, rs, h, dtype=DTYPE,
                                            segment_bound_fn=target.segment_bound_fn(),
                                            rate_offset=target.offset)
            node = st["right_node"]
            if node is None:
                assert st["accepted"] or st["violated"], "no right_node on a clean empty window"
                continue
            n_cached += 1
            rg_next, _ = target.closures(s + h)
            y, d = rg_next(torch.zeros(1, dtype=DTYPE))
            scale = 1.0 + float(y.abs().max()) + float(d.abs().max())
            diff = max(float((node[0] - y[..., 0]).abs().max()),
                       float((node[1] - d[..., 0]).abs().max()))
            worst = max(worst, diff / scale)
        assert worst < 1e-12, f"{target.name}: relative diff {worst}"
        print(f"    {target.name:18s} {n_cached} cached nodes, max relative diff {worst:.1e}")


def check_4_event_law(n_reps):
    print(f"\n[4] Lambda(T) ~ Exp(1), outer loop with node cache, {n_reps} events per row")
    print(f"    {'target':18s} {'rule':9s} {'KS p':>8s} {'viol':>6s} {'evals/event':>12s}")
    failures = []
    for target in TARGETS:
        cum = cumulative_rate(target)
        for rule in ["alg4", "balanced"]:
            run = Run("single", target.grid_spacing, adapt=rule)
            T, evals, viol = np.empty(n_reps), 0, 0
            for r in range(n_reps):
                T[r], out = first_event(target, run)
                evals += out["evals"]
                viol += out["violations"]
            p = sps.kstest(cum(T), "expon").pvalue
            print(f"    {target.name:18s} {rule:9s} {p:8.3f} {viol:6d} {evals / n_reps:12.2f}")
            # With violations the bound was invalid somewhere, so the law is
            # not guaranteed -- report those, only fail clean runs
            if viol == 0 and p < 1e-3:
                failures.append((target.name, rule, p))
    assert not failures, f"KS failures: {failures}"


def check_5_adaptation(n_reps):
    print(f"\n[5] Adaptation study, evals per event ({n_reps} events per row)")
    for target in TARGETS:
        print(f"\n    {target.name}")
        # Fixed-width sweep: cost per event as a function of t_max
        widths = target.grid_spacing * np.logspace(-1, 2, 10)
        sweep = []
        print(f"      {'fixed t_max':>12s} {'evals/event':>12s} {'empty/event':>12s} "
              f"{'rej/event':>10s} {'viol/event':>11s}")
        for w in widths:
            run = Run("single", float(w), fixed=True)
            tot = {"evals": 0, "empty": 0, "rejections": 0, "violations": 0}
            for _ in range(n_reps):
                _, out = first_event(target, run)
                for k in tot:
                    tot[k] += out[k]
            per = {k: v / n_reps for k, v in tot.items()}
            sweep.append((w, per))
            print(f"      {w:12.4g} {per['evals']:12.2f} {per['empty']:12.2f} "
                  f"{per['rejections']:10.2f} {per['violations']:11.3f}")
        clean = [(w, p) for w, p in sweep if p["violations"] == 0] or sweep
        w_opt, p_opt = min(clean, key=lambda wp: wp[1]["evals"])
        print(f"      cost-optimal fixed width (no violations): {w_opt:.4g} "
              f"at {p_opt['evals']:.2f} evals/event")

        # Adaptive rules from several starting widths; stats over the last half
        print(f"      {'method':28s} {'t_max init':>10s} {'settled t_max':>14s} "
              f"{'evals/event':>12s} {'viol/event':>11s}")
        rows = [("single_segment alg4", "single", "alg4"),
                ("single_segment balanced", "single", "balanced")]
        for label, method, rule in rows:
            for init in [target.grid_spacing / 10, target.grid_spacing, target.grid_spacing * 10]:
                run = Run(method, init, adapt=rule)
                t_hist, evals, viol = [], [], []
                for _ in range(n_reps):
                    _, out = first_event(target, run)
                    t_hist.append(run.t_max)
                    evals.append(out["evals"])
                    viol.append(out["violations"])
                half = n_reps // 2
                print(f"      {label:28s} {init:10.4g} {np.median(t_hist[half:]):14.4g} "
                      f"{np.mean(evals[half:]):12.2f} {np.mean(viol[half:]):11.3f}")
        run = Run("grid", target.grid_t_max_init, adapt="alg4", spacing=target.grid_spacing)
        t_hist, evals, viol = [], [], []
        for _ in range(n_reps):
            _, out = first_event(target, run)
            t_hist.append(run.t_max)
            evals.append(out["evals"])
            viol.append(out["violations"])
        half = n_reps // 2
        print(f"      {'grid_thinning alg4 (ref)':28s} {target.grid_t_max_init:10.4g} "
              f"{np.median(t_hist[half:]):14.4g} {np.mean(evals[half:]):12.2f} "
              f"{np.mean(viol[half:]):11.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-reps", type=int, default=5000,
                        help="events per row in check 4 (check 5 uses a fifth of this)")
    parser.add_argument("--skip-study", action="store_true", help="skip check 5")
    args = parser.parse_args()

    t0 = time.perf_counter()
    check_1_same_bound()
    check_2_same_draws()
    check_3_cache()
    check_4_event_law(args.n_reps)
    if not args.skip_study:
        check_5_adaptation(max(args.n_reps // 5, 200))
    print(f"\nAll checks passed ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
