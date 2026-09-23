"""
Boomerang sweep over (refresh_rate, Sigma_inv scale) on the banana target.

WHY THIS SWEEP
--------------
The Boomerang is exact for ANY reference measure N(x_ref, Sigma): the Poisson
rate carries a correction term that cancels whatever the reference gets wrong.
So Sigma is a MIXING knob, not a modelling choice, and any dependence of the
sampled distribution on Sigma is either (a) finite-run mixing or (b) a defect.
This grid separates the two, with ZigZag -- which has no Sigma at all -- as a
fixed control repeated in every panel.

The two axes are NOT independent. The elliptical orbits have period ~ sqrt(Sigma),
so a fixed refresh_rate refreshes fewer times PER ORBIT as Sigma grows. The
dimensionless quantity that actually governs behaviour is roughly

    refreshes per orbit  ~  refresh_rate * sqrt(Sigma)  =  refresh_rate / sqrt(s)

for Sigma_inv scale s (Sigma scale = 1/s). It is printed per cell so the grid
can be read along its near-degenerate diagonals: cells sharing this value
should behave alike, and if they do, refresh_rate and Sigma are not two
independent knobs but one.

Caching: every cell is stored under results/paper/banana_sweep/ keyed by its
settings, so reruns only compute what is missing. --force recomputes.

Usage:
    python notebooks/banana_sweep.py
    python notebooks/banana_sweep.py --force
    python notebooks/banana_sweep.py --refresh 0.1 1.0 10.0 --sigma-scale 0.1 1.0 10.0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Run from anywhere: as a notebook cell (cwd=notebooks) or as a script from the
# repo root. Results paths below are relative, so cwd is pinned to the root.
_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != _ROOT:
    os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sazz.models.math_targets import make_banana
from sazz.gpu_friendly.samplers.grid_boomerang import GridBoomerangSampler
from sazz.gpu_friendly.samplers.grid_zigzag import GridZigZagSampler
from sazz.gpu_friendly.utils.resample import (
    resample_zigzag_path_torch, resample_boomerang_path_torch,
)

OUT = Path("results/paper/banana_sweep")
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def run_zigzag(target, N, n_resample, burnin_frac, seed, n_segments=20,
               grid_t_max_init=1.0):
    torch.manual_seed(seed)
    sampler = GridZigZagSampler(
        grad_target=target.grad_target, D=target.D, gamma=0.01,
        grid_t_max_init=grid_t_max_init, n_segments=n_segments, dtype=DTYPE,
    )
    t0 = time.perf_counter()
    res = sampler.sample(N=N, x0=target.x_ref, diagnostics=True)
    sec = time.perf_counter() - t0
    draws = resample_zigzag_path_torch(
        res["positions"], res["velocities"], res["times"],
        N_resample=n_resample, burnin_frac=burnin_frac,
    )
    return {"draws": draws, "sec": sec,
            "bv": int(res["bound_violations"]),
            "grad_evals": int(res["gradient_evals"]), "n_skel": N}


def run_boomerang(target, N, refresh_rate, sigma_scale, n_resample, burnin_frac,
                  seed, n_segments=20, grid_t_max_init=math.pi / 4):
    """Boomerang at one (refresh_rate, Sigma_inv scale) setting.

    NOTE: the scaling is applied HERE, explicitly, and the target is built with
    sigma_inv_scale=1.0. Scaling in both places silently multiplies -- the
    earlier brute-force script did exactly that (a hardcoded *0.1 on top of the
    target's own scale), so its three "distinct" settings were really
    0.1 / 0.01 / 1.0 rather than 1 / 0.1 / 10.
    """
    torch.manual_seed(seed)
    sampler = GridBoomerangSampler(
        grad_target=target.grad_target, D=target.D, refresh_rate=refresh_rate,
        grid_t_max_init=grid_t_max_init, n_segments=n_segments, dtype=DTYPE,
    )
    sampler.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv * sigma_scale)
    t0 = time.perf_counter()
    res = sampler.sample(N=N, diagnostics=True)
    sec = time.perf_counter() - t0
    draws = resample_boomerang_path_torch(
        res["positions"], res["velocities"], res["times"], target.x_ref,
        N_resample=n_resample, burnin_frac=burnin_frac,
    )
    return {"draws": draws, "sec": sec,
            "bv": int(res["bound_violations"]),
            "grad_evals": int(res["gradient_evals"]), "n_skel": N}


# ---------------------------------------------------------------------------
# Quantitative error against the analytic marginals
# ---------------------------------------------------------------------------

def marginal_l1(target, draws: torch.Tensor) -> tuple[float, float]:
    """L1 distance between the sampled and analytic marginal, per coordinate.

    A banana scatter shows TAIL REACH, which is not the same as correctness: a
    sampler can cover the arms and still weight them wrongly. The analytic
    marginals are available for this target, so each cell also gets a number.
    L1 (total-variation-like) is used rather than a moment: it is sensitive to
    the tail mass the scatter is about, whereas a mean or variance can look
    fine while the shape is wrong.
    """
    x = draws.numpy()
    out = []
    for c in (0, 1):
        info = target.marginal_grids[c]
        grid, pdf = info["grid"], info["pdf"]
        edges = np.concatenate([
            [grid[0] - 0.5 * (grid[1] - grid[0])],
            0.5 * (grid[1:] + grid[:-1]),
            [grid[-1] + 0.5 * (grid[-1] - grid[-2])],
        ])
        emp, _ = np.histogram(x[:, c], bins=edges, density=True)
        widths = np.diff(edges)
        out.append(float(np.sum(np.abs(emp - pdf) * widths)))
    return out[0], out[1]


def tail_frac(draws: torch.Tensor, thresh: float = 8.0) -> float:
    """Fraction of draws in the banana's arms (b1 > thresh).

    This is the number the 3x3 scatter is qualitatively showing. Making it
    explicit lets the panels be compared without eyeballing point density.
    """
    return float((draws[:, 1] > thresh).to(torch.float64).mean())


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cell_path(out: Path, refresh: float, sigma: float, N: int, seed: int) -> Path:
    return out / f"boom_rr{refresh:g}_sig{sigma:g}_N{N}_seed{seed}.pt"


def zz_path(out: Path, N: int, seed: int) -> Path:
    return out / f"zigzag_N{N}_seed{seed}.pt"


def load_or_run(path: Path, force: bool, fn, label: str):
    if path.exists() and not force:
        print(f"  [cached] {label}")
        return torch.load(path, weights_only=False)
    print(f"  [run]    {label} ...", end="", flush=True)
    res = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, path)
    print(f" {res['sec']:.1f}s  bv={res['bv']}")
    return res


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(target, zz, cells, refresh_rates, sigma_scales, path: Path,
                tail_thresh: float) -> None:
    """3x3 grid: rows = refresh_rate, cols = Sigma_inv scale.

    ZigZag is drawn in EVERY panel and is the same run throughout -- it has no
    Sigma and no refresh rate, so it is a fixed control. Anything that differs
    between panels is the Boomerang responding to its reference measure.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    zz_x = zz["draws"].numpy()

    # Analytic contours of the banana, for orientation.
    g0 = np.linspace(-10, 10, 300)
    g1 = np.linspace(-5, 20, 300)
    G0, G1 = np.meshgrid(g0, g1)
    a, scale = 1.0, 2.0
    U = 0.5 * (G0 / scale) ** 2 + 0.5 * (G1 - a * (G0 / scale) ** 2) ** 2

    nr, nc = len(refresh_rates), len(sigma_scales)
    fig, axes = plt.subplots(nr, nc, figsize=(11, 10),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for i, rr in enumerate(refresh_rates):
        for j, ss in enumerate(sigma_scales):
            ax = axes[i, j]
            c = cells[(rr, ss)]
            bx = c["draws"].numpy()

            ax.contour(G0, G1, U, levels=np.arange(1, 10, 1.5),
                       colors="0.7", linewidths=0.5, zorder=1)
            ax.scatter(zz_x[:, 0], zz_x[:, 1], s=2, alpha=0.40, linewidths=0,
                       color="C2", label="ZigZag", zorder=2)
            ax.scatter(bx[:, 0], bx[:, 1], s=2, alpha=0.50, linewidths=0,
                       color="C0", label="Boomerang", zorder=3)

            # refreshes per orbit: the quantity the two axes jointly control.
            rpo = rr / math.sqrt(ss)
            ax.set_title(f"$\\lambda_{{\\rm ref}}={rr:g}$,  "
                         f"$\gamma={ss:g}$", fontsize=18)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-5, 20)
            ax.set_xticks([-10, -5, 0, 5, 10])
            ax.set_yticks([-5, 0, 5, 10, 20])
            ax.tick_params(axis='both', labelsize=18)
            if i == nr - 1:
                ax.set_xlabel(r"$\beta_1$", fontsize=18)
            if j == 0:
                ax.set_ylabel(r"$\beta_2$", fontsize=18)
            #ax.tick_params(labelsize=7.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

    h, l = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper center", ncol=2, fontsize=20,
                     frameon=False, bbox_to_anchor=(0.5, 1.05))
    for lh in leg.legend_handles:
        lh.set_sizes([100]); lh.set_alpha(0.9)

    #fig.suptitle("Boomerang reference measure sweep on the banana "
    #             f"(tail = fraction with $\\beta_2>{tail_thresh:g}$; "
    #             "L1 = marginal error vs analytic)",
    #             fontsize=10, y=1.045)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nfigure -> {path}  (and .pdf)")
    return fig


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--sigma-scale", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--N", type=int, default=50_000, help="skeleton points")
    ap.add_argument("--n-resample", type=int, default=10_000)
    ap.add_argument("--burnin-frac", type=float, default=0.5)
    ap.add_argument("--tail-thresh", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # sigma_inv_scale=1.0: the sweep applies its own scaling in run_boomerang,
    # so the target carries none. Scaling in both places would multiply.
    target = make_banana(a=1.0, scale=2.0, sigma_inv_scale=1.0)
    print(f"banana: D={target.D}  Sigma_inv(base)=diag"
          f"{tuple(float(v) for v in torch.diagonal(target.Sigma_inv))}")
    print(f"N={args.N} skeleton, {args.n_resample} resampled, "
          f"burnin_frac={args.burnin_frac}, seed={args.seed}\n")

    print("ZigZag control (no Sigma, no refresh rate):")
    zz = load_or_run(
        zz_path(args.out, args.N, args.seed), args.force,
        lambda: run_zigzag(target, args.N, args.n_resample, args.burnin_frac,
                           args.seed),
        "zigzag")
    zz["l1_0"], zz["l1_1"] = marginal_l1(target, zz["draws"])
    zz["tail"] = tail_frac(zz["draws"], args.tail_thresh)

    print(f"\nBoomerang grid ({len(args.refresh)}x{len(args.sigma_scale)}):")
    cells = {}
    for rr in args.refresh:
        for ss in args.sigma_scale:
            res = load_or_run(
                cell_path(args.out, rr, ss, args.N, args.seed), args.force,
                lambda rr=rr, ss=ss: run_boomerang(
                    target, args.N, rr, ss, args.n_resample, args.burnin_frac,
                    args.seed),
                f"refresh={rr:<6g} sigma_scale={ss:g}")
            res["l1_0"], res["l1_1"] = marginal_l1(target, res["draws"])
            res["tail"] = tail_frac(res["draws"], args.tail_thresh)
            cells[(rr, ss)] = res

    # -- table --
    print(f"\n=== Marginal L1 error vs analytic, and tail mass "
          f"(beta2 > {args.tail_thresh:g}) ===")
    print("(Sigma is a MIXING knob: an exact Boomerang converges to the same")
    print(" marginals at every scale, so systematic drift across a row is the")
    print(" signal to look for. lam*sqrt(Sig) = refreshes per orbit.)\n")
    print(f"{'refresh':>9}{'sig_scale':>11}{'lam*sqrtSig':>13}"
          f"{'L1 b1':>9}{'L1 b2':>9}{'tail':>8}{'bv':>6}{'grad/skel':>11}{'sec':>8}")
    print("-" * 84)
    print(f"{'ZigZag':>9}{'--':>11}{'--':>13}"
          f"{zz['l1_0']:>9.4f}{zz['l1_1']:>9.4f}{zz['tail']:>8.4f}"
          f"{zz['bv']:>6}{zz['grad_evals']/zz['n_skel']:>11.1f}{zz['sec']:>8.1f}")
    for rr in args.refresh:
        for ss in args.sigma_scale:
            c = cells[(rr, ss)]
            print(f"{rr:>9g}{ss:>11g}{rr/math.sqrt(ss):>13.2f}"
                  f"{c['l1_0']:>9.4f}{c['l1_1']:>9.4f}{c['tail']:>8.4f}"
                  f"{c['bv']:>6}{c['grad_evals']/c['n_skel']:>11.1f}{c['sec']:>8.1f}")

    best = min(cells, key=lambda k: cells[k]["l1_0"] + cells[k]["l1_1"])
    print(f"\nlowest total L1: refresh={best[0]:g}, sigma_scale={best[1]:g} "
          f"(L1 {cells[best]['l1_0']:.4f}/{cells[best]['l1_1']:.4f})")

    make_figure(target, zz, cells, args.refresh, args.sigma_scale,
                args.out / f"banana_sweep_N{args.N}_seed{args.seed}.png",
                args.tail_thresh)


if __name__ == "__main__":
    main()
