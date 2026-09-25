"""Boomerang on a 2-D banana over a grid of refresh rates and reference
precision scales, with ZigZag (no reference, no refreshment) as a control in
every panel. Every run has K = 5e4 events.

    U(b) = 0.5 (b1 / 2)^2 + 0.5 (b2 - (b1 / 2)^2)^2

    python -m sazz.paper_ready.scripts.banana
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from scipy import integrate

from ..samplers import Boomerang, ZigZag, run_and_resample

SCALE, A = 2.0, 1.0
X_REF = torch.tensor([0.0, A], dtype=torch.float64)
SIGMA_INV = torch.tensor([1.0, 1.0 / 3.0], dtype=torch.float64)


def grad_U(b):
    u = b[0] / SCALE
    r = b[1] - A * u ** 2
    return torch.stack([u / SCALE - 2 * A * u / SCALE * r, r])


def marginals():
    """Analytic marginal densities of b1 and b2 on grids."""
    g1 = np.linspace(-4 * SCALE, 4 * SCALE, 500)
    g2 = np.linspace(-4, 4 + 16 * A, 500)
    p1 = np.exp(-0.5 * (g1 / SCALE) ** 2) / (SCALE * np.sqrt(2 * np.pi))
    joint = lambda b1, b2: np.exp(-0.5 * (b1 / SCALE) ** 2 - 0.5 * (b2 - A * (b1 / SCALE) ** 2) ** 2)
    p2 = np.array([integrate.quad(joint, -8 * SCALE, 8 * SCALE, args=(v,))[0]
                   for v in g2]) / (SCALE * 2 * np.pi)
    return [(g1, p1), (g2, p2)]


def l1(draws, grid, pdf):
    edges = np.concatenate([[grid[0] - (grid[1] - grid[0]) / 2], (grid[1:] + grid[:-1]) / 2,
                            [grid[-1] + (grid[-1] - grid[-2]) / 2]])
    emp, _ = np.histogram(draws, bins=edges, density=True)
    return float(np.sum(np.abs(emp - pdf) * np.diff(edges)))


def run(sampler, args, path: Path):
    if path.exists():
        return torch.load(path)
    torch.manual_seed(args.seed)
    out = run_and_resample(sampler, X_REF, n_out=args.n_draws, burnin_frac=0.5,
                           n_events=args.n_events, progress=False)
    res = {"draws": out["samples"], "bv": out["bound_violations"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, path)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    p.add_argument("--sigma-scale", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    p.add_argument("--n-events", type=int, default=50_000)
    p.add_argument("--n-draws", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/banana"))
    args = p.parse_args()

    zz = run(ZigZag(grad_U, 2, gamma=0.01, t_max_init=0.05), args, args.out / "zigzag.pt")
    cells = {(rr, ss): run(Boomerang(grad_U, 2, X_REF, ss * SIGMA_INV, refresh_rate=rr, t_max_init=0.04),
                           args, args.out / f"boomerang_rr{rr:g}_sig{ss:g}.pt")
             for rr in args.refresh for ss in args.sigma_scale}

    marg = marginals()
    print(f"{'refresh':>8}{'scale':>7}{'L1 b1':>8}{'L1 b2':>8}{'viol':>6}")
    for key, res in [(("ZigZag", ""), zz)] + list(cells.items()):
        d = res["draws"].numpy()
        print(f"{key[0]:>8}{key[1]:>7}{l1(d[:, 0], *marg[0]):>8.4f}{l1(d[:, 1], *marg[1]):>8.4f}{res['bv']:>6}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G1, G2 = np.meshgrid(np.linspace(-10, 10, 300), np.linspace(-5, 20, 300))
    U = 0.5 * (G1 / SCALE) ** 2 + 0.5 * (G2 - A * (G1 / SCALE) ** 2) ** 2
    fig, axes = plt.subplots(len(args.refresh), len(args.sigma_scale), figsize=(11, 10),
                             sharex=True, sharey=True, squeeze=False)
    for i, rr in enumerate(args.refresh):
        for j, ss in enumerate(args.sigma_scale):
            ax = axes[i, j]
            ax.contour(G1, G2, U, levels=np.arange(1, 10, 1.5), colors="0.7", linewidths=0.5)
            for res, color, label in [(zz, "C2", "ZigZag"), (cells[(rr, ss)], "C0", "Boomerang")]:
                d = res["draws"].numpy()
                ax.scatter(d[:, 0], d[:, 1], s=2, alpha=0.45, lw=0, color=color, label=label)
            ax.set_title(rf"$\lambda_{{\rm ref}}={rr:g}$,  $\gamma={ss:g}$", fontsize=18)
            ax.set(xlim=(-10, 10), ylim=(-5, 20))
            ax.tick_params(labelsize=18)
            ax.spines[["top", "right"]].set_visible(False)
            if i == len(args.refresh) - 1:
                ax.set_xlabel(r"$\beta_1$", fontsize=18)
            if j == 0:
                ax.set_ylabel(r"$\beta_2$", fontsize=18)
    leg = fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2,
                     fontsize=20, frameon=False, bbox_to_anchor=(0.5, 1.05))
    for h in leg.legend_handles:
        h.set_sizes([100])
    fig.tight_layout()
    fig.savefig(args.out / "banana_sweep.pdf", bbox_inches="tight")
    print(f"figure -> {args.out / 'banana_sweep.pdf'}")


if __name__ == "__main__":
    main()
