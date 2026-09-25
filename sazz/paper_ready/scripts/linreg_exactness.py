"""Exactness of the sticky samplers on sparse linear regression.

    y | beta ~ N(X beta, sigma^2 I),   beta_i ~ w N(0, tau^2) + (1 - w) delta_0

The exact posterior over inclusion indicators comes from collapsed Gibbs
(beta integrated out). `run` samples one data set per seed with Gibbs and the
two sticky PDMPs (K = 2e5 events). `summary` pools the seeds and draws the
figure, posterior coefficients and the signed deviation of the PDMP inclusion
probabilities from Gibbs.

    python -m sazz.paper_ready.scripts.linreg_exactness run --seeds 0 1 2 3 4
    python -m sazz.paper_ready.scripts.linreg_exactness summary --seeds 0 1 2 3 4
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from ..samplers import StickyBoomerang, StickyZigZag, run_and_resample

DT = torch.float64
SAMPLERS = [("sticky_zigzag", "Sticky ZigZag", "C2", "o"), ("sticky_boomerang", "Sticky Boomerang", "C0", "^")]


def make_data(N, D, n_signals, seed, scale=1.5, sigma=1.0):
    """Signals of decaying size and alternating sign, so that some inclusion
    probabilities are strictly between 0 and 1."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, D, generator=g, dtype=DT)
    beta = torch.zeros(D, dtype=DT)
    beta[:n_signals] = torch.tensor([scale * (-1) ** i * 0.5 ** i for i in range(n_signals)], dtype=DT)
    return X, X @ beta + sigma * torch.randn(N, generator=g, dtype=DT), beta


def collapsed_gibbs(X, y, sigma, tau, w, n_draws, burnin, seed):
    """Gibbs over gamma in {0,1}^D with beta integrated out. beta | gamma, y is
    then drawn exactly. Returns beta draws [n_draws, D]."""
    torch.manual_seed(seed)
    D = X.shape[1]
    XtX, Xty = X.T @ X / sigma ** 2, X.T @ y / sigma ** 2
    prior_logit = math.log(w / (1 - w))

    def log_marginal(idx):
        if idx.numel() == 0:
            return 0.0, None, None
        L = torch.linalg.cholesky(XtX[idx][:, idx] + torch.eye(idx.numel(), dtype=DT) / tau ** 2)
        m = torch.cholesky_solve(Xty[idx].unsqueeze(-1), L).squeeze(-1)
        return 0.5 * float(m @ Xty[idx]) - float(torch.log(L.diagonal()).sum()) - idx.numel() * math.log(tau), m, L

    gamma = torch.zeros(D, dtype=torch.bool)
    gamma[(X.T @ y).abs().topk(5).indices] = True
    draws = torch.zeros(n_draws, D, dtype=DT)
    for it in range(burnin + n_draws):
        for i in torch.randperm(D).tolist():
            on, off = gamma.clone(), gamma.clone()
            on[i], off[i] = True, False
            z = prior_logit + log_marginal(on.nonzero()[:, 0])[0] - log_marginal(off.nonzero()[:, 0])[0]
            gamma[i] = torch.rand(1).item() < 1 / (1 + math.exp(-max(min(z, 30), -30)))
        if it >= burnin:
            idx = gamma.nonzero()[:, 0]
            if idx.numel():
                _, m, L = log_marginal(idx)
                eps = torch.randn(idx.numel(), 1, dtype=DT)
                draws[it - burnin, idx] = m + torch.linalg.solve_triangular(L.T, eps, upper=True)[:, 0]
    return draws


def run(args):
    for seed in args.seeds:
        path = args.out / f"linreg_seed{seed}.pt"
        if path.exists():
            continue
        X, y, beta = make_data(args.N, args.D, args.n_signals, seed)
        sigma, tau, w = 1.0, 1.0, args.w
        grad = lambda b: X.T @ (X @ b - y) / sigma ** 2 + b / tau ** 2   # slab-only energy
        prec = X.T @ X / sigma ** 2 + torch.eye(args.D, dtype=DT) / tau ** 2
        x_ref = torch.linalg.solve(prec, X.T @ y / sigma ** 2)
        kappa = w / (1 - w) / (tau * math.sqrt(2 * math.pi))
        res = {"beta_true": beta, "gibbs": collapsed_gibbs(X, y, sigma, tau, w, 20_000, 2_000, seed)}
        for name, cls, kw in [("sticky_zigzag", StickyZigZag, dict(gamma=1e-3)),
                              ("sticky_boomerang", StickyBoomerang,
                               dict(x_ref=x_ref, Sigma_inv=0.1 * prec.diagonal()))]:
            torch.manual_seed(seed)
            out = run_and_resample(cls(grad, args.D, kappa=kappa, t_max_init=1e-3, **kw), x_ref,
                                   n_out=args.n_draws, n_events=args.n_events)
            res[name] = out["samples"]
            print(f"  seed {seed} {name}: {out['bound_violations']} bound violations")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(res, path)


def ess(x: np.ndarray) -> float:
    """Effective sample size of a 0/1 trace by batch means."""
    b = max(1, int(x.size ** 0.5))
    means = x[: x.size // b * b].reshape(-1, b).mean(1)
    return float(x.size * x.var(ddof=1) / (b * means.var(ddof=1))) if means.var() > 0 else float(x.size)


def summary(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    incl = lambda d: (d.abs() > 0).double().mean(0).numpy()
    res = [torch.load(args.out / f"linreg_seed{s}.pt") for s in args.seeds]
    p_ref = np.concatenate([incl(r["gibbs"]) for r in res])
    print(f"{'sampler':<18}{'mean signed dev':>16}{'SE':>9}{'z':>7}{'mean |dev|':>12}")
    for key, label, _, _ in SAMPLERS:
        dev = np.concatenate([incl(r[key]) for r in res]) - p_ref
        se = dev.std(ddof=1) / np.sqrt(dev.size)
        print(f"{label:<18}{dev.mean():>16.5f}{se:>9.5f}{dev.mean() / se:>7.2f}{np.abs(dev).mean():>12.5f}")

    beta = res[0]["beta_true"].numpy()
    n_show = int((beta != 0).sum()) + 4
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    ax = axes[0]
    for (key, label, color, marker), dx in zip([("gibbs", "Gibbs", "C4", "D")] + SAMPLERS, [-0.26, 0, 0.26]):
        m = np.mean([r[key].mean(0).numpy() for r in res], 0)[:n_show]
        sd = np.mean([r[key].std(0).numpy() for r in res], 0)[:n_show]
        ax.errorbar(np.arange(n_show) + dx, m, yerr=sd, fmt=marker, color=color, ms=3.4, lw=0,
                    elinewidth=1.0, label=label)
    ax.scatter(np.arange(n_show), beta[:n_show], marker="*", s=15, color="C1", label=r"$\beta_i$", zorder=4)
    ax.axhline(0, color="0.75", lw=0.7)
    ax.set(xlabel="coordinate $i$", ylabel="Posterior mean", title="Posterior coefficients")
    ax.legend(fontsize=9, ncol=2)
    ax = axes[1]
    ax.axhline(0, color="0.35", lw=0.9)
    # +-2 Monte Carlo SD of the deviation, sqrt(p(1-p)(1/ESS_gibbs + 1/ESS_pdmp)),
    # with the median ESS-based constant over the uncertain coordinates
    ratios = []
    for r in res:
        p = incl(r["gibbs"])
        for key, *_ in SAMPLERS:
            for i in np.nonzero((p > 0.05) & (p < 0.95))[0]:
                e = [ess((r[k][:, i] != 0).double().numpy()) for k in ("gibbs", key)]
                ratios.append(np.sqrt(1 / e[0] + 1 / e[1]))
    grid = np.linspace(0, 1, 201)
    band = 2 * np.median(ratios) * np.sqrt(grid * (1 - grid))
    ax.fill_between(grid, -band, band, color="0.75", alpha=0.55, lw=0)
    for key, label, color, marker in SAMPLERS:
        dev = np.concatenate([incl(r[key]) for r in res]) - p_ref
        ax.scatter(p_ref, dev, s=20, color=color, marker=marker, alpha=0.75, lw=0, label=label)
    ax.set(xlabel=r"Gibbs $P(\gamma_i=1\mid y)$", ylabel="Sticky PDMP deviation", title="Signed deviation")
    ax.legend(fontsize=9)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "linreg_exactness.pdf", bbox_inches="tight")
    print(f"figure -> {args.out / 'linreg_exactness.pdf'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "summary"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--D", type=int, default=20)
    p.add_argument("--n-signals", type=int, default=5)
    p.add_argument("--w", type=float, default=0.2, help="prior inclusion probability")
    p.add_argument("--n-events", type=int, default=200_000)
    p.add_argument("--n-draws", type=int, default=50_000)
    p.add_argument("--out", type=Path, default=Path("results/linreg_exactness"))
    args = p.parse_args()
    (run if args.command == "run" else summary)(args)


if __name__ == "__main__":
    main()
