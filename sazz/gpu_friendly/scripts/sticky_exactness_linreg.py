"""
Exactness check for the sticky PDMP samplers on sparse linear regression.

The point: sticky ZigZag / sticky Boomerang claim to target the spike-and-slab
posterior

    y | beta ~ N(X beta, sigma^2 I),    beta_i ~ w N(0, tau^2) + (1-w) delta_0

exactly. On a conjugate linear-Gaussian model that posterior is available to a
COLLAPSED GIBBS sampler: integrate beta out analytically and sample the
inclusion indicators gamma in {0,1}^D from their exact conditionals. Gibbs
therefore provides a ground-truth reference for
  * marginal inclusion probabilities P(gamma_i = 1 | y), and
  * posterior means E[beta_i | y],
against which the PDMPs' time-averaged draws can be compared directly. Any
systematic disagreement is a defect in the sticky mechanism (freeze/thaw
balance, rate bound, resampling), not a modelling difference -- all three
samplers target the same object by construction.

The PDMPs' kappa is built with the SAME formula the BNN drivers use
(build_kappa_from_inclusion, sazz/gpu_friendly/models/priors.py):
    kappa_i = (w / (1-w)) / (tau * sqrt(2 pi))
so the spike-and-slab prior the PDMPs see and the one Gibbs marginalizes are
the same prior, not two parameterizations that happen to look similar.

Results are cached: the samplers run only if the .pt is missing (or --force),
otherwise the script goes straight to the plot/table from disk.

Usage:
    python -m sazz.gpu_friendly.scripts.sticky_exactness_linreg
    python -m sazz.gpu_friendly.scripts.sticky_exactness_linreg --force
    python -m sazz.gpu_friendly.scripts.sticky_exactness_linreg --D 20 --n-signals 5
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from sazz.gpu_friendly.samplers.fast_grid_sticky_zigzag import FastGridStickyZigZagSampler
from sazz.gpu_friendly.samplers.fast_grid_sticky_boomerang import FastGridStickyBoomerangSampler
from sazz.gpu_friendly.utils.resample import (
    resample_zigzag_path_sticky_torch, resample_boomerang_path_sticky_torch,
)

DTYPE = torch.float64
DEVICE = "cpu"          # D=20: CPU is faster than any GPU round-trip here
OUT = Path("results/paper/sticky_exactness")

# ---------------------------------------------------------------------------
# Model / data
# ---------------------------------------------------------------------------

def make_data(N: int, D: int, n_signals: int, signal_scale: float,
              noise_std: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, D, generator=g, dtype=DTYPE)
    beta_true = torch.zeros(D, dtype=DTYPE)
    # Geometrically decaying magnitudes with alternating signs: the largest
    # signals are unambiguous (P(gamma=1|y) ~ 1) while the smallest sit near
    # the noise floor, producing INTERMEDIATE inclusion probabilities. Those
    # are the coordinates that actually discriminate between samplers -- a
    # scatter with mass only at 0 and 1 would validate very little.
    signs = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(n_signals)], dtype=DTYPE)
    decay = torch.tensor([0.5 ** i for i in range(n_signals)], dtype=DTYPE)
    beta_true[:n_signals] = signal_scale * signs * decay
    y = X @ beta_true + noise_std * torch.randn(N, generator=g, dtype=DTYPE)
    return X, y, beta_true


def log_posterior_grad(X: Tensor, y: Tensor, sigma: float, tau: float):
    """grad of the ENERGY (= -log density) of the SLAB-only posterior.

    The sticky sampler supplies the spike itself through freeze/thaw (kappa);
    the gradient it needs is of the continuous part only, i.e. Gaussian
    likelihood + N(0, tau^2) slab prior. Returned as a closure over
    precomputed X'X / X'y so each call is O(D^2), not O(N D).
    """
    XtX = (X.T @ X) / (sigma ** 2)
    Xty = (X.T @ y) / (sigma ** 2)
    prior_prec = 1.0 / (tau ** 2)

    def grad_energy(beta: Tensor) -> Tensor:
        return XtX @ beta - Xty + prior_prec * beta

    return grad_energy


# ---------------------------------------------------------------------------
# Exact reference: collapsed Gibbs over inclusion indicators
# ---------------------------------------------------------------------------

def collapsed_gibbs(X: Tensor, y: Tensor, sigma: float, tau: float, w: float,
                    n_draws: int, burnin: int, seed: int):
    """
    Exact spike-and-slab posterior via collapsed Gibbs on gamma in {0,1}^D.

    With beta_S ~ N(0, tau^2 I) on the active set S = {i: gamma_i = 1} and
    y ~ N(X_S beta_S, sigma^2 I), beta_S integrates out in closed form:

    log p(y | gamma) = 0.5*m_S' M_S m_S - 0.5*log|M_S| - k*log(tau) + const

    where M_S = X_S'X_S/sigma^2 + I/tau^2 and m_S = M_S^{-1} X_S'y/sigma^2.
    Each sweep visits coordinates in a random order and flips gamma_i from its
    exact Bernoulli conditional, so the chain is an exact-target MCMC on the
    discrete model. beta is then drawn from its exact Gaussian conditional
    given gamma, giving posterior draws of beta on the same footing as the
    PDMPs' draws.

    Returns (beta_draws [n_draws, D], incl_prob [D]).
    """
    torch.manual_seed(seed)
    N, D = X.shape
    XtX = (X.T @ X) / (sigma ** 2)
    Xty = (X.T @ y) / (sigma ** 2)
    prior_prec = 1.0 / (tau ** 2)
    log_odds_prior = math.log(w / (1.0 - w))

    def log_marg(idx: Tensor) -> tuple[float, Tensor, Tensor]:
        """log p(y|gamma) up to a gamma-independent constant, plus (m, L)."""
        k = int(idx.numel())
        if k == 0:
            return 0.0, torch.zeros(0, dtype=DTYPE), torch.zeros(0, 0, dtype=DTYPE)
        M = XtX[idx][:, idx] + prior_prec * torch.eye(k, dtype=DTYPE)
        L = torch.linalg.cholesky(M)
        rhs = Xty[idx]
        m = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        # 0.5*m'M m - sum(log diag L) - k*log(tau)
        quad = 0.5 * float(m @ rhs)
        logdet = float(torch.log(torch.diagonal(L)).sum())
        return quad - logdet - k * math.log(tau), m, L

    gamma = torch.zeros(D, dtype=torch.bool)
    # Warm, data-driven init: marginal correlation screen. Only affects mixing.
    gamma[(X.T @ y).abs().topk(min(D, 5)).indices] = True

    beta_draws = torch.zeros(n_draws, D, dtype=DTYPE)
    incl_count = torch.zeros(D, dtype=DTYPE)
    total = burnin + n_draws

    for it in range(total):
        for i in torch.randperm(D).tolist():
            g_on = gamma.clone();  g_on[i] = True
            g_off = gamma.clone(); g_off[i] = False
            lm_on, _, _ = log_marg(torch.nonzero(g_on, as_tuple=True)[0])
            lm_off, _, _ = log_marg(torch.nonzero(g_off, as_tuple=True)[0])
            # P(gamma_i=1 | rest, y) via the log-odds, numerically stable.
            z = log_odds_prior + lm_on - lm_off
            p_on = 1.0 / (1.0 + math.exp(-z)) if abs(z) < 30 else (1.0 if z > 0 else 0.0)
            gamma[i] = bool(torch.rand(1).item() < p_on)

        if it >= burnin:
            j = it - burnin
            idx = torch.nonzero(gamma, as_tuple=True)[0]
            beta = torch.zeros(D, dtype=DTYPE)
            if idx.numel() > 0:
                _, m, L = log_marg(idx)
                # beta_S | gamma, y ~ N(m, M^{-1}); sample via the Cholesky.
                eps = torch.randn(idx.numel(), dtype=DTYPE)
                beta[idx] = m + torch.linalg.solve_triangular(
                    L.T, eps.unsqueeze(-1), upper=True).squeeze(-1)
            beta_draws[j] = beta
            incl_count += gamma.to(DTYPE)

    return beta_draws, incl_count / n_draws


# ---------------------------------------------------------------------------
# Sticky PDMP runs
# ---------------------------------------------------------------------------

def run_sticky_zigzag(X, y, sigma, tau, w, D, n_skel, n_resample, burnin_frac,
                      x0, seed):
    torch.manual_seed(seed)
    kappa = (w / (1.0 - w)) / (tau * math.sqrt(2.0 * math.pi))
    sampler = FastGridStickyZigZagSampler(
        grad_target=log_posterior_grad(X, y, sigma, tau),
        D=D, kappa=kappa, can_freeze=torch.ones(D, dtype=torch.bool),
        gamma=1e-3, grid_t_max_init=0.1, n_segments=40, grid_spacing=1e-3,
        alpha_plus=1.01, alpha_minus=1.04, alpha_violation=1.1,
        dtype=DTYPE, device=DEVICE,
    )
    t0 = time.perf_counter()
    res = sampler.sample(N=n_skel, x0=x0, diagnostics=False)
    elapsed = time.perf_counter() - t0
    draws = resample_zigzag_path_sticky_torch(
        res["positions"], res["velocities"], res["times"],
        N_resample=n_resample, burnin_frac=burnin_frac,
    )
    return draws, elapsed, int(res["bound_violations"])


def run_sticky_boomerang(X, y, sigma, tau, w, D, n_skel, n_resample, burnin_frac,
                         x_ref, Sigma_inv, seed):
    torch.manual_seed(seed)
    kappa = (w / (1.0 - w)) / (tau * math.sqrt(2.0 * math.pi))
    sampler = FastGridStickyBoomerangSampler(
        grad_target=log_posterior_grad(X, y, sigma, tau),
        D=D, kappa=kappa, can_freeze=torch.ones(D, dtype=torch.bool),
        refresh_rate=1.0, grid_t_max_init=0.1, n_segments=40, grid_spacing=1e-3,
        alpha_plus=1.01, alpha_minus=1.04, alpha_violation=1.1,
        dtype=DTYPE, device=DEVICE,
    )
    sampler.preprocess(x_ref=x_ref, Sigma_inv=0.1*Sigma_inv)
    t0 = time.perf_counter()
    res = sampler.sample(N=n_skel, x0=x_ref, diagnostics=False)
    elapsed = time.perf_counter() - t0
    draws = resample_boomerang_path_sticky_torch(
        res["positions"], res["velocities"], res["times"], x_ref,
        N_resample=n_resample, burnin_frac=burnin_frac,
    )
    return draws, elapsed, int(res["bound_violations"])


# ---------------------------------------------------------------------------
# Run-or-load
# ---------------------------------------------------------------------------

def produce_results(args, path: Path) -> dict:
    X, y, beta_true = make_data(args.N, args.D, args.n_signals,
                                args.signal_scale, args.noise_std, args.seed)
    sigma, tau, w = args.noise_std, args.tau, args.w

    # Boomerang reference measure: the SLAB-only Gaussian posterior, available
    # in closed form here (no MAP/Laplace needed at D=20).
    XtX = (X.T @ X) / sigma ** 2
    Sigma_inv_full = XtX + torch.eye(args.D, dtype=DTYPE) / tau ** 2
    x_ref = torch.linalg.solve(Sigma_inv_full, (X.T @ y) / sigma ** 2)
    Sigma_inv = torch.diagonal(Sigma_inv_full).clone()

    print(f"D={args.D}  N={args.N}  n_signals={args.n_signals}  w={w}  tau={tau}")

    print("  [gibbs] collapsed Gibbs (exact reference) ...")
    t0 = time.perf_counter()
    gibbs_draws, gibbs_incl = collapsed_gibbs(
        X, y, sigma, tau, w, args.gibbs_draws, args.gibbs_burnin, args.seed)
    gibbs_sec = time.perf_counter() - t0
    print(f"          {args.gibbs_draws} draws in {gibbs_sec:.1f}s")

    print("  [sticky_zigzag] ...")
    zz, zz_sec, zz_bv = run_sticky_zigzag(
        X, y, sigma, tau, w, args.D, args.n_skel, args.n_resample,
        args.burnin_frac, x_ref, args.seed)
    print(f"          {args.n_skel} events in {zz_sec:.1f}s, {zz_bv} bound violations")

    print("  [sticky_boomerang] ...")
    bm, bm_sec, bm_bv = run_sticky_boomerang(
        X, y, sigma, tau, w, args.D, args.n_skel, args.n_resample,
        args.burnin_frac, x_ref, Sigma_inv, args.seed)
    print(f"          {args.n_skel} events in {bm_sec:.1f}s, {bm_bv} bound violations")

    out = {
        "beta_true": beta_true, "w": w, "tau": tau, "sigma": sigma,
        "D": args.D, "N": args.N, "n_signals": args.n_signals,
        "gibbs": {"draws": gibbs_draws, "incl": gibbs_incl, "sec": gibbs_sec},
        "sticky_zigzag": {"draws": zz, "sec": zz_sec, "bv": zz_bv},
        "sticky_boomerang": {"draws": bm, "sec": bm_sec, "bv": bm_bv},
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"  saved -> {path}")
    return out


# ---------------------------------------------------------------------------
# Plot + table
# ---------------------------------------------------------------------------

ZERO_TOL = 1e-10


def incl_prob(draws: Tensor) -> Tensor:
    """Marginal inclusion probability = fraction of draws with beta_i != 0.

    Exact for these samplers: a frozen coordinate is written as EXACTLY 0.0
    (the sticky resamplers force it), and the slab is continuous, so a
    nonzero test is an unambiguous indicator read-out.
    """
    return (draws.abs() > ZERO_TOL).to(DTYPE).mean(dim=0)


def make_plot(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = res["D"]
    beta_true = res["beta_true"].numpy()
    order = np.arange(D)

    names = [("gibbs", "Collapsed Gibbs (exact)", "#333333", "o"),
             ("sticky_zigzag", "Sticky ZigZag", "#1f77b4", "s"),
             ("sticky_boomerang", "Sticky Boomerang", "#d62728", "^")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # -- left: marginal inclusion probability, per coordinate --
    ax = axes[0]
    width = 0.26
    for k, (key, lab, col, _) in enumerate(names):
        p = incl_prob(res[key]["draws"]).numpy()
        ax.bar(order + (k - 1) * width, p, width, label=lab, color=col, alpha=0.85)
    for i in range(res["n_signals"]):
        ax.axvspan(i - 0.5, i + 0.5, color="gold", alpha=0.18, zorder=0)
    ax.set_xlabel("coordinate")
    ax.set_ylabel(r"$P(\gamma_i = 1 \mid y)$")
    ax.set_title("Marginal inclusion probability\n(shaded = true signals)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="center right")

    # -- right: exact vs PDMP, scatter (the exactness check) --
    ax = axes[1]
    p_ref = incl_prob(res["gibbs"]["draws"]).numpy()
    for key, lab, col, mk in names[1:]:
        p = incl_prob(res[key]["draws"]).numpy()
        mad = np.abs(p - p_ref).mean()
        ax.scatter(p_ref, p, s=46, color=col, marker=mk, alpha=0.8,
                   label=f"{lab}  (MAD {mad:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, zorder=0)
    ax.set_xlabel("Gibbs (exact)  $P(\\gamma_i=1\\mid y)$")
    ax.set_ylabel("sticky PDMP  $P(\\gamma_i=1\\mid y)$")
    ax.set_title("Sticky PDMP vs exact posterior")
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  plot -> {path}")


def make_table(res: dict) -> str:
    p_ref = incl_prob(res["gibbs"]["draws"])
    m_ref = res["gibbs"]["draws"].mean(dim=0)
    beta_true = res["beta_true"]
    sig = beta_true.abs() > 0

    # Monte Carlo scale of the REFERENCE itself: Gibbs is an exact-target MCMC,
    # not an oracle, so its own inclusion estimates carry error ~ sqrt(p(1-p)/n)
    # (independence bound; true Gibbs error is larger, since sweeps autocorrelate).
    # Printing it makes the deviations below interpretable: a sampler agreeing to
    # within this scale is indistinguishable from exact at this sample size.
    n_g = res["gibbs"]["draws"].shape[0]
    mc_ref = float((p_ref * (1 - p_ref) / n_g).sqrt().mean())

    lines = []
    lines.append(f"{'sampler':<24}{'incl MAD':>10}{'incl max':>10}"
                 f"{'mean MAD':>10}{'sparsity':>10}{'bound viol':>12}{'sec':>9}")
    lines.append("-" * 85)
    for key, lab in [("gibbs", "Collapsed Gibbs (exact)"),
                     ("sticky_zigzag", "Sticky ZigZag"),
                     ("sticky_boomerang", "Sticky Boomerang")]:
        d = res[key]["draws"]
        p = incl_prob(d)
        m = d.mean(dim=0)
        spars = float((d.abs() <= ZERO_TOL).to(DTYPE).mean())
        bv = res[key].get("bv", 0)
        if key == "gibbs":
            # The reference has no deviation FROM ITSELF -- these columns are
            # undefined for it, not zero. Printing 0.0000 would read as a score
            # Gibbs won, which is the opposite of what the row means.
            cols = f"{'--':>10}{'--':>10}{'--':>10}"
        else:
            imad = float((p - p_ref).abs().mean())
            imax = float((p - p_ref).abs().max())
            mmad = float((m - m_ref).abs().mean())
            cols = f"{imad:>10.4f}{imax:>10.4f}{mmad:>10.4f}"
        lines.append(f"{lab:<24}{cols}"
                     f"{spars:>10.3f}{bv:>12}{res[key]['sec']:>9.1f}")
    lines.append("")
    lines.append(f"incl MAD / incl max / mean MAD are deviations FROM the Gibbs reference;")
    lines.append(f"undefined (--) for Gibbs itself. Gibbs' own MC error on the inclusion")
    lines.append(f"probabilities is ~{mc_ref:.4f} (mean over coords, n={n_g}), so deviations")
    lines.append(f"at or below that scale are within reference noise.")

    lines.append("")
    lines.append("Per-coordinate inclusion probability (first 10 coords; * = true signal)")
    lines.append(f"{'coord':<8}{'true beta':>11}{'Gibbs':>9}{'stickyZZ':>10}{'stickyBoom':>12}")
    p_zz = incl_prob(res["sticky_zigzag"]["draws"])
    p_bm = incl_prob(res["sticky_boomerang"]["draws"])
    for i in range(min(10, res["D"])):
        star = "*" if sig[i] else " "
        lines.append(f"{i:<3}{star:<5}{float(beta_true[i]):>11.2f}"
                     f"{float(p_ref[i]):>9.3f}{float(p_zz[i]):>10.3f}{float(p_bm[i]):>12.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--D", type=int, default=20)
    ap.add_argument("--n-signals", type=int, default=5)
    ap.add_argument("--signal-scale", type=float, default=1.5)
    ap.add_argument("--noise-std", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0, help="slab std")
    ap.add_argument("--w", type=float, default=0.2, help="prior inclusion prob")
    ap.add_argument("--n-skel", type=int, default=200_000)
    ap.add_argument("--n-resample", type=int, default=50_000)
    ap.add_argument("--burnin-frac", type=float, default=0.2)
    ap.add_argument("--gibbs-draws", type=int, default=20_000)
    ap.add_argument("--gibbs-burnin", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true", help="resample even if cached")
    args = ap.parse_args()

    res_path = args.out / f"linreg_D{args.D}_w{args.w}_seed{args.seed}.pt"
    if res_path.exists() and not args.force:
        print(f"Loading cached results <- {res_path}  (--force to resample)")
        res = torch.load(res_path, weights_only=False)
    else:
        res = produce_results(args, res_path)

    make_plot(res, args.out / f"linreg_D{res['D']}_w{res['w']}_seed{args.seed}.png")
    table = make_table(res)
    print()
    print(table)
    (args.out / f"linreg_D{res['D']}_w{res['w']}_seed{args.seed}.txt").write_text(table + "\n")


if __name__ == "__main__":
    main()
