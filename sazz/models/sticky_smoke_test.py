"""
sticky_smoke_test.py

Validation targets for the Sticky Boomerang sampler.
Each target has genuine sparsity (point mass at zero) so the
freeze/thaw mechanics are well-specified.

Targets
-------
1. Spike-and-slab Gaussian: independent coords with
       p(beta_i) = w * delta(0) + (1-w) * N(0, sigma^2)
   The continuous part of the posterior has a known density, and
   the marginal inclusion probability is 1-w.

2. Sparse orthogonal regression: y = X @ beta + noise with
   orthogonal X and spike-and-slab prior.  Each coordinate
   decouples, giving a known marginal.

3. Independent Laplace: E(x) = sum lambda_i |x_i|.  Cusp at
   zero stress-tests freeze/thaw even though true P(x_i=0)=0.
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import integrate


# ===========================================================================
# Target container (same as smoke_test.py)
# ===========================================================================

class TorchTarget:
    def __init__(self, name, D, grad_target, x_ref, Sigma_inv, marginal_grids=None, meta=None):
        self.name = name
        self.D = D
        self.grad_target = grad_target
        self.x_ref = x_ref
        self.Sigma_inv = Sigma_inv
        self.marginal_grids = marginal_grids
        self.meta = meta or {}


# ===========================================================================
# 1. Spike-and-slab Gaussian
# ===========================================================================

def make_spike_and_slab(
    D=10,
    w_spike=0.7,
    sigma_slab=2.0,
    dtype=torch.float64,
):
    """
    Independent spike-and-slab on each coordinate:
        p(beta_i) = w * delta(0) + (1-w) * N(0, sigma^2)

    The sticky sampler targets the continuous (slab) part of the
    density.  The energy for the slab component is:
        E(beta) = 0.5 * sum_i beta_i^2 / sigma^2

    The expected fraction of frozen coordinates at stationarity
    should converge to w_spike.

    Parameters
    ----------
    D : int
        Dimension.
    w_spike : float
        Prior probability of being exactly zero (spike weight).
    sigma_slab : float
        Standard deviation of the slab (nonzero) component.
    """
    prec = 1.0 / sigma_slab**2
    Sigma_inv_t = prec * torch.eye(D, dtype=dtype)

    def grad_target(x):
        # grad of E = 0.5 * x^T (sigma^-2 I) x  =>  sigma^-2 * x
        return prec * x

    # Marginal: mixture of delta(0) and N(0, sigma^2)
    grid = np.linspace(-4 * sigma_slab, 4 * sigma_slab, 500)
    slab_pdf = (1.0 - w_spike) * np.exp(-0.5 * grid**2 / sigma_slab**2) / (
        sigma_slab * np.sqrt(2 * np.pi)
    )
    # The spike contributes a delta at 0 — visible as the fraction of
    # samples exactly at zero, not as a smooth PDF.  We store the
    # slab density and the spike weight separately for plotting.

    marginal_grids = {}
    for i in range(D):
        marginal_grids[i] = {
            "grid": grid,
            "pdf": slab_pdf,
            "spike_weight": w_spike,
            "label": rf"$\beta_{{{i+1}}}$",
        }

    return TorchTarget(
        name=f"spike_slab_D{D}_w{w_spike}_s{sigma_slab}",
        D=D,
        grad_target=grad_target,
        x_ref=torch.zeros(D, dtype=dtype),
        Sigma_inv=Sigma_inv_t,
        marginal_grids=marginal_grids,
        meta={
            "w_spike": w_spike,
            "sigma_slab": sigma_slab,
            "expected_sparsity": w_spike,
        },
    )


# ===========================================================================
# 2. Sparse orthogonal regression
# ===========================================================================

def make_sparse_regression(
    D=20,
    n_nonzero=5,
    n_obs=100,
    noise_std=1.0,
    sigma_slab=3.0,
    seed=42,
    dtype=torch.float64,
):
    """
    y = X @ beta_true + noise,  X orthogonal,  spike-and-slab prior.

    With orthogonal X (X^T X = n * I), each coordinate decouples:
        p(beta_i | y) propto  w * delta(0)
                            + (1-w) * N(beta_i; mu_post_i, sigma_post^2)
    where
        sigma_post^2 = 1 / (n/noise^2 + 1/sigma_slab^2)
        mu_post_i    = sigma_post^2 * (n/noise^2) * betahat_i
        betahat_i    = (X^T y)_i / n

    The energy for the slab component is quadratic:
        E(beta) = 0.5 * sum_i (beta_i - mu_post_i)^2 / sigma_post^2
    """
    rng = np.random.default_rng(seed)

    # True sparse coefficients
    beta_true = np.zeros(D)
    nonzero_idx = rng.choice(D, size=n_nonzero, replace=False)
    beta_true[nonzero_idx] = rng.normal(0, sigma_slab, size=n_nonzero)

    # Orthogonal design (QR of random matrix, scaled)
    X_raw = rng.normal(size=(n_obs, D))
    Q, _ = np.linalg.qr(X_raw)
    X = Q * np.sqrt(n_obs)  # so X^T X = n_obs * I

    # Response
    y = X @ beta_true + noise_std * rng.normal(size=n_obs)

    # Posterior parameters (decoupled)
    betahat = X.T @ y / n_obs
    sigma_post2 = 1.0 / (n_obs / noise_std**2 + 1.0 / sigma_slab**2)
    mu_post = sigma_post2 * (n_obs / noise_std**2) * betahat
    sigma_post = np.sqrt(sigma_post2)

    # Torch tensors
    mu_post_t = torch.tensor(mu_post, dtype=dtype)
    prec_post = 1.0 / sigma_post2
    Sigma_inv_t = prec_post * torch.eye(D, dtype=dtype)

    def grad_target(x):
        # grad of E = 0.5 * (x - mu)^T P (x - mu)  =>  P (x - mu)
        return prec_post * (x - mu_post_t)

    # Marginals: N(mu_post_i, sigma_post^2) for the slab part
    marginal_grids = {}
    for i in range(D):
        lo = mu_post[i] - 5 * sigma_post
        hi = mu_post[i] + 5 * sigma_post
        # Include zero in the grid range
        lo = min(lo, -3 * sigma_post)
        hi = max(hi, 3 * sigma_post)
        grid = np.linspace(lo, hi, 500)
        pdf = np.exp(-0.5 * (grid - mu_post[i])**2 / sigma_post2) / (
            sigma_post * np.sqrt(2 * np.pi)
        )
        marginal_grids[i] = {
            "grid": grid,
            "pdf": pdf,
            "is_true_nonzero": i in nonzero_idx,
            "mu_post": mu_post[i],
            "sigma_post": sigma_post,
            "label": rf"$\beta_{{{i+1}}}$",
        }

    return TorchTarget(
        name=f"sparse_regression_D{D}_k{n_nonzero}",
        D=D,
        grad_target=grad_target,
        x_ref=mu_post_t,
        Sigma_inv=Sigma_inv_t,
        marginal_grids=marginal_grids,
        meta={
            "beta_true": beta_true,
            "nonzero_idx": nonzero_idx,
            "mu_post": mu_post,
            "sigma_post": sigma_post,
            "n_nonzero": n_nonzero,
            "n_obs": n_obs,
            "expected_sparsity": (D - n_nonzero) / D,
        },
    )


# ===========================================================================
# 3. Independent Laplace
# ===========================================================================

def make_laplace(D=5, lam=1.0, dtype=torch.float64):
    """
    Independent Laplace prior:  p(x_i) = (lam/2) exp(-lam |x_i|)
    Energy:  E(x) = lam * sum |x_i|

    The gradient has a sign discontinuity at zero — the sticky sampler
    handles this via freeze/thaw at the cusp.  The marginal is the
    known Laplace(0, 1/lam) density.

    Note: grad E is undefined at x_i=0.  We use the subgradient 0 there,
    which is consistent with the sticky dynamics (frozen at zero, velocity
    is zero, so the gradient value doesn't matter).
    """
    def grad_target(x):
        return lam * torch.sign(x)

    grid = np.linspace(-5.0 / lam, 5.0 / lam, 500)
    pdf = (lam / 2.0) * np.exp(-lam * np.abs(grid))

    marginal_grids = {
        i: {"grid": grid, "pdf": pdf, "label": rf"$\beta_{{{i+1}}}$"}
        for i in range(D)
    }

    return TorchTarget(
        name=f"laplace_D{D}_lam{lam}",
        D=D,
        grad_target=grad_target,
        x_ref=torch.zeros(D, dtype=dtype),
        # Reference precision: Laplace variance = 2/lam^2, so use that
        Sigma_inv=(lam**2 / 2.0) * torch.eye(D, dtype=dtype),
        marginal_grids=marginal_grids,
        meta={"lam": lam},
    )

# ===========================================================================
# 4. Different type slabs
# ===========================================================================

# # Student-t slab with nu degrees of freedom
# def make_student_t_slab(D=5, nu=3.0, dtype=torch.float64):
#     def grad_target(x):
#         # grad of -log t_nu(x) = (nu+1)/(nu + x^2) * x
#         return (nu + 1.0) * x / (nu + x**2)

# ===========================================================================
# Plotting (with spike-mass awareness)
# ===========================================================================

# def plot_sticky_marginals(target, sampler_results, zero_tol=1e-8, figname=None):
#     """
#     Overlay sample histograms against known marginals.
#     For spike-and-slab targets, also shows the fraction of exact zeros
#     vs the expected spike weight.
#     """
#     D = target.D
#     n_cols = min(D, 5)
#     n_rows = math.ceil(D / n_cols)
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
#                              squeeze=False)

#     for i in range(D):
#         ax = axes[i // n_cols, i % n_cols]
#         mg = target.marginal_grids[i]
#         ax.plot(mg["grid"], mg["pdf"], "k-", lw=2, label="exact (slab)")

#         for name, samples in sampler_results.items():
#             col = samples[:, i]
#             nonzero = col[np.abs(col) > zero_tol]

#             # Histogram of nonzero samples (density of the slab part)
#             if len(nonzero) > 10:
#                 ax.hist(nonzero, bins=80, density=True, alpha=0.35, label=f"{name}")

#             # Annotate spike fraction
#             frac_zero = np.mean(np.abs(col) <= zero_tol)
#             # Inside the per-coordinate loop, after computing frac_zero:
#             if frac_zero > 0.01:
#                 # Draw a bar at x=0 whose area represents the spike mass.
#                 # Scale its height relative to the slab density so they're
#                 # visually comparable.
#                 bin_width = (mg["grid"][-1] - mg["grid"][0]) / 80  # match hist bins
#                 spike_height = frac_zero / bin_width
#                 ax.bar(0, spike_height, width=bin_width, color="red", alpha=0.4,
#                     label=f"spike ({frac_zero:.0%})", zorder=5)
#             ax.axvline(0, color="red", ls="--", lw=0.8, alpha=0.5)
#             ax.text(
#                 0.97, 0.95, f"P(0)={frac_zero:.2f}",
#                 transform=ax.transAxes, ha="right", va="top", fontsize=7,
#                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
#             )

#         # Mark expected spike weight if available
#         if "spike_weight" in mg:
#             ax.set_title(f"{mg['label']}  (expect P(0)={mg['spike_weight']:.2f})")
#         elif "is_true_nonzero" in mg:
#             status = "nonzero" if mg["is_true_nonzero"] else "zero"
#             ax.set_title(f"{mg['label']}  (true: {status})")
#         else:
#             ax.set_title(mg.get("label", f"dim {i}"))

#         ax.legend(fontsize=6)

#     for j in range(D, n_rows * n_cols):
#         axes[j // n_cols, j % n_cols].set_visible(False)

#     fig.suptitle(target.name, fontsize=14)
#     fig.tight_layout()
#     if figname:
#         fig.savefig(figname, dpi=150)
#         print(f"Saved {figname}")
#     plt.show()

def plot_sticky_marginals(target, sampler_results, zero_tol=1e-8, figname=None):
    """
    Two figures:
      1. Marginal densities (slab part only, clean histograms)
      2. Sparsity bar chart (P(0) per coordinate)
    """
    D = target.D
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # ── Figure 1: marginal densities ──
    n_cols = min(D, 5)
    n_rows = math.ceil(D / n_cols)
    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                              squeeze=False)

    for i in range(D):
        ax = axes[i // n_cols, i % n_cols]
        mg = target.marginal_grids[i]
        ax.plot(mg["grid"], mg["pdf"], "k-", lw=2, label="exact (slab)")

        for ci, (name, samples) in enumerate(sampler_results.items()):
            col = samples[:, i]
            nonzero = col[np.abs(col) > zero_tol]
            if len(nonzero) > 10:
                ax.hist(nonzero, bins=80, density=True, alpha=0.35,
                        color=colors[ci % len(colors)], label=name)

        ax.axvline(0, color="grey", ls=":", lw=0.6, alpha=0.5)

        if "is_true_nonzero" in mg:
            status = "nonzero" if mg["is_true_nonzero"] else "zero"
            ax.set_title(f"{mg['label']}  (true: {status})")
        else:
            ax.set_title(mg.get("label", f"dim {i}"))
        ax.legend(fontsize=6)

    for j in range(D, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig1.suptitle(f"{target.name} — slab marginals", fontsize=14)
    fig1.tight_layout()

    # ── Figure 2: sparsity per coordinate ──
    fig2, ax2 = plt.subplots(figsize=(max(D * 0.5, 6), 3.5))

    x_pos = np.arange(D)
    bar_width = 0.8 / max(len(sampler_results), 1)

    for ci, (name, samples) in enumerate(sampler_results.items()):
        frac_zeros = np.array([np.mean(np.abs(samples[:, i]) <= zero_tol)
                               for i in range(D)])
        ax2.bar(x_pos + ci * bar_width, frac_zeros, width=bar_width,
                alpha=0.7, color=colors[ci % len(colors)], label=name)

    # Expected sparsity line
    n_true_zero = sum(1 for i in range(D) 
                    if not target.marginal_grids[i].get("is_true_nonzero", True))
    ax2.text(0.98, 0.92, f"True zeros: {n_true_zero}/{D}", 
            transform=ax2.transAxes, ha="right", fontsize=8,
            bbox=dict(fc="white", alpha=0.7))

    # Colour x-tick labels by true status
    labels = []
    label_colors = []
    for i in range(D):
        mg = target.marginal_grids[i]
        labels.append(mg.get("label", f"{i}"))
        if "is_true_nonzero" in mg:
            label_colors.append("tab:red" if mg["is_true_nonzero"] else "tab:blue")
        else:
            label_colors.append("black")

    ax2.set_xticks(x_pos + bar_width * (len(sampler_results) - 1) / 2)
    ax2.set_xticklabels(labels, fontsize=8)
    for tick_label, col in zip(ax2.get_xticklabels(), label_colors):
        tick_label.set_color(col)

    ax2.set_ylabel("P(frozen)")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.set_title(f"{target.name} — sparsity per coordinate "
                  r"({\color{blue}true zero} / {\color{red}true nonzero})",
                  fontsize=11)
    # Simpler title since LaTeX color doesn't work in all backends
    ax2.set_title(f"{target.name} — sparsity per coordinate\n"
                  "(blue labels = true zero, red labels = true nonzero)",
                  fontsize=11)

    fig2.tight_layout()

    if figname:
        stem = figname.rsplit(".", 1)[0] if "." in figname else figname
        ext = figname.rsplit(".", 1)[1] if "." in figname else "png"
        fig1.savefig(f"{stem}_marginals.{ext}", dpi=150)
        fig2.savefig(f"{stem}_sparsity.{ext}", dpi=150)
        print(f"Saved {stem}_marginals.{ext} and {stem}_sparsity.{ext}")

    plt.show()   
    
    
def print_sparsity_summary(target, sampler_results, zero_tol=1e-8):
    print(f"\n{'='*60}")
    print(f"Sparsity summary: {target.name}")
    print(f"{'='*60}")

    for name, samples in sampler_results.items():
        is_zero = np.abs(samples) <= zero_tol
        per_coord = is_zero.mean(axis=0)

        print(f"\n  {name}:")

        if "beta_true" in target.meta:
            beta_true = target.meta["beta_true"]
            print(f"    {'coord':<8} {'beta_true':>10} {'P(0)':>8}  {'status'}")
            print(f"    {'-'*40}")
            for i in range(target.D):
                status = "zero" if beta_true[i] == 0 else "NONZERO"
                print(f"    {i:<8} {beta_true[i]:>10.4f} {per_coord[i]:>8.3f}  {status}")
        else:
            print(f"    {'coord':<8} {'P(0)':>8}")
            print(f"    {'-'*20}")
            for i in range(target.D):
                print(f"    {i:<8} {per_coord[i]:>8.3f}")

        print(f"\n    Overall sparsity: {is_zero.mean():.3f}")