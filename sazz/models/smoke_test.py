import math
import numpy as np
import torch
import matplotlib.pyplot as plt


#from samplers.AutomaticBoomerangSampler import AutomaticBoomerangSampler

# ===========================================================================
# Torch-native targets
# ===========================================================================

class TorchTarget:
    """Minimal container matching what the sampler needs."""
    def __init__(self, name, D, grad_target, x_ref, Sigma_inv, marginal_grids=None):
        self.name = name
        self.D = D
        self.grad_target = grad_target        # callable: Tensor -> Tensor
        self.x_ref = x_ref                    # Tensor [D]
        self.Sigma_inv = Sigma_inv            # Tensor [D,D]
        self.marginal_grids = marginal_grids  # dict[int, {grid, pdf, label}]


# ---- 1. Multivariate Gaussian -------------------------------------------

def make_gaussian(D=5, cov="diagonal", seed=42, dtype=torch.float64):
    """
    Gaussian target  E(x) = 0.5 x^T Sigma_inv x.
    grad E = Sigma_inv @ x.
    """
    rng = np.random.default_rng(seed)

    if cov == "diagonal":
        diag_prec = rng.uniform(0.5, 3.0, size=D)
        Sigma_inv_np = np.diag(diag_prec)
        Sigma_np = np.diag(1.0 / diag_prec)
    elif cov == "ar1":
        rho = 0.9
        idx = np.arange(D)
        Sigma_np = rho ** np.abs(idx[:, None] - idx[None, :])
        Sigma_inv_np = np.linalg.inv(Sigma_np)
    elif cov == "random":
        A = rng.normal(size=(D, D))
        Sigma_np = A @ A.T / D + 0.1 * np.eye(D)
        d = np.sqrt(np.diag(Sigma_np))
        Sigma_np = Sigma_np / np.outer(d, d)
        Sigma_inv_np = np.linalg.inv(Sigma_np)
    else:
        raise ValueError(f"Unknown cov={cov}")

    Sigma_inv_t = torch.tensor(Sigma_inv_np, dtype=dtype)

    def grad_target(x):
        return Sigma_inv_t @ x

    # Marginals
    marginal_grids = {}
    for i in range(D):
        sd = np.sqrt(Sigma_np[i, i])
        grid = np.linspace(-4 * sd, 4 * sd, 500)
        pdf = np.exp(-0.5 * grid**2 / Sigma_np[i, i]) / np.sqrt(
            2 * np.pi * Sigma_np[i, i]
        )
        marginal_grids[i] = {"grid": grid, "pdf": pdf, "label": rf"$\beta_{{{i+1}}}$"}

    return TorchTarget(
        name=f"gaussian_{cov}_D{D}",
        D=D,
        grad_target=grad_target,
        x_ref=torch.zeros(D, dtype=dtype),
        Sigma_inv=Sigma_inv_t,
        marginal_grids=marginal_grids,
    )


# ---- 2. Rosenbrock banana ------------------------------------------------

def make_banana(D=2, a=1.0, scale=1.0, dtype=torch.float64):
    """
    Banana  E(b0,b1) = 0.5*(b0/s)^2 + 0.5*(b1 - a*(b0/s)^2)^2.
    Hand-coded gradient (no autograd needed for 2-D).
    """
    if D != 2:
        raise NotImplementedError("Only D=2 for now; extend for stacked.")

    s2 = scale ** 2

    def grad_target(beta):
        b0, b1 = beta[0], beta[1]
        u = b0 / scale
        r = b1 - a * u ** 2
        dE_db0 = u / scale - 2.0 * a * u / scale * r
        dE_db1 = r
        return torch.stack([dE_db0, dE_db1])

    # Marginals (same numerical integration as your validation.py)
    from scipy import integrate

    grid_0 = np.linspace(-4 * scale, 4 * scale, 500)
    grid_1 = np.linspace(-4, 4 + 16 * a, 500)

    Z = scale * 2 * np.pi
    marg_0 = np.exp(-0.5 * grid_0**2 / s2) / (scale * np.sqrt(2 * np.pi))

    def unnorm_joint(b0, b1):
        u = b0 / scale
        return np.exp(-0.5 * u**2 - 0.5 * (b1 - a * u**2)**2)

    marg_1 = np.zeros_like(grid_1)
    b0_lim = 8 * scale
    for i, b1_val in enumerate(grid_1):
        val, _ = integrate.quad(lambda b0: unnorm_joint(b0, b1_val), -b0_lim, b0_lim)
        marg_1[i] = val / Z

    marginal_grids = {
        0: {"grid": grid_0, "pdf": marg_0, "label": r"$\beta_1$"},
        1: {"grid": grid_1, "pdf": marg_1, "label": r"$\beta_2$"},
    }

    return TorchTarget(
        name=f"banana_D{D}_a{a}_s{scale}",
        D=D,
        grad_target=grad_target,
        # moment-matched reference (same as your notebook)
        x_ref=torch.tensor([0.0, a], dtype=dtype),
        Sigma_inv=torch.diag(torch.tensor([1.0, 1.0 / 3.0], dtype=dtype)),
        marginal_grids=marginal_grids,
    )


# ---- 3. Gaussian mixture (1-D) ------------------------------------------

def make_gaussian_mixture(D=1, preset="bimodal", dtype=torch.float64):
    """
    Independent 1-D Gaussian mixture on each coordinate.
    E = -sum_i logsumexp_k [ log w_k - 0.5*z_ik^2 - log s_k ]
    """
    presets = {
        "bimodal":      dict(weights=[0.5, 0.5], locs=[-3.0, 3.0], scales=[1.0, 1.0]),
        "heavy_tailed": dict(weights=[0.9, 0.1], locs=[0.0, 0.0], scales=[1.0, 5.0]),
        "skewed":       dict(weights=[0.7, 0.3], locs=[0.0, 3.0], scales=[1.0, 1.5]),
    }
    spec = presets[preset]
    w = torch.tensor(spec["weights"], dtype=dtype)
    mu = torch.tensor(spec["locs"], dtype=dtype)
    s = torch.tensor(spec["scales"], dtype=dtype)
    log_w = torch.log(w)
    log_s = torch.log(s)

    def grad_target(beta):
        """
        grad E  = -d/dbeta [ sum_i logsumexp_k (...) ]
        For independent coords this is just the per-coord mixture gradient.
        """
        # beta: [D],  mu: [K]
        z = (beta.unsqueeze(-1) - mu) / s                   # [D, K]
        log_comp = log_w - 0.5 * z**2 - log_s               # [D, K]
        # softmax weights
        weights = torch.softmax(log_comp, dim=-1)            # [D, K]
        # d/dbeta_i of -logsumexp = sum_k w_k * z_ik / s_k
        grad_per_coord = (weights * z / s).sum(dim=-1)       # [D]
        return grad_per_coord

    # Marginal PDF for plotting
    w_np = np.array(spec["weights"])
    mu_np = np.array(spec["locs"])
    s_np = np.array(spec["scales"])
    lo = float(np.min(mu_np - 5 * s_np))
    hi = float(np.max(mu_np + 5 * s_np))
    grid = np.linspace(lo, hi, 500)
    pdf = np.zeros_like(grid)
    for wi, mi, si in zip(w_np, mu_np, s_np):
        pdf += wi * np.exp(-0.5 * ((grid - mi) / si) ** 2) / (si * np.sqrt(2 * np.pi))

    marginal_grids = {
        i: {"grid": grid, "pdf": pdf, "label": rf"$\beta_{{{i+1}}}$"}
        for i in range(D)
    }

    return TorchTarget(
        name=f"gaussian_mixture_{preset}_D{D}",
        D=D,
        grad_target=grad_target,
        # Centre between modes, overdispersed reference
        x_ref=torch.zeros(D, dtype=dtype),
        Sigma_inv=0.1 * torch.eye(D, dtype=dtype),
        marginal_grids=marginal_grids,
    )


# ===========================================================================
# Plotting
# ===========================================================================

def plot_marginals(target, sampler_results, figname=None):
    """
    Overlay sample histograms against known marginal PDFs.

    sampler_results: dict[str, np.ndarray]  name -> resampled samples [N, D]
    """
    D = target.D
    n_cols = min(D, 5)
    n_rows = math.ceil(D / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for i in range(D):
        ax = axes[i // n_cols, i % n_cols]
        mg = target.marginal_grids[i]
        ax.plot(mg["grid"], mg["pdf"], "k-", lw=2, label="exact")

        for name, samples in sampler_results.items():
            ax.hist(samples[:, i], bins=120, density=True, alpha=0.4, label=name)

        ax.set_title(mg.get("label", f"dim {i}"))
        ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(D, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig.suptitle(target.name, fontsize=14)
    fig.tight_layout()
    if figname:
        fig.savefig(figname, dpi=150)
        print(f"Saved {figname}")
    plt.show()
