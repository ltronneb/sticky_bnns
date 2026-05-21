import numpy as np
import torch
from scipy import integrate

from .model import TorchTarget


def make_gaussian(D=5, cov="diagonal", seed=42, dtype=torch.float64):
    """
    Gaussian target E(x) = 0.5 x^T Sigma_inv x.
    The residual gradient is zero everywhere, so the Boomerang never bounces.
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
        raise ValueError(f"Unknown cov={cov!r}")

    Sigma_inv_t = torch.tensor(Sigma_inv_np, dtype=dtype)

    def grad_target(x):
        return Sigma_inv_t @ x

    marginal_grids = {}
    for i in range(D):
        sd = np.sqrt(Sigma_np[i, i])
        grid = np.linspace(-4 * sd, 4 * sd, 500)
        pdf = np.exp(-0.5 * grid**2 / Sigma_np[i, i]) / np.sqrt(2 * np.pi * Sigma_np[i, i])
        marginal_grids[i] = {"grid": grid, "pdf": pdf, "label": rf"$\beta_{{{i+1}}}$"}

    return TorchTarget(
        name=f"gaussian_{cov}_D{D}",
        D=D,
        grad_target=grad_target,
        x_ref=torch.zeros(D, dtype=dtype),
        Sigma_inv=Sigma_inv_t,
        marginal_grids=marginal_grids,
    )


def make_banana(D=2, a=1.0, scale=1.0, dtype=torch.float64):
    """
    Banana E(b0,b1) = 0.5*(b0/scale)^2 + 0.5*(b1 - a*(b0/scale)^2)^2.
    """
    if D != 2:
        raise NotImplementedError("Only D=2 supported.")

    def grad_target(beta):
        b0, b1 = beta[0], beta[1]
        u = b0 / scale
        r = b1 - a * u**2
        dE_db0 = u / scale - 2.0 * a * u / scale * r
        dE_db1 = r
        return torch.stack([dE_db0, dE_db1])

    s2 = scale**2
    Z = scale * 2 * np.pi
    grid_0 = np.linspace(-4 * scale, 4 * scale, 500)
    grid_1 = np.linspace(-4, 4 + 16 * a, 500)
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
        x_ref=torch.tensor([0.0, a], dtype=dtype),
        Sigma_inv=torch.diag(torch.tensor([1.0, 1.0 / 3.0], dtype=dtype)),
        marginal_grids=marginal_grids,
    )


def make_gaussian_mixture(D=1, preset="bimodal", dtype=torch.float64):
    """
    Independent 1-D Gaussian mixture on each coordinate.
    """
    presets = {
        "bimodal":      dict(weights=[0.5, 0.5], locs=[-3.0, 3.0],  scales=[1.0, 1.0]),
        "heavy_tailed": dict(weights=[0.9, 0.1], locs=[0.0, 0.0],   scales=[1.0, 5.0]),
        "skewed":       dict(weights=[0.7, 0.3], locs=[0.0, 3.0],   scales=[1.0, 1.5]),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset={preset!r}. Choose from {list(presets)}")
    spec = presets[preset]

    w   = torch.tensor(spec["weights"], dtype=dtype)
    mu  = torch.tensor(spec["locs"],    dtype=dtype)
    s   = torch.tensor(spec["scales"],  dtype=dtype)
    log_w = torch.log(w)
    log_s = torch.log(s)

    def grad_target(beta):
        z = (beta.unsqueeze(-1) - mu) / s          # [D, K]
        log_comp = log_w - 0.5 * z**2 - log_s      # [D, K]
        weights = torch.softmax(log_comp, dim=-1)   # [D, K]
        return (weights * z / s).sum(dim=-1)        # [D]

    w_np  = np.array(spec["weights"])
    mu_np = np.array(spec["locs"])
    s_np  = np.array(spec["scales"])
    lo = float(np.min(mu_np - 5 * s_np))
    hi = float(np.max(mu_np + 5 * s_np))
    grid = np.linspace(lo, hi, 500)
    pdf = sum(
        wi * np.exp(-0.5 * ((grid - mi) / si)**2) / (si * np.sqrt(2 * np.pi))
        for wi, mi, si in zip(w_np, mu_np, s_np)
    )

    marginal_grids = {
        i: {"grid": grid, "pdf": pdf, "label": rf"$\beta_{{{i+1}}}$"}
        for i in range(D)
    }

    return TorchTarget(
        name=f"gaussian_mixture_{preset}_D{D}",
        D=D,
        grad_target=grad_target,
        x_ref=torch.zeros(D, dtype=dtype),
        Sigma_inv=0.1 * torch.eye(D, dtype=dtype),
        marginal_grids=marginal_grids,
    )


def make_neals_funnel(D=10, sigma_v=3.0, dtype=torch.float64):
    """
    Neal's funnel: v ~ N(0, sigma_v^2), x_i | v ~ N(0, exp(v)) i.i.d. for i=1..D-1.
    """
    if D < 2:
        raise ValueError("Funnel needs D >= 2 (one v plus at least one x).")

    sv2 = sigma_v ** 2
    Dm1 = D - 1

    def grad_target(beta):
        v = beta[0]
        x = beta[1:]
        exp_neg_v = torch.exp(-v)
        sum_x2 = (x * x).sum()
        dE_dv = v / sv2 + 0.5 * Dm1 - 0.5 * exp_neg_v * sum_x2
        dE_dx = exp_neg_v * x
        return torch.cat([dE_dv.unsqueeze(0), dE_dx])

    grid_v = np.linspace(-4 * sigma_v, 4 * sigma_v, 500)
    pdf_v = np.exp(-0.5 * grid_v ** 2 / sv2) / (sigma_v * np.sqrt(2 * np.pi))

    # x_i marginal: integrate N(x_i; 0, exp(v)) against N(v; 0, sigma_v^2).
    # No closed form; use a v-quadrature.
    v_quad = np.linspace(-4 * sigma_v, 4 * sigma_v, 400)
    pv_quad = np.exp(-0.5 * v_quad ** 2 / sv2) / (sigma_v * np.sqrt(2 * np.pi))
    dv = v_quad[1] - v_quad[0]
    grid_x = np.linspace(-30.0, 30.0, 500)
    pdf_x = np.zeros_like(grid_x)
    for j, xj in enumerate(grid_x):
        cond = np.exp(-0.5 * xj ** 2 * np.exp(-v_quad)) / np.sqrt(2 * np.pi * np.exp(v_quad))
        pdf_x[j] = np.sum(cond * pv_quad) * dv

    marginal_grids = {0: {"grid": grid_v, "pdf": pdf_v, "label": r"$v$"}}
    for i in range(1, D):
        marginal_grids[i] = {"grid": grid_x, "pdf": pdf_x, "label": rf"$x_{{{i}}}$"}


    Sigma_inv = torch.eye(D, dtype=dtype)
    Sigma_inv[0, 0] = 1.0 / sv2

    return TorchTarget(
        name=f"neals_funnel_D{D}_sv{sigma_v}",
        D=D,
        grad_target=grad_target,
        x_ref=torch.zeros(D, dtype=dtype),
        Sigma_inv=Sigma_inv,
        marginal_grids=marginal_grids,
    )