"""
Continuous validation targets with known marginals.

Three factories, each parameterized to cover multiple stress regimes:
  - gaussian(D, cov=...)         : diagonal | AR(1) | random-PD covariance
  - gaussian_mixture(D, preset=..): bimodal | heavy-tailed | skewed 1-D marginals
  - banana(D, a=..., stacked=...) : 2-D Rosenbrock or stacked bananas

Each target carries a `marginal_grids` dict in target.meta for plot overlays.
"""
import numpy as np
import autograd.numpy as anp
from autograd import grad
from scipy import integrate

from .base import Target


# =====================================================================
# 1. Gaussian (diagonal / correlated)
# =====================================================================

def _build_covariance(D, cov, seed):
    """Return (Sigma, Sigma_inv) for the requested covariance structure."""
    rng = np.random.default_rng(seed)

    if cov == "diagonal":
        diag_prec = rng.uniform(0.5, 3.0, size=D)
        Sigma = np.diag(1.0 / diag_prec)
        Sigma_inv = np.diag(diag_prec)

    elif cov == "ar1":
        # AR(1) with rho = 0.9, unit marginal variance
        rho = 0.9
        idx = np.arange(D)
        Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
        Sigma_inv = np.linalg.inv(Sigma)

    elif cov == "random":
        # Random positive-definite; normalise to ~unit marginal variance
        A = rng.normal(size=(D, D))
        Sigma = A @ A.T / D + 0.1 * np.eye(D)
        d = np.sqrt(np.diag(Sigma))
        Sigma = Sigma / np.outer(d, d)
        Sigma_inv = np.linalg.inv(Sigma)

    else:
        raise ValueError(f"Unknown cov: {cov!r}. Use 'diagonal' | 'ar1' | 'random'.")

    return Sigma, Sigma_inv


def gaussian(D=5, cov="diagonal", seed=42):
    """
    Multivariate Gaussian target E(beta) = 0.5 * beta^T Sigma_inv beta.

    cov:
      "diagonal" : independent coords (reference-measure sanity check when
                   the sampler uses Sigma_inv as the reference measure).
      "ar1"      : AR(1) correlation, rho=0.9. Tests reflection + preconditioning.
      "random"   : random positive-definite with moderate off-diagonal mass.

    Marginals are N(0, Sigma_ii) for every coordinate.
    """
    Sigma, Sigma_inv = _build_covariance(D, cov, seed)
    Sigma_inv_anp = anp.array(Sigma_inv)

    def E(beta):
        return 0.5 * beta @ Sigma_inv_anp @ beta

    gradE_fn = grad(E)

    marginal_grids = {}
    for i in range(D):
        sd = np.sqrt(Sigma[i, i])
        grid = np.linspace(-4 * sd, 4 * sd, 500)
        pdf = np.exp(-0.5 * grid**2 / Sigma[i, i]) / np.sqrt(2 * np.pi * Sigma[i, i])
        marginal_grids[i] = {"grid": grid, "pdf": pdf, "label": f"$\\beta_{{{i+1}}}$"}

    return Target(
        name=f"gaussian_{cov}_D{D}",
        task_type="validation",
        D=D,
        E=E,
        gradE=gradE_fn,
        x_ref=np.zeros(D),
        Sigma_inv=Sigma_inv,
        meta={
            "cov_kind": cov,
            "Sigma": Sigma,
            "marginal_grids": marginal_grids,
            "preprocess_method": "manual",
        },
    )


# =====================================================================
# 2. Gaussian mixture (multimodal / heavy-tailed / skewed)
# =====================================================================

# Each preset specifies a 1-D mixture applied independently to every coord.
_MIXTURE_PRESETS = {
    # Two well-separated modes at +/- 3, equal weight.
    "bimodal":      dict(weights=[0.5, 0.5],
                         locs=[-3.0, 3.0],
                         scales=[1.0, 1.0]),
    # Narrow core + wide component: smooth stand-in for heavy tails.
    "heavy_tailed": dict(weights=[0.9, 0.1],
                         locs=[0.0, 0.0],
                         scales=[1.0, 5.0]),
    # Asymmetric mixture: skewed marginal with nonzero mean.
    "skewed":       dict(weights=[0.7, 0.3],
                         locs=[0.0, 3.0],
                         scales=[1.0, 1.5]),
}


def _mixture_energy(weights, locs, scales):
    """E = -sum_i logsumexp_k (log w_k - 0.5*z_ik^2 - log s_k)."""
    log_w = anp.log(anp.array(weights))
    locs_a = anp.array(locs)
    scales_a = anp.array(scales)
    log_scales = anp.log(scales_a)

    def E(beta):
        beta_col = beta[:, None]
        z = (beta_col - locs_a[None, :]) / scales_a[None, :]
        log_comp = log_w[None, :] - 0.5 * z**2 - log_scales[None, :]
        m = anp.max(log_comp, axis=1, keepdims=True)
        lse = m.squeeze(1) + anp.log(anp.sum(anp.exp(log_comp - m), axis=1))
        return -anp.sum(lse)

    return E


def _mixture_marginal_pdf(grid, weights, locs, scales):
    pdf = np.zeros_like(grid)
    for w, mu, s in zip(weights, locs, scales):
        pdf += w * np.exp(-0.5 * (grid - mu)**2 / s**2) / (s * np.sqrt(2 * np.pi))
    return pdf


def gaussian_mixture(D=2, preset="bimodal", weights=None, locs=None, scales=None):
    """
    Independent mixture-of-Gaussians marginals, identical across coordinates.

    Use a preset or pass explicit (weights, locs, scales) 1-D arrays.

    presets: "bimodal", "heavy_tailed", "skewed".
    """
    if weights is None:
        if preset not in _MIXTURE_PRESETS:
            raise ValueError(f"Unknown preset: {preset!r}. "
                             f"Known: {list(_MIXTURE_PRESETS)}")
        spec = _MIXTURE_PRESETS[preset]
        weights, locs, scales = spec["weights"], spec["locs"], spec["scales"]

    weights = np.asarray(weights, dtype=float)
    locs = np.asarray(locs, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Mixture weights must sum to 1.")

    E = _mixture_energy(weights, locs, scales)
    gradE_fn = grad(E)

    lo = float(np.min(locs - 5 * scales))
    hi = float(np.max(locs + 5 * scales))
    grid = np.linspace(lo, hi, 500)
    pdf = _mixture_marginal_pdf(grid, weights, locs, scales)
    marginal_grids = {
        i: {"grid": grid, "pdf": pdf, "label": f"$\\beta_{{{i+1}}}$"}
        for i in range(D)
    }

    return Target(
        name=f"gaussian_mixture_{preset}_D{D}",
        task_type="validation",
        D=D,
        E=E,
        gradE=gradE_fn,
        meta={
            "preset": preset,
            "weights": weights,
            "locs": locs,
            "scales": scales,
            "marginal_grids": marginal_grids,
        },
    )


# =====================================================================
# 3. Rosenbrock banana (2-D or stacked to higher D)
# =====================================================================
 
def _banana_energy_stacked(a, scale, D):
    """
    D=2 : classic 2-D banana, with b0-axis rescaled by `scale`.
    D>2 : stacked independent 2-D bananas on pairs (b_1,b_2), (b_3,b_4), ...
          (requires even D).
 
    E(b0, b1) = 0.5 * (b0 / scale)^2 + 0.5 * (b1 - a * (b0 / scale)^2)^2
    """
    s2 = scale**2
 
    if D == 2:
        def E(beta):
            u = beta[0] / scale
            return 0.5 * u**2 + 0.5 * (beta[1] - a * u**2)**2
    else:
        if D % 2 != 0:
            raise ValueError("Stacked banana requires even D.")
 
        def E(beta):
            u = beta[0::2] / scale
            b1 = beta[1::2]
            return 0.5 * anp.sum(u**2 + (b1 - a * u**2)**2)
 
    return E
 
def _banana_marginals_2d(a, scale, grid_0, grid_1):
    """
    With E(b0, b1) = 0.5 (b0/s)^2 + 0.5 (b1 - a (b0/s)^2)^2, the joint is
        p(b0, b1) = p(b0) * N(b1 | a*(b0/s)^2, 1),   p(b0) = N(0, s^2).
    So Z = scale * 2*pi exactly, and the b0 marginal IS N(0, s^2). Only the
    b1 marginal needs numerical integration.
    """
    Z = scale * 2 * np.pi
 
    # b0 marginal is analytic: N(0, scale^2)
    marginal_0 = np.exp(-0.5 * grid_0**2 / scale**2) / (scale * np.sqrt(2 * np.pi))
 
    # b1 marginal: integrate out b0 numerically. Integrand decays like a
    # Gaussian in b0 (width = scale), so [-8*scale, 8*scale] is safe.
    b0_lim = 8 * scale
 
    def unnorm_joint(b0, b1):
        u = b0 / scale
        return np.exp(-0.5 * u**2 - 0.5 * (b1 - a * u**2)**2)
 
    marginal_1 = np.zeros_like(grid_1)
    for i, b1 in enumerate(grid_1):
        val, _ = integrate.quad(lambda b0: unnorm_joint(b0, b1), -b0_lim, b0_lim)
        marginal_1[i] = val / Z
 
    return marginal_0, marginal_1
 
def banana(D=2, a=1.0, scale=1.0):
    """
    Rosenbrock banana with anisotropic rescaling of the b0 axis:
        E(b0, b1) = 0.5 * (b0 / scale)^2 + 0.5 * (b1 - a * (b0 / scale)^2)^2.
 
    - `a`     : curvature. Larger => tighter bend.
    - `scale` : standard deviation of the b0 marginal. Combined with `a` this
                gives an independent knob on curvature + scale mismatch.
 
    D=2 gives the classic 2-D banana; even D>2 gives stacked independent
    bananas on coordinate pairs. Each pair has the same two marginals
    (b0-type on even indices, b1-type on odd).
    """
    E = _banana_energy_stacked(a, scale, D)
    gradE_fn = grad(E)
 
    grid_0 = np.linspace(-4 * scale, 4 * scale, 500)
    # b1 support: conditional mean a*(b0/s)^2 ranges from 0 to ~16a over b0
    # grid, plus N(0,1) fluctuation. The right tail needs the full envelope.
    grid_1 = np.linspace(-4, 4 + 16 * a, 500)
    marg_0, marg_1 = _banana_marginals_2d(a, scale, grid_0, grid_1)
 
    marginal_grids = {}
    for i in range(D):
        if i % 2 == 0:
            marginal_grids[i] = {"grid": grid_0, "pdf": marg_0,
                                 "label": f"$\\beta_{{{i+1}}}$"}
        else:
            marginal_grids[i] = {"grid": grid_1, "pdf": marg_1,
                                 "label": f"$\\beta_{{{i+1}}}$"}
 
    return Target(
        name=f"banana_D{D}_a{a}_s{scale}",
        task_type="validation",
        D=D,
        E=E,
        gradE=gradE_fn,
        x_ref=np.array([0.0, 1.0]),
        Sigma_inv = np.diag([1.0, 1.0/3.0]),
        meta={
            "a": a,
            "scale": scale,
            "marginal_grids": marginal_grids,
        },
    )

# =====================================================================
# Registry
# =====================================================================

VALIDATION_TARGETS = {
    "gaussian":         gaussian,
    "gaussian_mixture": gaussian_mixture,
    "banana":           banana,
}
