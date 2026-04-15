"""
Validation targets with known marginals.

Three factories:
  - beta_binomial(D, ...)   : D independent Beta-Binomial posteriors on log-odds
  - neals_funnel(D, ...)    : Neal's funnel distribution
  - rosenbrock_banana(a=1)  : 2-D banana-shaped distribution

Each target carries a `marginal_pdf` dict in target.meta so that downstream
plotting can overlay the true density on sample histograms.
"""
import numpy as np
import autograd.numpy as anp
from autograd import grad
from scipy import special, integrate

from .base import Target

# =====================================================================
# 0. Gaussian (reference measure sanity check)
# =====================================================================

def gaussian_sanity_check(D=5, seed=42):
    """
    Multivariate Gaussian target where E(beta) = 0.5 * beta^T Sigma_inv beta,
    i.e. the target IS the reference measure.

    If the sampler is correct, U(beta) = 0 everywhere, the switching rate is
    identically zero, and no bounces should occur. All skeleton points should
    come from velocity refreshments alone.
    """
    rng = np.random.default_rng(seed)
    # Random diagonal precision so it's not just the identity
    diag_prec = rng.uniform(0.5, 3.0, size=D)
    Sigma_inv = np.diag(diag_prec)
    Sigma_diag = 1.0 / diag_prec

    def E(beta):
        return 0.5 * anp.sum(anp.array(diag_prec) * beta**2)

    gradE_fn = grad(E)

    # Marginals are independent normals
    marginal_grids = {}
    for i in range(D):
        sd = np.sqrt(Sigma_diag[i])
        grid = np.linspace(-4 * sd, 4 * sd, 500)
        pdf = np.exp(-0.5 * grid**2 / Sigma_diag[i]) / np.sqrt(2 * np.pi * Sigma_diag[i])
        marginal_grids[i] = {"grid": grid, "pdf": pdf, "label": f"$\\beta_{{{i+1}}}$"}

    return Target(
        name=f"gaussian_refcheck_D{D}",
        task_type="validation",
        D=D,
        E=E,
        gradE=gradE_fn,
        x_ref=np.zeros(D),
        Sigma_inv=Sigma_inv,
        meta={
            "Sigma_diag": Sigma_diag,
            "marginal_grids": marginal_grids,
            "preprocess_method": "manual",
        },
    )

# =====================================================================
# 1. Beta-Binomial on log-odds scale
# =====================================================================

def _bb_energy(a_post, b_post):
    a_post = anp.array(a_post, dtype=float)
    b_post = anp.array(b_post, dtype=float)

    def E(theta):
        return anp.sum(
            a_post * anp.log(1 + anp.exp(-theta))
            + b_post * theta
            + b_post * anp.log(1 + anp.exp(-theta))
        )
    return E


def _bb_marginal_pdf(theta, a_j, b_j):
    sigma = 1.0 / (1.0 + np.exp(-theta))
    return sigma**a_j * (1 - sigma)**b_j / special.beta(a_j, b_j)


def beta_binomial(D=5, n_obs=None, x_obs=None, a_prior=2.0, b_prior=2.0, seed=42):
    """
    D independent Beta-Binomial posteriors, parameterized on the log-odds scale.

    If n_obs/x_obs are not provided, synthetic data is generated.
    """
    if n_obs is None or x_obs is None:
        rng = np.random.default_rng(seed)
        n_obs = rng.integers(20, 80, size=D)
        p_true = rng.beta(a_prior, b_prior, size=D)
        x_obs = rng.binomial(n_obs, p_true)

    n_obs = np.asarray(n_obs)
    x_obs = np.asarray(x_obs)
    a_posterior = np.full(D, a_prior) + x_obs
    b_posterior = np.full(D, b_prior) + (n_obs - x_obs)

    E = _bb_energy(a_posterior, b_posterior)
    gradE = grad(E)

    # Pre-compute marginal pdfs for each coordinate
    marginal_grids = {}
    for i in range(D):
        grid = np.linspace(-5, 7, 500)
        pdf = _bb_marginal_pdf(grid, a_posterior[i], b_posterior[i])
        marginal_grids[i] = {"grid": grid, "pdf": pdf, "label": f"$\\theta_{{{i+1}}}$"}

    return Target(
        name=f"beta_binomial_D{D}",
        task_type="validation",
        D=D,
        E=E,
        gradE=gradE,
        data={"n_obs": n_obs, "x_obs": x_obs},
        meta={
            "a_posterior": a_posterior,
            "b_posterior": b_posterior,
            "marginal_grids": marginal_grids,
        },
    )


# =====================================================================
# 2. Neal's Funnel
# =====================================================================

def _funnel_energy(sigma_v, D):
    def E(theta):
        v = theta[0]
        x = theta[1:]
        nll_v = 0.5 * v**2 / sigma_v**2 + 0.5 * (D - 1) * v
        nll_x = anp.sum(0.5 * x**2 * anp.exp(-v))
        return nll_v + nll_x
    return E


def _funnel_marginal_v(v_val, sigma_v):
    return np.exp(-0.5 * v_val**2 / sigma_v**2) / np.sqrt(2 * np.pi * sigma_v**2)


def _funnel_marginal_x(x_val, sigma_v):
    def integrand(v):
        log_p_v = -0.5 * v**2 / sigma_v**2 - 0.5 * np.log(2 * np.pi * sigma_v**2)
        log_p_x = -0.5 * x_val**2 * np.exp(-v) - 0.5 * v - 0.5 * np.log(2 * np.pi)
        return np.exp(log_p_v + log_p_x)
    result, _ = integrate.quad(integrand, -10 * sigma_v, 10 * sigma_v)
    return result


def neals_funnel(D=4, sigma_v=3.0):
    """
    Neal's funnel: v ~ N(0, sigma_v^2), x_i | v ~ N(0, exp(v)).

    The funnel requires manual preprocessing because the geometry is
    far from Gaussian. x_ref and Sigma_inv are provided in the target.
    """
    E = _funnel_energy(sigma_v, D)
    gradE = grad(E)

    # Reference: center at zero, match marginal variances
    x_ref = np.zeros(D)
    Sigma_inv = np.eye(D)
    Sigma_inv[0, 0] = 1.0 / sigma_v**2

    # Pre-compute marginal pdfs
    grid_v = np.linspace(-12, 12, 500)
    pdf_v = _funnel_marginal_v(grid_v, sigma_v)

    grid_x = np.linspace(-30, 30, 500)
    pdf_x = np.array([_funnel_marginal_x(xi, sigma_v) for xi in grid_x])

    marginal_grids = {
        0: {"grid": grid_v, "pdf": pdf_v, "label": "$v$"},
    }
    for i in range(1, D):
        marginal_grids[i] = {"grid": grid_x, "pdf": pdf_x, "label": f"$x_{{{i}}}$"}

    return Target(
        name=f"neals_funnel_D{D}",
        task_type="validation",
        D=D,
        E=E,
        gradE=gradE,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={
            "sigma_v": sigma_v,
            "marginal_grids": marginal_grids,
            "preprocess_method": "manual",
        },
    )


# =====================================================================
# 3. Rosenbrock Banana
# =====================================================================

def _banana_energy(a):
    def E(beta):
        return 0.5 * (beta[0]**2 + (beta[1] - a * beta[0]**2)**2)
    return E


def _banana_marginals(a, grid_0, grid_1):
    def unnorm_joint(b0, b1):
        return np.exp(-0.5 * (b0**2 + (b1 - a * b0**2)**2))

    Z, _ = integrate.dblquad(unnorm_joint, -8, 8, -8, 8)

    marginal_0 = np.zeros_like(grid_0)
    for i, b0 in enumerate(grid_0):
        val, _ = integrate.quad(lambda b1: unnorm_joint(b0, b1), -8, 8)
        marginal_0[i] = val / Z

    marginal_1 = np.zeros_like(grid_1)
    for i, b1 in enumerate(grid_1):
        val, _ = integrate.quad(lambda b0: unnorm_joint(b0, b1), -8, 8)
        marginal_1[i] = val / Z

    return marginal_0, marginal_1


def rosenbrock_banana(a=1.0):
    """
    2-D Rosenbrock banana: E(x1, x2) = 0.5 * x1^2 + 0.5 * (x2 - a*x1^2)^2.

    x1 marginal is N(0,1). x2 marginal requires numerical integration.
    """
    E = _banana_energy(a)
    gradE = grad(E)

    grid_0 = np.linspace(-4, 4, 300)
    grid_1 = np.linspace(-2, 6, 300)
    marg_0, marg_1 = _banana_marginals(a, grid_0, grid_1)

    marginal_grids = {
        0: {"grid": grid_0, "pdf": marg_0, "label": "$x_1$"},
        1: {"grid": grid_1, "pdf": marg_1, "label": "$x_2$"},
    }

    return Target(
        name=f"rosenbrock_banana_a{a}",
        task_type="validation",
        D=2,
        E=E,
        gradE=gradE,
        meta={
            "a": a,
            "marginal_grids": marginal_grids,
        },
    )


# =====================================================================
# Registry
# =====================================================================

VALIDATION_TARGETS = {
    "gaussian": gaussian_sanity_check,
    "beta_binomial": beta_binomial,
    "neals_funnel": neals_funnel,
    "rosenbrock_banana": rosenbrock_banana,
}
