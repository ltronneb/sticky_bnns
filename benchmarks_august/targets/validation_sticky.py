"""
Sticky-sampler validation targets.

Sticky PDMP samplers impose a spike-and-slab posterior:
    pi(beta_i) = w_i * delta_0 + (1 - w_i) * slab_i(beta_i)
The slab density is the continuous target here; the spike at zero is produced
by the sticky dynamics with freezing rate kappa_i.

Three factories:
  - spike_slab_gaussian(D, cov=..., spike_weights=...)
        Gaussian slab (diagonal / AR1 / random). Closed-form conditional
        slab marginals given any frozen pattern.
  - spike_slab_linreg(n, p, sparsity=..., tau=..., spike_weight=...)
        Conjugate Bayesian linear regression with spike-and-slab prior.
        Exact marginal inclusion probabilities via enumeration (p <= 15).
  - spike_slab_logreg(n, p, sparsity=..., prior=...)
        Bayesian logistic regression with Gaussian slab + sticky spike.
        Validated via posterior contraction (frozen-fraction vs true sparsity).
"""
import numpy as np
import autograd.numpy as anp
from autograd import grad
from itertools import product

from .base import Target
from .validation import _build_covariance
from .logreg import _logreg_likelihood
from .priors import build_prior, combine_likelihood_and_prior


# =====================================================================
# Helpers
# =====================================================================

def _kappa_from_weights(spike_weights, slab_scales):
    """
    Standard spike-and-slab -> kappa conversion for a Gaussian slab whose
    density height at zero is 1/(s * sqrt(2*pi)):
        kappa_i = (w_i / (1 - w_i)) / (s_i * sqrt(2*pi)).
    """
    w = np.asarray(spike_weights, dtype=float)
    s = np.asarray(slab_scales, dtype=float)
    return (w / (1 - w)) / (s * np.sqrt(2 * np.pi))


def _default_spike_weights(D):
    """Spread of sparsity levels across coordinates."""
    return np.linspace(0.8, 0.2, D)


# =====================================================================
# 1. Spike-and-slab Gaussian (diagonal / correlated)
# =====================================================================

def spike_slab_gaussian(D=5, cov="diagonal", spike_weights=None, seed=42):
    """
    Gaussian slab with spike-and-slab prior. Reuses the covariance builders
    from validation.gaussian.

    Validation checks:
      - Frozen-fraction of coord i should equal spike_weight w_i.
      - Conditional slab marginal on active coords A given frozen set S:
            N(0, Sigma_AA - Sigma_AS Sigma_SS^{-1} Sigma_SA).
        Available via meta['conditional_marginal'](active_idx, frozen_idx).

    kappa is stored in meta['kappa'].
    """
    Sigma, Sigma_inv = _build_covariance(D, cov, seed)
    slab_scales = np.sqrt(np.diag(Sigma))

    if spike_weights is None:
        spike_weights = _default_spike_weights(D)
    w = np.asarray(spike_weights[:D], dtype=float)
    kappa = _kappa_from_weights(w, slab_scales)

    Sigma_inv_anp = anp.array(Sigma_inv)

    def E(beta):
        return 0.5 * beta @ Sigma_inv_anp @ beta

    gradE_fn = grad(E)

    marginal_grids = {}
    for i in range(D):
        sd = slab_scales[i]
        grid = np.linspace(-4 * sd, 4 * sd, 500)
        pdf = np.exp(-0.5 * grid**2 / sd**2) / (sd * np.sqrt(2 * np.pi))
        marginal_grids[i] = {
            "grid": grid,
            "pdf": pdf,
            "label": f"$\\beta_{{{i+1}}}$",
            "spike_weight": float(w[i]),
        }

    def conditional_marginal(active_idx, frozen_idx):
        """
        Closed-form conditional covariance of beta_A given beta_S = 0.
        Returns Sigma_cond of shape (|A|, |A|).
        """
        A = np.asarray(active_idx, dtype=int)
        S = np.asarray(frozen_idx, dtype=int)
        Sigma_AA = Sigma[np.ix_(A, A)]
        if len(S) == 0:
            return Sigma_AA
        Sigma_AS = Sigma[np.ix_(A, S)]
        Sigma_SS = Sigma[np.ix_(S, S)]
        return Sigma_AA - Sigma_AS @ np.linalg.solve(Sigma_SS, Sigma_AS.T)

    return Target(
        name=f"spike_slab_gaussian_{cov}_D{D}",
        task_type="validation",
        D=D,
        E=E,
        gradE=gradE_fn,
        x_ref=np.zeros(D),
        Sigma_inv=Sigma_inv,
        meta={
            "cov_kind": cov,
            "Sigma": Sigma,
            "slab_scales": slab_scales,
            "spike_weights": w,
            "kappa": kappa,
            "marginal_grids": marginal_grids,
            "conditional_marginal": conditional_marginal,
            "preprocess_method": "manual",
        },
    )


# =====================================================================
# 2. Spike-and-slab linear regression (conjugate, exact inclusion probs)
# =====================================================================

def _linreg_log_marginal(X_gamma, y, tau2, sigma2):
    """
    Log marginal likelihood of the Gaussian linear model with active design
    X_gamma (n x k) and Gaussian slab beta_gamma ~ N(0, tau2 I_k):
        y | gamma ~ N(0, sigma2 I + tau2 X_gamma X_gamma^T).

    Uses Woodbury to avoid the n x n inverse. With
        M = (1/sigma2) X^T X + (1/tau2) I_k,
    we have
        (sigma2 I + tau2 X X^T)^{-1} = (1/sigma2) I - (1/sigma2^2) X M^{-1} X^T,
        log|sigma2 I + tau2 X X^T| = n log sigma2 + log|I_k + (tau2/sigma2) X^T X|.
    """
    n = len(y)
    k = X_gamma.shape[1]

    if k == 0:
        return -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * (y @ y) / sigma2

    XtX = X_gamma.T @ X_gamma
    M = XtX / sigma2 + np.eye(k) / tau2
    Xty = X_gamma.T @ y
    # z = M^{-1} (Xty / sigma2)
    rhs = Xty / sigma2
    z = np.linalg.solve(M, rhs)

    quad = (y @ y) / sigma2 - (Xty @ z)

    eigs = np.linalg.eigvalsh(XtX)
    logdet_Ik = np.sum(np.log1p((tau2 / sigma2) * eigs))
    logdet = n * np.log(sigma2) + logdet_Ik

    return -0.5 * n * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * quad


def spike_slab_linreg(n=100, p=8, sparsity="sparse", tau=1.0, sigma=1.0,
                      spike_weight=0.5, seed=1):
    """
    Conjugate Bayesian linear regression with spike-and-slab prior.

    Likelihood:  y = X beta + eps,  eps ~ N(0, sigma^2 I).
    Prior:       beta_i ~ w * delta_0 + (1 - w) * N(0, tau^2),   iid.

    Exact marginal inclusion probabilities
        P(beta_i != 0 | y)
    are computed by enumerating all 2^p models (requires p <= 15).
    Stored in meta['inclusion_probs']; the sampler's frozen-fraction of
    coord i should match (1 - inclusion_probs[i]).

    Energy exposed to the sampler is the slab density under the "all active"
    configuration; the sticky mechanism produces the spike via kappa.
    """
    if p > 15:
        raise ValueError("Exact enumeration requires p <= 15.")

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))

    if sparsity == "sparse":
        beta_true = np.zeros(p)
        beta_true[:min(3, p)] = np.array([2.0, -1.5, 0.5])[:min(3, p)]
    elif sparsity == "dense":
        beta_true = rng.normal(0, 1.0, size=p)
    else:
        raise ValueError(f"Unknown sparsity: {sparsity!r}")

    y = X @ beta_true + rng.normal(0, sigma, size=n)

    tau2, sigma2 = tau**2, sigma**2
    log_w = np.log(spike_weight)
    log_1mw = np.log(1 - spike_weight)

    gammas = np.array(list(product([0, 1], repeat=p)), dtype=bool)
    log_post = np.empty(len(gammas))
    for k, gamma in enumerate(gammas):
        X_g = X[:, gamma]
        # prior: (1-w) for each active coord, w for each inactive
        log_prior = log_1mw * gamma.sum() + log_w * (p - gamma.sum())
        log_lik = _linreg_log_marginal(X_g, y, tau2, sigma2)
        log_post[k] = log_prior + log_lik

    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= post.sum()
    inclusion_probs = np.array([post[gammas[:, i]].sum() for i in range(p)])

    # Slab energy: 0.5 ||y - X beta||^2 / sigma^2 + 0.5 ||beta||^2 / tau^2
    Xa, ya = anp.array(X), anp.array(y)
    inv_sigma2 = 1.0 / sigma2
    inv_tau2 = 1.0 / tau2

    def E(beta):
        resid = ya - Xa @ beta
        return 0.5 * inv_sigma2 * anp.sum(resid**2) + 0.5 * inv_tau2 * anp.sum(beta**2)

    X_np, y_np = X, y

    def gradE_fn(beta):
        resid = y_np - X_np @ beta
        return -inv_sigma2 * (X_np.T @ resid) + inv_tau2 * beta

    w_vec = np.full(p, spike_weight)
    slab_scales = np.full(p, tau)
    kappa = _kappa_from_weights(w_vec, slab_scales)

    marginal_grids = {}
    for i in range(p):
        grid = np.linspace(-4 * tau, 4 * tau, 500)
        pdf = np.exp(-0.5 * grid**2 / tau2) / (tau * np.sqrt(2 * np.pi))
        marginal_grids[i] = {
            "grid": grid,
            "pdf": pdf,
            "label": f"$\\beta_{{{i+1}}}$",
            "spike_weight": float(spike_weight),
        }

    return Target(
        name=f"spike_slab_linreg_{sparsity}_n{n}_p{p}",
        task_type="validation",
        D=p,
        E=E,
        gradE=gradE_fn,
        data={"X": X, "y": y},
        true_params=beta_true,
        meta={
            "tau": tau,
            "sigma": sigma,
            "spike_weight": spike_weight,
            "spike_weights": w_vec,
            "slab_scales": slab_scales,
            "kappa": kappa,
            "inclusion_probs": inclusion_probs,
            "sparsity": sparsity,
            "marginal_grids": marginal_grids,
        },
    )


# =====================================================================
# 3. Spike-and-slab logistic regression (realistic; contraction-based)
# =====================================================================

def spike_slab_logreg(n=300, p=8, sparsity="sparse", prior=None,
                      spike_weight=0.5, seed=1):
    """
    Bayesian logistic regression with Gaussian slab + sticky spike.

    No closed-form inclusion probabilities. Validation is qualitative:
      - frozen-fraction should be high for coords where beta_true_i = 0,
      - frozen-fraction should be low for coords where beta_true_i != 0.
    For a quantitative check, rerun with increasing n and confirm
    frozen-fraction -> indicator(beta_true_i == 0).

    `prior` is the slab prior (default: gaussian, scale=1.0). Must be Gaussian
    because the kappa formula here uses the Gaussian slab density at zero.
    """
    if prior is None:
        prior = {"kind": "gaussian", "scale": 1.0}
    if prior["kind"] != "gaussian":
        raise ValueError(
            "Sticky dynamics here assume a Gaussian slab for the kappa formula. "
            "Pass a Gaussian prior spec or derive kappa yourself."
        )

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))

    if sparsity == "sparse":
        beta_true = np.zeros(p)
        beta_true[:min(3, p)] = np.array([2.0, -1.5, 0.5])[:min(3, p)]
    elif sparsity == "dense":
        beta_true = rng.normal(0, 1.0, size=p)
    else:
        raise ValueError(f"Unknown sparsity: {sparsity!r}")

    logits = X @ beta_true
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    prior_obj = build_prior(p, prior)
    E_lik, gradE_lik = _logreg_likelihood(X, y)
    E, gradE_fn = combine_likelihood_and_prior(E_lik, gradE_lik, prior_obj)

    slab_scale = prior["scale"]
    slab_scales = np.broadcast_to(np.asarray(slab_scale, dtype=float), (p,)).copy()
    w_vec = np.full(p, spike_weight)
    kappa = _kappa_from_weights(w_vec, slab_scales)

    expected_frozen = (beta_true == 0).astype(float)

    marginal_grids = {}
    for i in range(p):
        s = slab_scales[i]
        grid = np.linspace(-4 * s, 4 * s, 500)
        pdf = np.exp(-0.5 * grid**2 / s**2) / (s * np.sqrt(2 * np.pi))
        marginal_grids[i] = {
            "grid": grid,
            "pdf": pdf,
            "label": f"$\\beta_{{{i+1}}}$",
            "spike_weight": float(spike_weight),
        }

    return Target(
        name=f"spike_slab_logreg_{sparsity}_n{n}_p{p}",
        task_type="validation",
        D=p,
        E=E,
        gradE=gradE_fn,
        data={"X": X, "y": y},
        true_params=beta_true,
        meta={
            "prior": prior_obj["meta"],
            "spike_weight": spike_weight,
            "spike_weights": w_vec,
            "slab_scales": slab_scales,
            "kappa": kappa,
            "sparsity": sparsity,
            "expected_frozen": expected_frozen,
            "marginal_grids": marginal_grids,
        },
    )


# =====================================================================
# Registry
# =====================================================================

STICKY_VALIDATION_TARGETS = {
    "spike_slab_gaussian": spike_slab_gaussian,
    "spike_slab_linreg":   spike_slab_linreg,
    "spike_slab_logreg":   spike_slab_logreg,
}
