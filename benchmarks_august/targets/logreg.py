"""
Logistic regression target.

Two factories:
  - logreg(X, y, prior=...)                : build from given data
  - logreg_synthetic(n, p, sparsity, ...)  : generate synthetic data with
                                             known beta_true, then build target

Both take a `prior` spec dict and compose likelihood + prior into E/gradE.
"""
import numpy as np
import autograd.numpy as anp
from .base import Target
from .priors import build_prior, combine_likelihood_and_prior


def _logreg_likelihood(X, y):
    """Return (E_lik, gradE_lik) for Bernoulli-logit likelihood."""
    Xa, ya = anp.array(X), anp.array(y)
    X_np, y_np = np.asarray(X), np.asarray(y)

    def E_lik(beta):
        eta = Xa @ beta
        return anp.sum(anp.logaddexp(0.0, eta) - ya * eta)

    def gradE_lik(beta):
        eta = X_np @ beta
        probs = 1.0 / (1.0 + np.exp(-eta))
        return X_np.T @ (probs - y_np)

    return E_lik, gradE_lik


def logreg(X, y, prior, name=None):
    """
    Build a Bayesian logistic regression target from given data.

    Parameters
    ----------
    X : (n, p) array of features
    y : (n,)   array of 0/1 labels
    prior : dict
        e.g. {"kind": "gaussian", "scale": 1.0}
    name : optional human-readable name
    """
    X = np.asarray(X)
    y = np.asarray(y)
    p = X.shape[1]

    prior_obj = build_prior(p, prior)
    E_lik, gradE_lik = _logreg_likelihood(X, y)
    E, gradE = combine_likelihood_and_prior(E_lik, gradE_lik, prior_obj)

    return Target(
        name=name or f"logreg_p{p}",
        task_type="classification",
        D=p,
        E=E,
        gradE=gradE,
        data={"X": X, "y": y},
        meta={"prior": prior_obj["meta"]},
    )


def logreg_synthetic(n=300, p=5, sparsity="sparse", prior=None, seed=1, name=None):
    """
    Generate synthetic logistic regression data with known beta_true, then
    build the target around it.

    `sparsity`: 'sparse' | 'dense'
    `prior`: required; e.g. {"kind": "gaussian", "scale": 1.0}
    """
    if prior is None:
        raise ValueError("logreg_synthetic requires a `prior` spec, e.g. "
                         "{'kind': 'gaussian', 'scale': 1.0}")

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

    target = logreg(
        X, y, prior=prior,
        name=name or f"logreg_{sparsity}_n{n}_p{p}",
    )
    target.true_params = beta_true
    target.meta.update({"sparsity": sparsity, "data_seed": seed})
    return target
