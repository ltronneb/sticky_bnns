"""
Continuous priors for Bayesian targets.

A prior exposes a negative-log-density `E_prior` and its gradient `gradE_prior`,
both taking a parameter vector of length D. For sticky PDMP samplers the
continuous prior here plays the role of the SLAB component — the spike at zero
is supplied implicitly by the sticky dynamics (controlled by kappa).

Registering a new prior:
  1. Write a function `my_prior(D, **kwargs) -> dict` returning
       {"E": callable, "gradE": callable, "meta": {...}}
     where meta is JSON-serializable and always contains a "kind" field.
  2. Register it in PRIORS below.
"""
import numpy as np
import autograd.numpy as anp


def gaussian(D, scale):
    """
    Isotropic zero-mean Gaussian prior with standard deviation `scale`.

    Supports either a scalar scale (same for all coordinates) or a length-D
    array (per-coordinate, e.g. for layered BNN priors).
    """
    scale_arr = np.broadcast_to(np.asarray(scale, dtype=float), (D,)).copy()
    precision = 1.0 / scale_arr**2
    precision_anp = anp.array(precision)

    def E(beta):
        return 0.5 * anp.sum(precision_anp * beta**2)

    def gradE(beta):
        return precision * beta

    return {
        "E": E,
        "gradE": gradE,
        "meta": {
            "kind": "gaussian",
            "scale": scale_arr.tolist() if scale_arr.size > 1 or not np.isscalar(scale) else float(scale),
        },
    }


def laplace(D, scale):
    """
    Zero-mean Laplace (double-exponential) prior with scale `b` = `scale`.
        p(beta) ∝ exp(-|beta|/b)
    Not differentiable at zero; gradE returns sign(beta)/b, with 0 at exactly 0.
    """
    scale_arr = np.broadcast_to(np.asarray(scale, dtype=float), (D,)).copy()
    inv_b = 1.0 / scale_arr
    inv_b_anp = anp.array(inv_b)

    def E(beta):
        return anp.sum(inv_b_anp * anp.abs(beta))

    def gradE(beta):
        return inv_b * np.sign(beta)

    return {
        "E": E,
        "gradE": gradE,
        "meta": {
            "kind": "laplace",
            "scale": scale_arr.tolist() if scale_arr.size > 1 or not np.isscalar(scale) else float(scale),
        },
    }


PRIORS = {
    "gaussian": gaussian,
    "laplace": laplace,
}


def build_prior(D, spec):
    """
    spec example:
        {"kind": "gaussian", "scale": 1.0}
        {"kind": "laplace",  "scale": 0.5}
    """
    kind = spec["kind"]
    if kind not in PRIORS:
        raise ValueError(f"Unknown prior kind: {kind!r}. Known: {list(PRIORS)}")
    kwargs = {k: v for k, v in spec.items() if k != "kind"}
    return PRIORS[kind](D, **kwargs)


def combine_likelihood_and_prior(E_lik, gradE_lik, prior):
    """
    Eagerly compose likelihood and prior into a single E / gradE pair.
    Called once at target construction; no per-sample lambda overhead.
    """
    E_pr = prior["E"]
    gradE_pr = prior["gradE"]

    def E(beta):
        return E_lik(beta) + E_pr(beta)

    def gradE(beta):
        return gradE_lik(beta) + gradE_pr(beta)

    return E, gradE
