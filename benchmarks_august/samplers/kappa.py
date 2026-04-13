"""
Helpers for constructing kappa arrays for Sticky samplers.

kappa is the sticky rate. For a Gaussian slab with scale sigma and
slab-mixing weight (1 - gamma), the sticky formula is:

    kappa = (gamma / (1 - gamma)) / (sigma * sqrt(2 * pi))

The slab is the continuous prior stored in target.meta["prior"], so the
scale is read from there — no need to re-specify it in the kappa block.
"""
import numpy as np


def _slab_scale_from_target(target):
    """Pull the continuous slab scale out of target.meta['prior']."""
    prior_meta = target.meta.get("prior")
    if prior_meta is None:
        raise ValueError(
            "kappa builder expected a 'prior' entry in target.meta, but found none. "
            "Ensure the target factory was called with a `prior` spec."
        )
    if prior_meta["kind"] not in ("gaussian",):
        raise ValueError(
            f"kappa formulas currently assume a Gaussian slab; got "
            f"prior kind={prior_meta['kind']!r}"
        )
    scale = prior_meta["scale"]
    return np.broadcast_to(np.asarray(scale, dtype=float), (target.D,)).copy()


def _kappa_from_gamma_and_scale(gamma, scale):
    """scale may be scalar or length-D array; returns length-D kappa."""
    scale = np.asarray(scale, dtype=float)
    return (gamma / (1 - gamma)) / (scale * np.sqrt(2 * np.pi))


def build_kappa(spec, target):
    """
    spec examples:
        {"kind": "uniform", "gamma_prior": 0.6}
            → uses slab scale from target.meta['prior']

        {"kind": "masked",  "gamma_prior": 0.2, "kappa_unmasked": 1e6}
            → sticky on target.meta['weight_mask'] only, biases get kappa_unmasked;
              slab scale still read from target.meta['prior']

        {"kind": "scalar",  "value": 0.5}
            → plain scalar kappa, broadcast to D (bypasses all prior logic)

        {"kind": "array",   "from": "target_meta"}
            → read a pre-built kappa array directly from target.meta['kappa']
    """
    kind = spec["kind"]

    if kind == "scalar":
        return np.full(target.D, float(spec["value"]))

    if kind == "array":
        return np.asarray(target.meta["kappa"])

    if kind == "uniform":
        scale = _slab_scale_from_target(target)
        return _kappa_from_gamma_and_scale(spec["gamma_prior"], scale)

    if kind == "masked":
        mask = target.meta.get("weight_mask")
        if mask is None:
            raise ValueError("kappa kind='masked' needs target.meta['weight_mask']")
        scale = _slab_scale_from_target(target)
        k = _kappa_from_gamma_and_scale(spec["gamma_prior"], scale)
        k_unmasked = spec.get("kappa_unmasked", 1e6)
        k = np.where(mask, k, k_unmasked)
        return k

    raise ValueError(f"Unknown kappa kind: {kind!r}")
