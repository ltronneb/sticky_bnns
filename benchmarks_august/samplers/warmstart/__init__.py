"""
Warmstart strategies. Each strategy takes a Target and returns
{"x_ref": ndarray, "Sigma_inv": ndarray}. The dispatcher attaches
the result to the Target so preprocess(method='manual') can consume it.

Add a new strategy:
  1. Write a module in samplers/warmstart/ (e.g. logreg.py) with a function
     `some_name(target, **kwargs) -> {'x_ref', 'Sigma_inv'}`
  2. Register it in WARMSTARTS below under a unique key.
"""
from . import bnn

WARMSTARTS = {
    "bnn_adam_fisher": bnn.adam_fisher,
    # "logreg_laplace": logreg.laplace,
    # "linreg_mle":     linreg.mle,
}


def apply_warmstart(spec, target):
    """
    spec example:
        {"kind": "bnn_adam_fisher", "kwargs": {"n_epochs": 3000, "lr": 0.003}}

    Calls the named strategy and attaches x_ref + Sigma_inv to the target.
    Returns the raw result dict as well, in case the caller wants it.
    """
    kind = spec["kind"]
    if kind not in WARMSTARTS:
        raise ValueError(
            f"Unknown warmstart kind: {kind!r}. Known: {list(WARMSTARTS)}"
        )
    result = WARMSTARTS[kind](target, **spec.get("kwargs", {}))
    target.x_ref = result["x_ref"]
    target.Sigma_inv = result["Sigma_inv"]
    return result
