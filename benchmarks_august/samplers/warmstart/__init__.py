from . import bnn_classification_warmstart, bnn_regression_warmstart
from .warm_reference import warmup_reference

WARMSTARTS = {
    "bnn_adam_fisher": bnn_classification_warmstart.adam_fisher,
    "bnn_regression_adam_fisher": bnn_regression_warmstart.adam_fisher_regression,
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
