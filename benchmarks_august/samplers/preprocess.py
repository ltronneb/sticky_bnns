"""
Preprocessing dispatch. Config says which mode; we call sampler.preprocess accordingly.

Modes:
  - diagonal : sampler.preprocess(method="diagonal")
  - identity : sampler.preprocess(method="identity")
  - manual   : sampler.preprocess(method="manual", x_ref=..., Sigma_inv=...)
               x_ref / Sigma_inv come from target.x_ref / target.Sigma_inv
               (set by the target factory or a warmstart step)
"""


def apply_preprocess(sampler, target, cfg):
    """
    cfg example:
        {"method": "diagonal"}
        {"method": "manual"}          # pulls from target.x_ref / target.Sigma_inv
        {"method": "manual", "source": "target"}   # same
    """
    method = cfg["method"]

    if method in ("diagonal", "identity"):
        sampler.preprocess(method=method)
        return

    if method == "manual":
        if target.x_ref is None or target.Sigma_inv is None:
            raise ValueError(
                f"preprocess method='manual' requires target.x_ref and target.Sigma_inv. "
                f"Target '{target.name}' has x_ref={target.x_ref is not None}, "
                f"Sigma_inv={target.Sigma_inv is not None}."
            )
        sampler.preprocess(method="manual", x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
        return

    raise ValueError(f"Unknown preprocess method: {method}")
