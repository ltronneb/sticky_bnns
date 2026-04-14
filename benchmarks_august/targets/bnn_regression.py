import numpy as np
import autograd.numpy as anp
from .base import Target
from .priors import build_prior, combine_likelihood_and_prior
from .bnn import _make_architecture, _expand_layered_scale


# ================================================================
# Regression likelihood (Gaussian, fixed tau)
# ================================================================

def _bnn_regression_likelihood(X, y, tau, layer_sizes, shapes, slices, D):
    """
    Gaussian likelihood:  E_lik = (tau/2) * sum_i (f(x_i; theta) - y_i)^2

    tau is the observation noise precision (1/sigma^2), treated as fixed.
    """
    X_a, y_a = anp.array(X), anp.array(y)
    X_np, y_np = np.asarray(X), np.asarray(y)
    n = X_np.shape[0]

    assert layer_sizes[-1] == 1, (
        f"Regression BNN must have 1 output node, got {layer_sizes[-1]}"
    )

    # ── autograd forward (for E_lik) ──────────────────────────────
    def _fwd_ag(theta, X_in):
        h = X_in
        for (w_shape, _), (w_s, b_s) in zip(shapes, slices):
            W = anp.reshape(theta[w_s], w_shape)
            b = theta[b_s]
            h = h @ W + b
            if w_s != slices[-1][0]:          # hidden layers: tanh
                h = anp.tanh(h)
        return h.ravel()                      # (n,)

    def E_lik(theta):
        residuals = _fwd_ag(theta, X_a) - y_a
        return 0.5 * tau * anp.sum(residuals ** 2)

    # ── hand-coded gradient for single-hidden-layer ───────────────
    if len(layer_sizes) == 3:
        d_in = X_np.shape[1]
        H = layer_sizes[1]

        def gradE_lik(theta):
            W1 = theta[slices[0][0]].reshape(d_in, H)
            b1 = theta[slices[0][1]]
            W2 = theta[slices[1][0]].reshape(H, 1)
            b2 = theta[slices[1][1]]

            # forward
            z1 = X_np @ W1 + b1
            a1 = np.tanh(z1)
            pred = (a1 @ W2 + b2).ravel()

            # backward
            delta_out = tau * (pred - y_np).reshape(-1, 1)   # (n, 1)
            dW2 = a1.T @ delta_out                           # (H, 1)
            db2 = delta_out.sum(axis=0)                      # (1,)
            delta_h = (delta_out @ W2.T) * (1.0 - a1 ** 2)  # (n, H)
            dW1 = X_np.T @ delta_h                           # (d_in, H)
            db1 = delta_h.sum(axis=0)                        # (H,)

            g = np.zeros(D)
            g[slices[0][0]] = dW1.ravel()
            g[slices[0][1]] = db1
            g[slices[1][0]] = dW2.ravel()
            g[slices[1][1]] = db2
            return g
    else:
        from autograd import grad as agrad
        gradE_lik = agrad(E_lik)

    return E_lik, gradE_lik


# ================================================================
# Target factory
# ================================================================

def bnn_regression(X, y, layer_sizes, prior, tau=1.0, name=None):
    """
    Build a BNN regression target with Gaussian likelihood.

    Parameters
    ----------
    X : (n, d_in) array of features (should be standardized)
    y : (n,) array of targets (should be standardized)
    layer_sizes : list[int], e.g. [d_in, 50, 1]
    prior : dict
        Standard prior spec or "layered_gaussian" (same as classification).
    tau : float
        Observation noise precision (1 / sigma_noise^2). Fixed, not sampled.
        Default 1.0 is natural when y is standardized.
    name : optional human-readable name
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if layer_sizes[0] != X.shape[1]:
        raise ValueError(
            f"layer_sizes[0]={layer_sizes[0]} != X.shape[1]={X.shape[1]}"
        )
    if layer_sizes[-1] != 1:
        raise ValueError(
            f"Regression requires layer_sizes[-1]==1, got {layer_sizes[-1]}"
        )

    shapes, slices, D, weight_mask = _make_architecture(layer_sizes)

    # ── resolve prior (identical logic to classification) ─────────
    if prior["kind"] == "layered_gaussian":
        sigma_w_layers = list(prior["sigma_w_layers"])
        sigma_b_layers = list(prior["sigma_b_layers"])
        if len(sigma_w_layers) != len(slices) or len(sigma_b_layers) != len(slices):
            raise ValueError(
                f"layered_gaussian prior needs one entry per layer "
                f"(got {len(sigma_w_layers)} weight scales, "
                f"{len(sigma_b_layers)} bias scales, expected {len(slices)} each)"
            )
        scale = _expand_layered_scale(
            layer_sizes, slices, D, weight_mask, sigma_w_layers, sigma_b_layers,
        )
        resolved_prior_spec = {"kind": "gaussian", "scale": scale}
        prior_meta_extra = {
            "layered": True,
            "sigma_w_layers": sigma_w_layers,
            "sigma_b_layers": sigma_b_layers,
        }
    else:
        resolved_prior_spec = prior
        sigma_w_layers = None
        sigma_b_layers = None
        prior_meta_extra = {}

    prior_obj = build_prior(D, resolved_prior_spec)
    E_lik, gradE_lik = _bnn_regression_likelihood(
        X, y, tau, layer_sizes, shapes, slices, D,
    )
    E, gradE = combine_likelihood_and_prior(E_lik, gradE_lik, prior_obj)

    meta = {
        "layer_sizes": layer_sizes,
        "weight_mask": weight_mask,
        "shapes": shapes,
        "slices": slices,
        "tau": tau,
        "prior": {**prior_obj["meta"], **prior_meta_extra},
    }
    if sigma_w_layers is not None:
        meta["sigma_w_layers"] = sigma_w_layers
        meta["sigma_b_layers"] = sigma_b_layers

    return Target(
        name=name or f"bnn_reg_{'_'.join(map(str, layer_sizes))}",
        task_type="bnn_regression",
        D=D,
        E=E,
        gradE=gradE,
        data={"X": X, "y": y, "tau": tau},
        meta=meta,
    )