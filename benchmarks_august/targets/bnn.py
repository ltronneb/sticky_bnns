"""
BNN classification target. Pure target definition: architecture, forward,
gradient, likelihood, prior injection. No training — warmstart lives in
samplers/warmstart/.
"""
import numpy as np
import autograd.numpy as anp
from .base import Target
from .priors import build_prior, combine_likelihood_and_prior


# ================================================================
# Architecture
# ================================================================

def _make_architecture(layer_sizes):
    shapes, slices = [], []
    weight_mask_parts = []
    offset = 0
    for i in range(len(layer_sizes) - 1):
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        w_size = fan_in * fan_out
        b_size = fan_out
        w_sl = slice(offset, offset + w_size)
        offset += w_size
        b_sl = slice(offset, offset + b_size)
        offset += b_size
        shapes.append(((fan_in, fan_out), (fan_out,)))
        slices.append((w_sl, b_sl))
        weight_mask_parts.append(np.ones(w_size, dtype=bool))
        weight_mask_parts.append(np.zeros(b_size, dtype=bool))
    D = offset
    weight_mask = np.concatenate(weight_mask_parts)
    return shapes, slices, D, weight_mask


def _sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# ================================================================
# Likelihood only (no prior baked in)
# ================================================================

def _bnn_likelihood(X, y, layer_sizes, shapes, slices, D):
    X_a, y_a = anp.array(X), anp.array(y)
    X_np, y_np = np.asarray(X), np.asarray(y)
    n_classes = layer_sizes[-1]

    def _fwd(theta, X_in, sh, sl):
        params = []
        for (w_shape, _), (w_s, b_s) in zip(sh, sl):
            params.append((anp.reshape(theta[w_s], w_shape), theta[b_s]))
        h = X_in
        for W, b in params[:-1]:
            h = anp.tanh(h @ W + b)
        Wl, bl = params[-1]
        return h @ Wl + bl  # (n, n_classes) for multiclass, (n, 1) for binary

    if n_classes == 1:
        # Binary: existing code
        def E_lik(theta):
            logits = _fwd(theta, X_a, shapes, slices).ravel()
            return anp.sum(anp.logaddexp(0.0, logits) - y_a * logits)
    else:
        # Multiclass: categorical cross-entropy
        y_int = np.asarray(y, dtype=int)
        y_int_a = anp.array(y_int)

        def E_lik(theta):
            logits = _fwd(theta, X_a, shapes, slices)  # (n, C)
            log_normalizer = anp.log(anp.sum(anp.exp(logits - anp.max(logits, axis=1, keepdims=True)), axis=1)) + anp.max(logits, axis=1)
            return -anp.sum(logits[anp.arange(len(y_int_a)), y_int_a]) + anp.sum(log_normalizer)

    # Hand-coded gradient for single-hidden-layer
    if len(layer_sizes) == 3:
        d_in = X_np.shape[1]
        H = layer_sizes[1]

        if n_classes == 1:
            def gradE_lik(theta):
                W1 = theta[slices[0][0]].reshape(d_in, H)
                b1 = theta[slices[0][1]]
                W2 = theta[slices[1][0]].reshape(H, 1)
                b2 = theta[slices[1][1]]
                z1 = X_np @ W1 + b1
                a1 = np.tanh(z1)
                logits = (a1 @ W2 + b2).ravel()
                p = _sigmoid(logits)
                delta_out = (p - y_np).reshape(-1, 1)
                dW2 = a1.T @ delta_out
                db2 = delta_out.sum(axis=0)
                delta_h = (delta_out @ W2.T) * (1.0 - a1**2)
                dW1 = X_np.T @ delta_h
                db1 = delta_h.sum(axis=0)
                g = np.zeros(D)
                g[slices[0][0]] = dW1.ravel()
                g[slices[0][1]] = db1
                g[slices[1][0]] = dW2.ravel()
                g[slices[1][1]] = db2
                return g
        else:
            C = n_classes
            y_onehot = np.zeros((len(y_np), C))
            y_onehot[np.arange(len(y_np)), y_np.astype(int)] = 1.0

            def _softmax(logits):
                e = np.exp(logits - logits.max(axis=1, keepdims=True))
                return e / e.sum(axis=1, keepdims=True)

            def gradE_lik(theta):
                W1 = theta[slices[0][0]].reshape(d_in, H)
                b1 = theta[slices[0][1]]
                W2 = theta[slices[1][0]].reshape(H, C)
                b2 = theta[slices[1][1]]
                z1 = X_np @ W1 + b1
                a1 = np.tanh(z1)
                logits = a1 @ W2 + b2
                delta_out = _softmax(logits) - y_onehot  # (n, C)
                dW2 = a1.T @ delta_out
                db2 = delta_out.sum(axis=0)
                delta_h = (delta_out @ W2.T) * (1.0 - a1**2)
                dW1 = X_np.T @ delta_h
                db1 = delta_h.sum(axis=0)
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
# Per-layer scale expansion
# ================================================================

def _expand_layered_scale(layer_sizes, slices, D, weight_mask,
                          sigma_w_layers, sigma_b_layers):
    """
    Expand per-layer (sigma_w, sigma_b) into a length-D scale vector,
    suitable for priors.gaussian(D, scale=<vector>).
    """
    scale = np.zeros(D)
    for l, (w_sl, b_sl) in enumerate(slices):
        scale[w_sl] = sigma_w_layers[l]
        scale[b_sl] = sigma_b_layers[l]
    return scale


# ================================================================
# Target factory
# ================================================================

def bnn_classification(X, y, layer_sizes, prior, name=None):
    """
    Build a BNN classification target.

    Parameters
    ----------
    X, y : data
    layer_sizes : list[int], e.g. [d_in, H, 1]
    prior : dict
        Either a standard prior spec, e.g.
            {"kind": "gaussian", "scale": 1.0}
        or a BNN-specific "layered_gaussian" spec with per-layer scales:
            {"kind": "layered_gaussian",
             "sigma_w_layers": [1.0, 1.0],
             "sigma_b_layers": [1.0, 1.0]}
        The latter is expanded into a length-D gaussian prior.

    Does NOT run any training. For warm-started runs, declare a `warmstart`
    block in the config; the runner will apply it before sampling.
    """
    shapes, slices, D, weight_mask = _make_architecture(layer_sizes)

    # Resolve the prior spec. Layered Gaussian is sugar for a per-coordinate
    # Gaussian with scales expanded from per-layer values.
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
    E_lik, gradE_lik = _bnn_likelihood(
        X, y, layer_sizes, shapes, slices, D,
    )
    E, gradE = combine_likelihood_and_prior(E_lik, gradE_lik, prior_obj)

    meta = {
        "layer_sizes": layer_sizes,
        "weight_mask": weight_mask,
        "shapes": shapes,
        "slices": slices,
        "prior": {**prior_obj["meta"], **prior_meta_extra},
    }
    if sigma_w_layers is not None:
        meta["sigma_w_layers"] = sigma_w_layers
        meta["sigma_b_layers"] = sigma_b_layers

    return Target(
        name=name or f"bnn_{'_'.join(map(str, layer_sizes))}",
        task_type="bnn_classification",
        D=D,
        E=E,
        gradE=gradE,
        data={"X": np.asarray(X), "y": np.asarray(y)},
        meta=meta,
    )
