"""
BNN warmstart: train with Adam, compute block-diagonal empirical Fisher
plus prior precision, return (x_ref, Sigma_inv).

Takes a bnn_classification Target and reads X, y and architecture info
from target.data / target.meta. Does not touch target.E or target.gradE.
"""
import numpy as np


# ================================================================
# BNN forward / gradient (numpy, warmstart-local)
# ================================================================

def _forward(theta, X, shapes, slices):
    h = X
    for l in range(len(shapes) - 1):
        w_sl, b_sl = slices[l]
        W = theta[w_sl].reshape(shapes[l][0])
        b = theta[b_sl]
        h = np.tanh(h @ W + b)
    w_sl, b_sl = slices[-1]
    W = theta[w_sl].reshape(shapes[-1][0])
    b = theta[b_sl]
    out = h @ W + b
    return out.ravel() if out.shape[1] == 1 else out

def _softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def _sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _grad_nll(theta, X, y, shapes, slices):
    n_layers = len(shapes)
    activations = [X]
    pre_acts = []
    h = X
    for l in range(n_layers - 1):
        w_sl, b_sl = slices[l]
        W = theta[w_sl].reshape(shapes[l][0])
        b = theta[b_sl]
        z = h @ W + b
        h = np.tanh(z)
        pre_acts.append(z)
        activations.append(h)
    w_sl, b_sl = slices[-1]
    W = theta[w_sl].reshape(shapes[-1][0])
    b = theta[b_sl]
    logits = h @ W + b
    n_classes = logits.shape[1] if logits.ndim == 2 else 1

    if n_classes == 1:
        logits = logits.ravel()
        p = _sigmoid(logits)
        delta = (p - y).reshape(-1, 1)
    else:
        y_onehot = np.zeros_like(logits)
        y_onehot[np.arange(len(y)), y.astype(int)] = 1.0
        delta = _softmax(logits) - y_onehot
        
    g = np.zeros_like(theta)
    dW = activations[-1].T @ delta
    db = delta.sum(axis=0)
    g[slices[-1][0]] = dW.ravel()
    g[slices[-1][1]] = db.ravel()
    for l in range(n_layers - 2, -1, -1):
        W_next = theta[slices[l + 1][0]].reshape(shapes[l + 1][0])
        delta = (delta @ W_next.T) * (1.0 - np.tanh(pre_acts[l])**2)
        dW = activations[l].T @ delta
        db = delta.sum(axis=0)
        g[slices[l][0]] = dW.ravel()
        g[slices[l][1]] = db.ravel()
    return g


# ================================================================
# Adam
# ================================================================

def _train_adam(X, y, shapes, slices, D, weight_mask,
                n_epochs=3000, lr=3e-3, l2_weight=1e-3, l1_weight=0.0, l2_bias=1e-4,
                batch_size=None, verbose=True, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    if batch_size is None:
        batch_size = n

    theta = np.zeros(D)
    for l, (w_sl, b_sl) in enumerate(slices):
        fan_in, fan_out = shapes[l][0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        theta[w_sl] = rng.normal(0, std, size=w_sl.stop - w_sl.start)
        theta[b_sl] = 0.01

    m, v = np.zeros(D), np.zeros(D)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_loss, theta_best = np.inf, theta.copy()

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            bs = len(idx)
            g = _grad_nll(theta, X[idx], y[idx], shapes, slices) * (n / bs)
            g[weight_mask] += 2.0 * l2_weight * theta[weight_mask]
            g[weight_mask] += l1_weight * np.sign(theta[weight_mask])
            g[~weight_mask] += 2.0 * l2_bias * theta[~weight_mask]

            t_adam = epoch * ((n + batch_size - 1) // batch_size) + start // batch_size + 1
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**t_adam)
            v_hat = v / (1 - beta2**t_adam)
            theta -= lr * m_hat / (np.sqrt(v_hat) + eps)

        logits = _forward(theta, X, shapes, slices)
        if logits.ndim == 1:
            loss = np.sum(np.logaddexp(0.0, logits) - y * logits)
        else:
            log_norm = np.log(np.sum(np.exp(logits - logits.max(axis=1, keepdims=True)), axis=1)) + logits.max(axis=1)
            loss = -np.sum(logits[np.arange(len(y)), y.astype(int)]) + np.sum(log_norm)

        if loss < best_loss:
            best_loss = loss
            theta_best = theta.copy()

        if verbose and (epoch % 500 == 0 or epoch == n_epochs - 1):
            if logits.ndim == 1:
                acc = np.mean((_sigmoid(logits) > 0.5).astype(float) == y)
            else:
                acc = np.mean(logits.argmax(axis=1) == y.astype(int))
            print(f"    epoch {epoch:5d}/{n_epochs}  nll={loss:.1f}  acc={acc:.3f}")

    return theta_best


# ================================================================
# Empirical Fisher
# ================================================================

def _empirical_fisher(theta, X, y, shapes, slices, n_fisher=500, seed=None):
    n = len(y)
    D = len(theta)
    n_fisher = min(n, n_fisher)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_fisher, replace=False)
    F = np.zeros((D, D))
    for i in idx:
        gi = _grad_nll(theta, X[i:i+1], y[i:i+1], shapes, slices)
        F += np.outer(gi, gi)
    F /= n_fisher
    return F

def _empirical_fisher_diagonal(theta, X, y, shapes, slices, n_fisher=500, seed=None):
    n = len(y)
    D = len(theta)
    n_fisher = min(n, n_fisher)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_fisher, replace=False)
    F_diag = np.zeros(D)
    for i in idx:
        gi = _grad_nll(theta, X[i:i+1], y[i:i+1], shapes, slices)
        F_diag += gi**2
    F_diag /= n_fisher
    return F_diag


# ================================================================
# Warmstart entry point
# ================================================================

def adam_fisher(target, n_epochs=3000, lr=3e-3, l2_weight=1e-3, l1_weight=0.0,
                l2_bias=1e-4, n_fisher=500, fisher_reg=1e-3,
                update_sigma_w=False, seed=42, verbose=True):
    """
    Train BNN target with Adam and compute block-diagonal empirical Fisher
    plus prior precision.

    Parameters
    ----------
    target : Target
        Must be a bnn_classification target (has X, y in data and
        shapes/slices/weight_mask/sigma_w_layers/sigma_b_layers in meta).
    update_sigma_w : bool
        If True, overwrite target.meta['sigma_w_layers'] with per-layer std
        of the trained weights. This matches the behavior of the monolithic
        warmstart and is what downstream consumers like kappa='masked' expect.
        NOTE: target.E and target.gradE are NOT rebuilt — they keep whatever
        sigma_w was set at target construction. Set False if you want the
        sigma_w in meta to remain as declared.

    Returns
    -------
    {"x_ref": ndarray, "Sigma_inv": ndarray}
    """
    if target.task_type != "bnn_classification":
        raise ValueError(
            f"bnn.adam_fisher warmstart only supports BNN targets; got "
            f"task_type={target.task_type!r}"
        )

    X = target.data["X"]
    y = target.data["y"]
    shapes = target.meta["shapes"]
    slices = target.meta["slices"]
    weight_mask = target.meta["weight_mask"]
    sigma_w_layers = list(target.meta["sigma_w_layers"])
    sigma_b_layers = list(target.meta["sigma_b_layers"])
    D = target.D

    if verbose:
        print(f"Warmstart (bnn.adam_fisher): {target.meta['layer_sizes']}  (D={D})")
        print("  Training with Adam...")

    theta = _train_adam(
        X, y, shapes, slices, D, weight_mask,
        n_epochs=n_epochs, lr=lr, l2_weight=l2_weight, l1_weight=l1_weight, l2_bias=l2_bias,
        seed=seed, verbose=verbose,
    )

    if verbose:
        logits = _forward(theta, X, shapes, slices)
        if logits.ndim == 1:
            acc = np.mean((_sigmoid(logits) > 0.5).astype(float) == y)
        else:
            acc = np.mean(logits.argmax(axis=1) == y.astype(int))
        print(f"  Train accuracy: {acc:.3f}")

    # NOTE: Intentionally disabled. Updating sigma_w from trained weights
    # conflates prior and likelihood and makes runs non-comparable across
    # sampler variants. Set sigma_w explicitly in the target config instead.
    # if update_sigma_w:
    #     for l, (w_sl, _) in enumerate(slices):
    #         sigma_w_layers[l] = max(float(np.std(theta[w_sl])), 0.1)
    #     target.meta["sigma_w_layers"] = sigma_w_layers
    #     if verbose:
    #         print(f"  σ_w per layer: {[f'{s:.3f}' for s in sigma_w_layers]}")

    if verbose:
        print("  Computing empirical Fisher (diagonal)...")

    if D > 2000:
        # Diagonal Fisher for large networks
        F_diag = _empirical_fisher_diagonal(theta, X, y, shapes, slices,
                                            n_fisher=n_fisher, seed=seed)
        prior_precision = np.zeros(D)
        for l, (w_sl, b_sl) in enumerate(slices):
            prior_precision[w_sl] = 1.0 / sigma_w_layers[l]**2
            prior_precision[b_sl] = 1.0 / sigma_b_layers[l]**2

        diag_vals = np.clip(F_diag + prior_precision, fisher_reg, None)
        Sigma_inv = np.diag(diag_vals)

        if verbose:
            print(f"  Σ⁻¹ diagonal range: [{diag_vals.min():.4f}, {diag_vals.max():.4f}]")
    else:
        # Block-diagonal Fisher for small networks
        F = _empirical_fisher(theta, X, y, shapes, slices,
                              n_fisher=n_fisher, seed=seed)
        Sigma_inv = np.zeros((D, D))
        for l, (w_sl, b_sl) in enumerate(slices):
            idx = np.concatenate([
                np.arange(w_sl.start, w_sl.stop),
                np.arange(b_sl.start, b_sl.stop),
            ])
            block = 0.5 * (F[np.ix_(idx, idx)] + F[np.ix_(idx, idx)].T)
            for k, i in enumerate(idx):
                if weight_mask[i]:
                    block[k, k] += 1.0 / sigma_w_layers[l]**2
                else:
                    block[k, k] += 1.0 / sigma_b_layers[l]**2
            eig_min = np.linalg.eigvalsh(block).min()
            if eig_min < fisher_reg:
                block += (fisher_reg - eig_min + 1e-6) * np.eye(len(idx))
            Sigma_inv[np.ix_(idx, idx)] = block

        if verbose:
            eigvals = np.linalg.eigvalsh(Sigma_inv)
            print(f"  Σ⁻¹ eigenvalues: [{eigvals.min():.4f}, {eigvals.max():.4f}]")

    if verbose:
        print("  Done.")

    return {"x_ref": theta, "Sigma_inv": Sigma_inv}
