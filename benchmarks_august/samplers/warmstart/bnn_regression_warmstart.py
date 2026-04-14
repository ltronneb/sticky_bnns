"""
BNN warmstart for regression: train with Adam (MSE loss), compute
block-diagonal empirical Fisher plus prior precision, return
(x_ref, Sigma_inv).

Mirrors bnn_adam_fisher for classification — the only differences are
the loss (Gaussian NLL / MSE) and the gradient (linear output layer).
"""
import numpy as np


# ================================================================
# BNN forward / gradient for regression (numpy, warmstart-local)
# ================================================================

def _forward(theta, X, shapes, slices):
    """Forward pass: hidden layers use tanh, output layer is linear."""
    h = X
    for l in range(len(shapes) - 1):
        w_sl, b_sl = slices[l]
        W = theta[w_sl].reshape(shapes[l][0])
        b = theta[b_sl]
        h = np.tanh(h @ W + b)
    w_sl, b_sl = slices[-1]
    W = theta[w_sl].reshape(shapes[-1][0])
    b = theta[b_sl]
    return (h @ W + b).ravel()  # (n,) — single output


def _grad_mse(theta, X, y, tau, shapes, slices):
    """
    Gradient of the Gaussian NLL:  E = (tau/2) * sum (f(x_i) - y_i)^2
    with respect to theta.  Backprop through [input -> tanh -> ... -> linear].
    """
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

    # Output layer (linear)
    w_sl, b_sl = slices[-1]
    W = theta[w_sl].reshape(shapes[-1][0])
    b = theta[b_sl]
    pred = (h @ W + b).ravel()  # (n,)

    # Output delta: d/d(logits) of (tau/2)*||pred - y||^2
    delta = tau * (pred - y).reshape(-1, 1)  # (n, 1)

    g = np.zeros_like(theta)

    # Last layer grads
    dW = activations[-1].T @ delta
    db = delta.sum(axis=0)
    g[slices[-1][0]] = dW.ravel()
    g[slices[-1][1]] = db.ravel()

    # Backprop through hidden layers
    for l in range(n_layers - 2, -1, -1):
        W_next = theta[slices[l + 1][0]].reshape(shapes[l + 1][0])
        delta = (delta @ W_next.T) * (1.0 - np.tanh(pre_acts[l]) ** 2)
        dW = activations[l].T @ delta
        db = delta.sum(axis=0)
        g[slices[l][0]] = dW.ravel()
        g[slices[l][1]] = db.ravel()

    return g


# ================================================================
# Adam (MSE loss)
# ================================================================

def _train_adam(X, y, tau, shapes, slices, D, weight_mask,
                n_epochs=3000, lr=3e-3, l2_weight=1e-3, l1_weight=0.0,
                l2_bias=1e-4, batch_size=None, verbose=True, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    if batch_size is None:
        batch_size = n

    # Xavier init
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

            # Gradient of Gaussian NLL, scaled to full dataset
            g = _grad_mse(theta, X[idx], y[idx], tau, shapes, slices) * (n / bs)

            # Regularization
            g[weight_mask] += 2.0 * l2_weight * theta[weight_mask]
            g[weight_mask] += l1_weight * np.sign(theta[weight_mask])
            g[~weight_mask] += 2.0 * l2_bias * theta[~weight_mask]

            t_adam = epoch * ((n + batch_size - 1) // batch_size) + start // batch_size + 1
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t_adam)
            v_hat = v / (1 - beta2 ** t_adam)
            theta -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Epoch loss: (tau/2) * sum (pred - y)^2
        pred = _forward(theta, X, shapes, slices)
        residuals = pred - y
        loss = 0.5 * tau * np.sum(residuals ** 2)
        mse = np.mean(residuals ** 2)

        if loss < best_loss:
            best_loss = loss
            theta_best = theta.copy()

        if verbose and (epoch % 500 == 0 or epoch == n_epochs - 1):
            rmse = np.sqrt(mse)
            print(f"    epoch {epoch:5d}/{n_epochs}  "
                  f"nll={loss:.1f}  mse={mse:.4f}  rmse={rmse:.4f}")

    return theta_best


# ================================================================
# Empirical Fisher (same logic as classification)
# ================================================================

def _empirical_fisher(theta, X, y, tau, shapes, slices, n_fisher=500, seed=None):
    n = len(y)
    D = len(theta)
    n_fisher = min(n, n_fisher)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_fisher, replace=False)
    F = np.zeros((D, D))
    for i in idx:
        gi = _grad_mse(theta, X[i:i + 1], y[i:i + 1], tau, shapes, slices)
        F += np.outer(gi, gi)
    F /= n_fisher
    return F


def _empirical_fisher_diagonal(theta, X, y, tau, shapes, slices,
                                n_fisher=500, seed=None):
    n = len(y)
    D = len(theta)
    n_fisher = min(n, n_fisher)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_fisher, replace=False)
    F_diag = np.zeros(D)
    for i in idx:
        gi = _grad_mse(theta, X[i:i + 1], y[i:i + 1], tau, shapes, slices)
        F_diag += gi ** 2
    F_diag /= n_fisher
    return F_diag


# ================================================================
# Warmstart entry point
# ================================================================

def adam_fisher_regression(target, n_epochs=3000, lr=3e-3, l2_weight=1e-3,
                           l1_weight=0.0, l2_bias=1e-4, n_fisher=500,
                           fisher_reg=1e-3, seed=42, verbose=True):
    """
    Train BNN regression target with Adam and compute block-diagonal
    empirical Fisher plus prior precision.

    Parameters
    ----------
    target : Target
        Must be a bnn_regression target (has X, y, tau in data and
        shapes/slices/weight_mask in meta).

    Returns
    -------
    {"x_ref": ndarray, "Sigma_inv": ndarray}
    """
    if target.task_type != "bnn_regression":
        raise ValueError(
            f"bnn_regression.adam_fisher_regression warmstart only supports "
            f"bnn_regression targets; got task_type={target.task_type!r}"
        )

    X = target.data["X"]
    y = target.data["y"]
    tau = target.data["tau"]
    shapes = target.meta["shapes"]
    slices = target.meta["slices"]
    weight_mask = target.meta["weight_mask"]
    D = target.D

    # Prior scales: either from layered_gaussian or scalar
    prior_meta = target.meta["prior"]
    if target.meta.get("sigma_w_layers") is not None:
        sigma_w_layers = list(target.meta["sigma_w_layers"])
        sigma_b_layers = list(target.meta["sigma_b_layers"])
    else:
        # Scalar prior: same scale everywhere
        scale = prior_meta.get("scale", 1.0)
        if isinstance(scale, np.ndarray):
            # Per-coordinate scale — extract per-layer from first weight in each layer
            sigma_w_layers = [float(scale[sl[0].start]) for sl in slices]
            sigma_b_layers = [float(scale[sl[1].start]) for sl in slices]
        else:
            sigma_w_layers = [float(scale)] * len(slices)
            sigma_b_layers = [float(scale)] * len(slices)

    if verbose:
        print(f"Warmstart (bnn_regression.adam_fisher): "
              f"{target.meta['layer_sizes']}  (D={D}, tau={tau})")
        print("  Training with Adam...")

    theta = _train_adam(
        X, y, tau, shapes, slices, D, weight_mask,
        n_epochs=n_epochs, lr=lr, l2_weight=l2_weight,
        l1_weight=l1_weight, l2_bias=l2_bias,
        seed=seed, verbose=verbose,
    )

    if verbose:
        pred = _forward(theta, X, shapes, slices)
        mse = np.mean((pred - y) ** 2)
        print(f"  Train MSE: {mse:.4f}  RMSE: {np.sqrt(mse):.4f}")

    if verbose:
        print("  Computing empirical Fisher...")

    if D > 2000:
        # Diagonal Fisher for large networks
        F_diag = _empirical_fisher_diagonal(
            theta, X, y, tau, shapes, slices,
            n_fisher=n_fisher, seed=seed,
        )
        prior_precision = np.zeros(D)
        for l, (w_sl, b_sl) in enumerate(slices):
            prior_precision[w_sl] = 1.0 / sigma_w_layers[l] ** 2
            prior_precision[b_sl] = 1.0 / sigma_b_layers[l] ** 2

        diag_vals = np.clip(F_diag + prior_precision, fisher_reg, None)
        Sigma_inv = np.diag(diag_vals)

        if verbose:
            print(f"  Σ⁻¹ diagonal range: "
                  f"[{diag_vals.min():.4f}, {diag_vals.max():.4f}]")
    else:
        # Block-diagonal Fisher for small networks
        F = _empirical_fisher(
            theta, X, y, tau, shapes, slices,
            n_fisher=n_fisher, seed=seed,
        )
        Sigma_inv = np.zeros((D, D))
        for l, (w_sl, b_sl) in enumerate(slices):
            idx = np.concatenate([
                np.arange(w_sl.start, w_sl.stop),
                np.arange(b_sl.start, b_sl.stop),
            ])
            block = 0.5 * (F[np.ix_(idx, idx)] + F[np.ix_(idx, idx)].T)

            # Add prior precision to diagonal
            for k, i in enumerate(idx):
                if weight_mask[i]:
                    block[k, k] += 1.0 / sigma_w_layers[l] ** 2
                else:
                    block[k, k] += 1.0 / sigma_b_layers[l] ** 2

            # Regularize if needed
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
