"""
Warm-start for the Sticky Boomerang BNN sampler.

Trains a neural network with Adam, then extracts:
  - x_ref      : trained parameters
  - Sigma_inv  : block-diagonal empirical Fisher + prior precision
  - E, gradE   : energy and gradient with per-layer prior variances
  - sigma_w_layers : per-layer slab σ (from trained weight magnitudes)

κ is NOT set here — that's a prior choice you make separately.

Usage
-----
    from bnn_warmstart import warmstart_bnn

    ws = warmstart_bnn(X, y, layer_sizes=[5, 10, 1])

    # Set κ from your prior
    w_prior = 0.2
    sigma_w = np.mean(ws["sigma_w_layers"])
    kappa = np.empty(ws["D"])
    kappa[ws["weight_mask"]] = w_prior / (1 - w_prior) / (sigma_w * np.sqrt(2*np.pi))
    kappa[~ws["weight_mask"]] = 1e6

    sampler = StickyBoomerangSampler(
        E=ws["E"], N=50000, D=ws["D"],
        grad_target=ws["gradE"], kappa=kappa,
    )
    sampler.preprocess(method="manual", x_ref=ws["x_ref"], Sigma_inv=ws["Sigma_inv"])
"""

import numpy as np
import autograd.numpy as anp


# ================================================================
# Architecture helpers
# ================================================================

def make_architecture(layer_sizes):
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


# ================================================================
# Forward / gradient (pure numpy)
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
    return (h @ W + b).ravel()


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
    logits = (h @ W + b).ravel()
    p = _sigmoid(logits)
    delta = (p - y).reshape(-1, 1)
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
# Adam optimizer
# ================================================================

def _train_adam(X, y, shapes, slices, D, weight_mask,
                n_epochs=3000, lr=3e-3, l2_weight=1e-3, l2_bias=1e-4,
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
            g[~weight_mask] += 2.0 * l2_bias * theta[~weight_mask]

            t_adam = epoch * ((n + batch_size - 1) // batch_size) + start // batch_size + 1
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**t_adam)
            v_hat = v / (1 - beta2**t_adam)
            theta -= lr * m_hat / (np.sqrt(v_hat) + eps)

        logits = _forward(theta, X, shapes, slices)
        loss = np.sum(np.logaddexp(0.0, logits) - y * logits)
        if loss < best_loss:
            best_loss = loss
            theta_best = theta.copy()

        if verbose and (epoch % 500 == 0 or epoch == n_epochs - 1):
            acc = np.mean((_sigmoid(logits) > 0.5).astype(float) == y)
            print(f"    epoch {epoch:5d}/{n_epochs}  nll={loss:.1f}  acc={acc:.3f}")

    return theta_best


# ================================================================
# Empirical Fisher
# ================================================================

def _empirical_fisher(theta, X, y, shapes, slices, n_fisher=500):
    n = len(y)
    D = len(theta)
    n_fisher = min(n, n_fisher)
    idx = np.random.choice(n, size=n_fisher, replace=False)
    F = np.zeros((D, D))
    for i in idx:
        gi = _grad_nll(theta, X[i:i+1], y[i:i+1], shapes, slices)
        F += np.outer(gi, gi)
    F /= n_fisher
    return F


# ================================================================
# Build E(θ) and gradE(θ)
# ================================================================

def _make_E_and_grad(X, y, layer_sizes, sigma_w_layers, sigma_b_layers):
    shapes, slices, D, weight_mask = make_architecture(layer_sizes)
    X_a, y_a = anp.array(X), anp.array(y)

    precision = np.zeros(D)
    for l, (w_sl, b_sl) in enumerate(slices):
        precision[w_sl] = 1.0 / sigma_w_layers[l]**2
        precision[b_sl] = 1.0 / sigma_b_layers[l]**2
    precision_anp = anp.array(precision)

    def _fwd(theta, X_in, sh, sl):
        params = []
        for (w_shape, _), (w_s, b_s) in zip(sh, sl):
            params.append((anp.reshape(theta[w_s], w_shape), theta[b_s]))
        h = X_in
        for W, b in params[:-1]:
            h = anp.tanh(h @ W + b)
        Wl, bl = params[-1]
        return (h @ Wl + bl).ravel()

    def E(theta):
        logits = _fwd(theta, X_a, shapes, slices)
        nll = anp.sum(anp.logaddexp(0.0, logits) - y_a * logits)
        return nll + 0.5 * anp.sum(precision_anp * theta**2)

    if len(layer_sizes) == 3:
        n_data, d_in = X.shape
        H = layer_sizes[1]
        X_np, y_np = np.asarray(X), np.asarray(y)

        def gradE(theta):
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
            g += precision * theta
            return g
    else:
        from autograd import grad as agrad
        gradE = agrad(E)

    return E, gradE, shapes, slices, D, weight_mask


# ================================================================
# Main entry point
# ================================================================

def warmstart_bnn(X, y, layer_sizes, n_epochs=3000, lr=3e-3,
                   l2_weight=1e-3, sigma_b=1.0, fisher_reg=1e-3,
                   n_fisher=500, seed=42, verbose=True):
    """
    Warm-start: train with Adam → extract x_ref and Σ⁻¹.

    Returns dict with: x_ref, Sigma_inv, E, gradE, D,
        shapes, slices, weight_mask, sigma_w_layers, sigma_b_layers
    """
    shapes, slices, D, weight_mask = make_architecture(layer_sizes)

    print(f"Warm-start: {layer_sizes}  (D={D})")

    # ── Train ──
    print("  Training with Adam...")
    theta = _train_adam(X, y, shapes, slices, D, weight_mask,
                        n_epochs=n_epochs, lr=lr, l2_weight=l2_weight,
                        seed=seed, verbose=verbose)

    logits = _forward(theta, X, shapes, slices)
    acc = np.mean((_sigmoid(logits) > 0.5).astype(float) == y)
    print(f"  Train accuracy: {acc:.3f}")

    # ── Per-layer σ_w from trained weights ──
    sigma_w_layers, sigma_b_layers = [], []
    for l, (w_sl, b_sl) in enumerate(slices):
        sigma_w = max(np.std(theta[w_sl]), 0.1)
        sigma_w_layers.append(sigma_w)
        sigma_b_layers.append(sigma_b)
    print(f"  σ_w per layer: {[f'{s:.3f}' for s in sigma_w_layers]}")

    # ── Block-diagonal Fisher + prior ──
    print("  Computing Fisher...")
    F = _empirical_fisher(theta, X, y, shapes, slices, n_fisher)
    Sigma_inv = np.zeros((D, D))
    for l, (w_sl, b_sl) in enumerate(slices):
        idx = list(range(w_sl.start, w_sl.stop)) + list(range(b_sl.start, b_sl.stop))
        idx = np.array(idx)
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

    eigvals = np.linalg.eigvalsh(Sigma_inv)
    print(f"  Σ⁻¹ eigenvalues: [{eigvals.min():.4f}, {eigvals.max():.4f}]")

    # ── Build E and gradE ──
    E_fn, gradE_fn, shapes, slices, D, weight_mask = _make_E_and_grad(
        X, y, layer_sizes, sigma_w_layers, sigma_b_layers,
    )

    print(f"  Ready. (x_ref acc={acc:.3f})")

    return {
        "x_ref": theta,
        "Sigma_inv": Sigma_inv,
        "E": E_fn,
        "gradE": gradE_fn,
        "D": D,
        "shapes": shapes,
        "slices": slices,
        "weight_mask": weight_mask,
        "sigma_w_layers": sigma_w_layers,
        "sigma_b_layers": sigma_b_layers,
    }