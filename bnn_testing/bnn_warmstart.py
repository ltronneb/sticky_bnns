# """
# Warm-start for the Sticky Boomerang BNN sampler.

# Trains a neural network with Adam + L1 penalty on weights,
# then extracts:
#   - x_ref      : trained parameters (MAP-ish estimate)
#   - Sigma_inv  : empirical Fisher / Hessian approximation
#   - kappa      : per-parameter sticky rates from learned sparsity pattern
#   - E, gradE   : energy and gradient with per-layer prior variances

# The L1 penalty during training encourages sparsity, giving us a
# data-driven estimate of which weights should be active. We use the
# resulting weight magnitudes to set per-layer σ_w and inform κ.

# Usage
# -----
#     from bnn_warmstart import warmstart_bnn

#     ws = warmstart_bnn(X, y, layer_sizes=[5, 10, 1], n_epochs=2000)

#     sampler = StickyBoomerangSampler(
#         E=ws["E"], N=50000, D=ws["D"],
#         grad_target=ws["gradE"], kappa=ws["kappa"],
#     )
#     sampler.preprocess(method="manual", x_ref=ws["x_ref"], Sigma_inv=ws["Sigma_inv"])
#     sampler.sample_auto()
# """

# import numpy as np
# import autograd.numpy as anp
# from copy import deepcopy


# # ================================================================
# # Architecture helpers (consistent with your bnn_target.py)
# # ================================================================

# def make_architecture(layer_sizes):
#     shapes, slices = [], []
#     weight_mask_parts = []
#     offset = 0
#     for i in range(len(layer_sizes) - 1):
#         fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
#         w_size = fan_in * fan_out
#         b_size = fan_out
#         w_sl = slice(offset, offset + w_size)
#         offset += w_size
#         b_sl = slice(offset, offset + b_size)
#         offset += b_size
#         shapes.append(((fan_in, fan_out), (fan_out,)))
#         slices.append((w_sl, b_sl))
#         weight_mask_parts.append(np.ones(w_size, dtype=bool))
#         weight_mask_parts.append(np.zeros(b_size, dtype=bool))
#     D = offset
#     weight_mask = np.concatenate(weight_mask_parts)
#     return shapes, slices, D, weight_mask


# # ================================================================
# # Forward / loss / gradient (pure numpy, no autograd needed)
# # ================================================================

# def _forward(theta, X, shapes, slices):
#     """Forward pass: tanh hidden layers, raw logit output."""
#     h = X
#     for l in range(len(shapes) - 1):
#         w_sl, b_sl = slices[l]
#         W = theta[w_sl].reshape(shapes[l][0])
#         b = theta[b_sl]
#         h = np.tanh(h @ W + b)
#     w_sl, b_sl = slices[-1]
#     W = theta[w_sl].reshape(shapes[-1][0])
#     b = theta[b_sl]
#     return (h @ W + b).ravel()


# def _sigmoid(x):
#     return np.where(x >= 0,
#                     1.0 / (1.0 + np.exp(-x)),
#                     np.exp(x) / (1.0 + np.exp(x)))


# def _nll(theta, X, y, shapes, slices):
#     """Binary cross-entropy."""
#     logits = _forward(theta, X, shapes, slices)
#     return np.sum(np.logaddexp(0.0, logits) - y * logits)


# def _grad_nll(theta, X, y, shapes, slices):
#     """Backprop gradient of NLL for arbitrary depth."""
#     n_layers = len(shapes)

#     # Forward pass storing activations
#     activations = [X]
#     pre_acts = []
#     h = X
#     for l in range(n_layers - 1):
#         w_sl, b_sl = slices[l]
#         W = theta[w_sl].reshape(shapes[l][0])
#         b = theta[b_sl]
#         z = h @ W + b
#         h = np.tanh(z)
#         pre_acts.append(z)
#         activations.append(h)

#     # Output layer
#     w_sl, b_sl = slices[-1]
#     W = theta[w_sl].reshape(shapes[-1][0])
#     b = theta[b_sl]
#     logits = (h @ W + b).ravel()

#     # Output delta
#     p = _sigmoid(logits)
#     delta = (p - y).reshape(-1, 1)

#     # Backprop
#     g = np.zeros_like(theta)

#     # Last layer
#     dW = activations[-1].T @ delta
#     db = delta.sum(axis=0)
#     g[slices[-1][0]] = dW.ravel()
#     g[slices[-1][1]] = db.ravel()

#     # Hidden layers (reverse)
#     for l in range(n_layers - 2, -1, -1):
#         W_next = theta[slices[l + 1][0]].reshape(shapes[l + 1][0])
#         delta = (delta @ W_next.T) * (1.0 - np.tanh(pre_acts[l])**2)
#         dW = activations[l].T @ delta
#         db = delta.sum(axis=0)
#         g[slices[l][0]] = dW.ravel()
#         g[slices[l][1]] = db.ravel()

#     return g


# # ================================================================
# # Adam optimizer with L1 proximal step on weights
# # ================================================================

# def train_adam(X, y, layer_sizes, n_epochs=2000, lr=1e-3,
#                l2_weight=1e-3, l1_weight=1e-3, l2_bias=1e-4,
#                batch_size=None, verbose=True, seed=42):
#     """
#     Train BNN with Adam + L1 on weights (proximal gradient step).

#     Parameters
#     ----------
#     X, y          : training data
#     layer_sizes   : e.g. [5, 10, 1]
#     n_epochs      : training epochs
#     lr            : learning rate
#     l2_weight     : L2 penalty on weights (Gaussian prior)
#     l1_weight     : L1 penalty on weights (encourages sparsity)
#     l2_bias       : L2 penalty on biases
#     batch_size    : mini-batch size (None = full batch)
#     verbose       : print progress

#     Returns
#     -------
#     theta_best    : best parameters found
#     history       : dict with loss, accuracy, sparsity traces
#     """
#     rng = np.random.default_rng(seed)
#     shapes, slices, D, weight_mask = make_architecture(layer_sizes)
#     n = len(y)

#     if batch_size is None:
#         batch_size = n

#     # Glorot init
#     theta = np.zeros(D)
#     for l, (w_sl, b_sl) in enumerate(slices):
#         fan_in, fan_out = shapes[l][0]
#         std = np.sqrt(2.0 / (fan_in + fan_out))
#         theta[w_sl] = rng.normal(0, std, size=w_sl.stop - w_sl.start)
#         theta[b_sl] = 0.01

#     # Adam state
#     m = np.zeros(D)
#     v = np.zeros(D)
#     beta1, beta2, eps = 0.9, 0.999, 1e-8

#     # Tracking
#     history = {"loss": [], "accuracy": [], "sparsity": []}
#     best_loss = np.inf
#     theta_best = theta.copy()

#     for epoch in range(n_epochs):
#         # Shuffle for mini-batches
#         perm = rng.permutation(n)
#         epoch_loss = 0.0

#         for start in range(0, n, batch_size):
#             idx = perm[start:start + batch_size]
#             X_b, y_b = X[idx], y[idx]
#             bs = len(idx)

#             # Gradient of NLL (scaled to full dataset)
#             g = _grad_nll(theta, X_b, y_b, shapes, slices) * (n / bs)

#             # Add L2 gradient
#             g[weight_mask] += 2.0 * l2_weight * theta[weight_mask]
#             g[~weight_mask] += 2.0 * l2_bias * theta[~weight_mask]

#             # Adam update
#             t_adam = epoch * ((n + batch_size - 1) // batch_size) + start // batch_size + 1
#             m = beta1 * m + (1 - beta1) * g
#             v = beta2 * v + (1 - beta2) * g**2
#             m_hat = m / (1 - beta1**t_adam)
#             v_hat = v / (1 - beta2**t_adam)
#             theta -= lr * m_hat / (np.sqrt(v_hat) + eps)

#             # Proximal step for L1 on weights only (soft thresholding)
#             w_vals = theta[weight_mask]
#             threshold = lr * l1_weight
#             theta[weight_mask] = np.sign(w_vals) * np.maximum(np.abs(w_vals) - threshold, 0.0)

#         # Compute full loss for tracking
#         logits = _forward(theta, X, shapes, slices)
#         nll = np.sum(np.logaddexp(0.0, logits) - y * logits)
#         l2_term = l2_weight * np.sum(theta[weight_mask]**2) + l2_bias * np.sum(theta[~weight_mask]**2)
#         l1_term = l1_weight * np.sum(np.abs(theta[weight_mask]))
#         total_loss = nll + l2_term + l1_term

#         preds = (_sigmoid(logits) > 0.5).astype(float)
#         acc = np.mean(preds == y)
#         sparsity = np.mean(np.abs(theta[weight_mask]) < 1e-6)

#         history["loss"].append(total_loss)
#         history["accuracy"].append(acc)
#         history["sparsity"].append(sparsity)

#         if total_loss < best_loss:
#             best_loss = total_loss
#             theta_best = theta.copy()

#         if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
#             print(f"  epoch {epoch:5d}/{n_epochs}  loss={total_loss:.2f}  "
#                   f"acc={acc:.3f}  sparsity={sparsity:.1%}")

#     return theta_best, history


# # ================================================================
# # Extract reference measure from trained network
# # ================================================================

# def _empirical_fisher(theta, X, y, shapes, slices, n_fisher=None):
#     """
#     Empirical Fisher information matrix:
#         F = (1/n) Σ_i  ∇log p(y_i|x_i,θ) ∇log p(y_i|x_i,θ)^T

#     This is a PSD approximation to the Hessian that's cheap to compute
#     and doesn't require second derivatives.
#     """
#     n = len(y)
#     D = len(theta)
#     if n_fisher is None:
#         n_fisher = min(n, 500)  # subsample for speed

#     idx = np.random.choice(n, size=n_fisher, replace=False)
#     F = np.zeros((D, D))

#     for i in idx:
#         gi = _grad_nll(theta, X[i:i+1], y[i:i+1], shapes, slices)
#         F += np.outer(gi, gi)

#     F /= n_fisher
#     return F


# def extract_reference(theta_trained, X, y, layer_sizes, shapes, slices,
#                        weight_mask, sigma_b=1.0, fisher_reg=1e-3,
#                        n_fisher=500):
#     """
#     From a trained network, extract:
#       - x_ref      : the trained parameters
#       - Sigma_inv  : block-diagonal Fisher + prior precision
#       - sigma_w_layers : per-layer slab σ estimated from trained weight magnitudes
#       - sigma_b_layers : per-layer bias σ

#     The per-layer σ_w is set from the empirical std of the
#     non-zero weights in that layer, floored at a minimum.
#     """
#     n_layers = len(shapes)

#     # --- Per-layer σ from trained weights ---
#     sigma_w_layers = []
#     sigma_b_layers = []
#     for l, (w_sl, b_sl) in enumerate(slices):
#         w_vals = theta_trained[w_sl]
#         # Use std of non-negligible weights as the slab width
#         active = np.abs(w_vals) > 1e-4
#         if active.sum() > 2:
#             sigma_w = max(np.std(w_vals[active]), 0.1)
#         else:
#             # Very sparse layer — use Glorot as fallback
#             fan_in, fan_out = shapes[l][0]
#             sigma_w = np.sqrt(2.0 / (fan_in + fan_out))
#         sigma_w_layers.append(sigma_w)
#         sigma_b_layers.append(sigma_b)

#     # --- Build prior precision per parameter ---
#     D = len(theta_trained)
#     prior_precision = np.zeros(D)
#     for l, (w_sl, b_sl) in enumerate(slices):
#         prior_precision[w_sl] = 1.0 / sigma_w_layers[l]**2
#         prior_precision[b_sl] = 1.0 / sigma_b_layers[l]**2

#     # --- Empirical Fisher (block-diagonal) ---
#     F_full = _empirical_fisher(theta_trained, X, y, shapes, slices, n_fisher)

#     # Block-diagonalise and add prior precision
#     Sigma_inv = np.zeros((D, D))
#     for (w_sl, b_sl) in slices:
#         idx = list(range(w_sl.start, w_sl.stop)) + list(range(b_sl.start, b_sl.stop))
#         idx = np.array(idx)
#         F_block = F_full[np.ix_(idx, idx)]
#         F_block = 0.5 * (F_block + F_block.T)

#         # Add prior precision to diagonal
#         P_block = np.diag(prior_precision[idx])
#         H_block = F_block + P_block

#         # Regularise
#         eigvals = np.linalg.eigvalsh(H_block)
#         if eigvals.min() < fisher_reg:
#             H_block += (fisher_reg - eigvals.min() + 1e-6) * np.eye(len(idx))

#         Sigma_inv[np.ix_(idx, idx)] = H_block

#     return Sigma_inv, sigma_w_layers, sigma_b_layers


# # ================================================================
# # Build E and gradE with per-layer priors
# # ================================================================

# def make_E_and_grad(X, y, layer_sizes, sigma_w_layers, sigma_b_layers):
#     """Build E(θ) and gradE(θ) using per-layer prior variances."""
#     shapes, slices, D, weight_mask = make_architecture(layer_sizes)
#     X_a = anp.array(X)
#     y_a = anp.array(y)

#     # Per-parameter precision
#     precision = np.zeros(D)
#     for l, (w_sl, b_sl) in enumerate(slices):
#         precision[w_sl] = 1.0 / sigma_w_layers[l]**2
#         precision[b_sl] = 1.0 / sigma_b_layers[l]**2
#     precision_anp = anp.array(precision)

#     def _fwd(theta, X_in, sh, sl):
#         params = []
#         for (w_shape, _), (w_s, b_s) in zip(sh, sl):
#             W = anp.reshape(theta[w_s], w_shape)
#             b = theta[b_s]
#             params.append((W, b))
#         h = X_in
#         for W, b in params[:-1]:
#             h = anp.tanh(h @ W + b)
#         Wl, bl = params[-1]
#         return (h @ Wl + bl).ravel()

#     def E(theta):
#         logits = _fwd(theta, X_a, shapes, slices)
#         nll = anp.sum(anp.logaddexp(0.0, logits) - y_a * logits)
#         nprior = 0.5 * anp.sum(precision_anp * theta**2)
#         return nll + nprior

#     # Manual gradient for 1-hidden-layer
#     if len(layer_sizes) == 3:
#         n_data, d_in = X.shape
#         H = layer_sizes[1]
#         X_np = np.asarray(X)
#         y_np = np.asarray(y)

#         def gradE(theta):
#             W1 = theta[slices[0][0]].reshape(d_in, H)
#             b1 = theta[slices[0][1]]
#             W2 = theta[slices[1][0]].reshape(H, 1)
#             b2 = theta[slices[1][1]]
#             z1 = X_np @ W1 + b1
#             a1 = np.tanh(z1)
#             logits = (a1 @ W2 + b2).ravel()
#             p = _sigmoid(logits)
#             delta_out = (p - y_np).reshape(-1, 1)
#             dW2 = a1.T @ delta_out
#             db2 = delta_out.sum(axis=0)
#             dtanh = 1.0 - a1**2
#             delta_hidden = (delta_out @ W2.T) * dtanh
#             dW1 = X_np.T @ delta_hidden
#             db1 = delta_hidden.sum(axis=0)
#             g = np.zeros(D)
#             g[slices[0][0]] = dW1.ravel()
#             g[slices[0][1]] = db1
#             g[slices[1][0]] = dW2.ravel()
#             g[slices[1][1]] = db2
#             g += precision * theta
#             return g
#     else:
#         from autograd import grad as agrad
#         gradE = agrad(E)

#     return E, gradE, shapes, slices, D, weight_mask, precision


# # ================================================================
# # Compute κ from trained sparsity pattern
# # ================================================================

# def compute_kappa_from_training(theta_trained, weight_mask, slices, shapes,
#                                  sigma_w_layers, w_prior_base=0.5,
#                                  kappa_bias=1e6):
#     """
#     Data-driven κ: use the trained network's sparsity to inform
#     per-layer inclusion probability w_l, then compute κ.

#     For each layer:
#       w_l = fraction of weights that are non-negligible after training
#       κ_l = w_l / (1 - w_l) * 1 / (σ_w_l * √(2π))

#     If w_l is very close to 0 or 1, we clip to [0.05, 0.95]
#     to keep κ finite and sensible.

#     Parameters
#     ----------
#     w_prior_base : fallback inclusion probability if training
#                    doesn't give a clear signal
#     """
#     D = len(theta_trained)
#     kappa = np.empty(D)

#     print("  Per-layer κ from training:")
#     for l, (w_sl, b_sl) in enumerate(slices):
#         w_vals = theta_trained[w_sl]
#         n_weights = len(w_vals)

#         # Fraction of active weights after L1 training
#         active_frac = np.mean(np.abs(w_vals) > 1e-4)
#         # Clip to avoid κ = 0 or κ = ∞
#         w_l = np.clip(active_frac, 0.05, 0.95)

#         kappa_l = w_l / (1.0 - w_l) / (sigma_w_layers[l] * np.sqrt(2 * np.pi))

#         kappa[w_sl] = kappa_l
#         kappa[b_sl] = kappa_bias

#         print(f"    layer {l+1}: {int(active_frac * n_weights)}/{n_weights} active "
#               f"({active_frac:.0%}) → w={w_l:.2f}, σ={sigma_w_layers[l]:.3f}, κ={kappa_l:.4f}")

#     return kappa


# # ================================================================
# # Main entry point
# # ================================================================

# def warmstart_bnn(X, y, layer_sizes, n_epochs=2000, lr=1e-3,
#                    l1_weight=1e-2, l2_weight=1e-3, sigma_b=1.0,
#                    batch_size=None, fisher_reg=1e-3, n_fisher=500,
#                    seed=42, verbose=True):
#     """
#     Complete warm-start pipeline:
#       1. Train with Adam + L1 → sparse point estimate
#       2. Extract per-layer σ_w from trained weight magnitudes
#       3. Build empirical Fisher → block-diagonal Σ⁻¹
#       4. Compute data-driven κ from observed sparsity
#       5. Build E(θ), gradE(θ) with correct per-layer priors

#     Returns
#     -------
#     dict with keys:
#         x_ref, Sigma_inv, kappa, E, gradE, D,
#         shapes, slices, weight_mask,
#         sigma_w_layers, sigma_b_layers,
#         theta_trained, train_history
#     """
#     shapes, slices, D, weight_mask = make_architecture(layer_sizes)

#     print("=" * 60)
#     print("BNN Warm-Start for Sticky Boomerang")
#     print("=" * 60)
#     print(f"Architecture: {layer_sizes}  (D = {D})")
#     print(f"Data: n = {len(y)}, d_in = {X.shape[1]}")
#     print()

#     # ── Step 1: Train with Adam + L1 ──
#     print("Step 1: Training with Adam + L1...")
#     theta_trained, history = train_adam(
#         X, y, layer_sizes, n_epochs=n_epochs, lr=lr,
#         l2_weight=l2_weight, l1_weight=l1_weight,
#         batch_size=batch_size, verbose=verbose, seed=seed,
#     )

#     # Training summary
#     logits_train = _forward(theta_trained, X, shapes, slices)
#     train_acc = np.mean((_sigmoid(logits_train) > 0.5).astype(float) == y)
#     n_zero = np.sum(np.abs(theta_trained[weight_mask]) < 1e-4)
#     n_weights = weight_mask.sum()
#     print(f"\n  Final train accuracy: {train_acc:.3f}")
#     print(f"  Weights pruned: {n_zero}/{n_weights} ({n_zero/n_weights:.0%})")
#     print()

#     # ── Step 2: Extract per-layer σ_w ──
#     print("Step 2: Extracting per-layer prior variances...")
#     Sigma_inv, sigma_w_layers, sigma_b_layers = extract_reference(
#         theta_trained, X, y, layer_sizes, shapes, slices, weight_mask,
#         sigma_b=sigma_b, fisher_reg=fisher_reg, n_fisher=n_fisher,
#     )
#     for l, sw in enumerate(sigma_w_layers):
#         print(f"  layer {l+1}: σ_w = {sw:.4f}")
#     print()

#     # ── Step 3: Σ⁻¹ diagnostics ──
#     print("Step 3: Block-diagonal Fisher + prior precision...")
#     eigvals = np.linalg.eigvalsh(Sigma_inv)
#     print(f"  Σ⁻¹ eigenvalue range: [{eigvals.min():.4f}, {eigvals.max():.4f}]")
#     cond = eigvals.max() / max(eigvals.min(), 1e-10)
#     print(f"  Condition number: {cond:.1f}")
#     print()

#     # ── Step 4: Data-driven κ ──
#     print("Step 4: Computing κ from training sparsity...")
#     kappa = compute_kappa_from_training(
#         theta_trained, weight_mask, slices, shapes, sigma_w_layers,
#     )
#     print()

#     # ── Step 5: Build E and gradE ──
#     print("Step 5: Building E(θ) and gradE(θ)...")
#     E_fn, gradE_fn, shapes, slices, D, weight_mask, precision = make_E_and_grad(
#         X, y, layer_sizes, sigma_w_layers, sigma_b_layers,
#     )

#     # Gradient check
#     from autograd import grad as agrad
#     theta_test = np.random.randn(D) * 0.1
#     g_auto = agrad(E_fn)(theta_test)
#     g_manual = gradE_fn(theta_test)
#     print(f"  Gradient check: max|diff| = {np.max(np.abs(g_auto - g_manual)):.2e}")
#     print()

#     # ── Summary ──
#     print("=" * 60)
#     print("Ready for sampler:")
#     print(f"  D = {D}  ({n_weights} weights, {D - n_weights} biases)")
#     print(f"  x_ref: trained network (train acc = {train_acc:.3f})")
#     print(f"  Σ⁻¹: block-diagonal Fisher ({len(slices)} blocks)")
#     print(f"  κ: data-driven from {n_zero}/{n_weights} pruned weights")
#     print("=" * 60)

#     return {
#         "x_ref": theta_trained,
#         "Sigma_inv": Sigma_inv,
#         "kappa": kappa,
#         "E": E_fn,
#         "gradE": gradE_fn,
#         "D": D,
#         "shapes": shapes,
#         "slices": slices,
#         "weight_mask": weight_mask,
#         "sigma_w_layers": sigma_w_layers,
#         "sigma_b_layers": sigma_b_layers,
#         "theta_trained": theta_trained,
#         "train_history": history,
#     }


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