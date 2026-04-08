import numpy as np
from bnn_testing.bnn_warmstart import warmstart_bnn, _grad_nll

def warmstart_bnn_subsampled(X, y, layer_sizes, batch_size=100, **kwargs):
    """
    Same as warmstart_bnn but returns subsample-ready functions.
    """
    ws = warmstart_bnn(X, y, layer_sizes, **kwargs)
    
    n_data = len(y)
    shapes = ws["shapes"]
    slices = ws["slices"]
    sigma_w_layers = ws["sigma_w_layers"]
    sigma_b_layers = ws["sigma_b_layers"]
    
    # Precision vector for the prior
    D = ws["D"]
    weight_mask = ws["weight_mask"]
    precision = np.zeros(D)
    for l, (w_sl, b_sl) in enumerate(slices):
        precision[w_sl] = 1.0 / sigma_w_layers[l]**2
        precision[b_sl] = 1.0 / ws["sigma_b_layers"][l]**2
    
    # Full gradient at x_ref (computed once, this is your control variate)
    cv_grad = ws["gradE"](ws["x_ref"])
    
    # Per-datapoint gradient function (returns gradient of -log p(y_i|θ))
    def grad_nll_single(theta, i):
        """Gradient of NLL for datapoint i only."""
        return _grad_nll(theta, X[i:i+1], y[i:i+1], shapes, slices)
    
    def grad_subsample(theta, indices):
        """
        Control-variate subsampled gradient estimate.
        
        Returns: (N/B) * Σ_{i∈S} [∇nll_i(θ) - ∇nll_i(x_ref)] + ∇U(x_ref)
        
        This equals ∇U(θ) exactly when θ = x_ref.
        """
        B = len(indices)
        g_batch = np.zeros(D)
        for i in indices:
            g_batch += grad_nll_single(theta, i) - grad_nll_single(ws["x_ref"], i)
        # Scale likelihood part, add back full CV baseline
        return (n_data / B) * g_batch + cv_grad
    
    def grad_subsample_batched(theta, batch_size=batch_size):
        """Draw a random batch and return CV gradient estimate."""
        indices = np.random.choice(n_data, size=batch_size, replace=False)
        return grad_subsample(theta, indices)
    
    # Precompute per-datapoint gradient norms at x_ref for bounding
    print("  Precomputing per-datapoint gradient norms for bounds...")
    grad_norms = np.zeros(n_data)
    for i in range(n_data):
        gi = grad_nll_single(ws["x_ref"], i)
        grad_norms[i] = np.linalg.norm(gi)
    
    # This gives you a bound on the CV noise
    # Var of CV estimator ≈ (N/B)^2 * (1/B) * Var(∇nll_i(x_ref))
    # But near x_ref this is small
    mean_grad_norm = np.mean(grad_norms)
    std_grad_norm = np.std(grad_norms)
    
    ws.update({
        "grad_subsample": grad_subsample_batched,
        "cv_grad": cv_grad,
        "n_data": n_data,
        "batch_size": batch_size,
        "grad_norms_at_ref": grad_norms,
        "mean_grad_norm": mean_grad_norm,
        "std_grad_norm": std_grad_norm,
        "precision": precision,
    })
    
    return ws