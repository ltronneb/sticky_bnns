"""
Sticky validation targets with known marginals.

"""
import numpy as np
import autograd.numpy as anp
from autograd import grad
from scipy import special, integrate

from .base import Target

# =====================================================================
# Spike-and-slab Gaussian (sticky sampler validation)
# =====================================================================

def spike_and_slab_gaussian(D=5, sigma=1.0, spike_weights=None, seed=42):
    """
    Independent Gaussian slab with known spike weights.

    The true marginal for coordinate i is:
        pi(beta_i) = w_i * delta_0 + (1 - w_i) * N(0, sigma_i^2)

    so the sampler should:
      - freeze coordinate i for fraction w_i of the time
      - produce N(0, sigma_i^2) samples when active

    kappa is derived from (w_i, sigma_i) via the spike-and-slab formula,
    and is stored in target.meta['kappa'] for use with build_kappa(kind='array').
    """
    if spike_weights is None:
        spike_weights = np.array([0.8, 0.6, 0.4, 0.2, 0.1])[:D]

    sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=float), (D,)).copy()
    w = np.asarray(spike_weights[:D], dtype=float)

    # kappa consistent with the spike weights
    kappa = (w / (1 - w)) / (sigma_arr * np.sqrt(2 * np.pi))

    # Target is just the Gaussian slab (the spike is handled by sticky dynamics)
    precision = 1.0 / sigma_arr**2
    precision_anp = anp.array(precision)

    def E(beta):
        return 0.5 * anp.sum(precision_anp * beta**2)

    gradE_fn = grad(E)

    # Marginals for the continuous part (what you see when not frozen)
    marginal_grids = {}
    for i in range(D):
        sd = sigma_arr[i]
        grid = np.linspace(-4 * sd, 4 * sd, 500)
        pdf = np.exp(-0.5 * grid**2 / sd**2) / np.sqrt(2 * np.pi * sd**2)
        marginal_grids[i] = {
            "grid": grid,
            "pdf": pdf,
            "label": f"$\\beta_{{{i+1}}}$",
            "spike_weight": float(w[i]),
        }

    return Target(
        name=f"spike_slab_gaussian_D{D}",
        task_type="validation",
        D=D,
        E=E,
        gradE=gradE_fn,
        x_ref=np.zeros(D),
        Sigma_inv=np.diag(precision),
        meta={
            "sigma": sigma_arr,
            "spike_weights": w,
            "kappa": kappa,
            "marginal_grids": marginal_grids,
            "preprocess_method": "manual",
        },
    )