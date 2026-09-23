"""
LBBNN -- Latent Binary Bayesian Neural Networks by variational inference,
after Hubin & Storvik, "Variational Inference for Bayesian Neural Networks
under Model and Parameter Uncertainty".

A baseline for the sticky PDMP samplers in this repo: LBBNN puts a
spike-and-slab (inclusion indicator x Gaussian) variational posterior on every
weight, so like the sticky samplers it does inference jointly over models and
parameters -- but by optimizing an ELBO rather than by sampling. Its per-weight
inclusion probabilities alpha are the direct analogue of a sticky sampler's
per-coordinate freeze fractions, which makes the two directly comparable on
sparsity as well as on predictive accuracy.

Ported from the reference implementation (standalone per-dataset scripts,
classification-only, fixed 3-layer architecture) and generalized here to
arbitrary depth and to Gaussian-likelihood regression -- see model.py for
the full list of changes.

Typical use, from a benchmark script:

    from sazz.gpu_friendly.lbbnn import LBBNNConfig, run_lbbnn

    cfg = LBBNNConfig(layer_sizes=[1, 100, 1], activation="tanh",
                      noise_std=0.3, temper=0.5, temper_prior=0.5)
    samples, elapsed, grad_evals, info = run_lbbnn(data, cfg, seed=42)
"""

from sazz.gpu_friendly.lbbnn.distributions import (
    BetaBinomial, Bernoulli, GaussGamma, Gaussian, GaussianPrior,
)
from sazz.gpu_friendly.lbbnn.model import BayesianLinear, LBBNN, LBBNNConfig
from sazz.gpu_friendly.lbbnn.train import run_lbbnn, train_lbbnn

__all__ = [
    "Gaussian", "Bernoulli", "GaussGamma", "GaussianPrior", "BetaBinomial",
    "BayesianLinear", "LBBNN", "LBBNNConfig",
    "train_lbbnn", "run_lbbnn",
]
