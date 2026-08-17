"""
gpu_friendly/scripts/toy_grid_demo.py

End-to-end validation of the grid-based (Andral & Kamatani 2024) Boomerang
sampler: build a small logistic-regression target using the gpu_friendly
model layer, find a reference measure via the EXISTING (read-only,
unmodified) sazz.utils.warmup.find_reference_glm, run GridBoomerangSampler,
and report diagnostics.

Usage:
    python -m sazz.gpu_friendly.scripts.toy_grid_demo
"""

import math

import torch
import torch.nn as nn

from sazz.utils.warmup import find_reference_glm

from sazz.gpu_friendly.models.priors import make_gaussian_prior
from sazz.gpu_friendly.models.likelihoods import make_bernoulli_likelihood
from sazz.gpu_friendly.models.model import make_energy, to_param_dict
from sazz.gpu_friendly.samplers.grid_boomerang import GridBoomerangSampler


def build_target(N: int = 300, D_in: int = 8, seed: int = 0, dtype=torch.float64):
    torch.manual_seed(seed)

    module = nn.Linear(D_in, 1, bias=True).to(dtype)
    names = [n for n, _ in module.named_parameters()]
    shapes = [p.shape for _, p in module.named_parameters()]
    D = sum(s.numel() for s in shapes)
    param_dict_fn = to_param_dict(names, shapes)

    true_beta = torch.randn(D, dtype=dtype)
    X = torch.randn(N, D_in, dtype=dtype)
    with torch.no_grad():
        logits = torch.func.functional_call(module, param_dict_fn(true_beta), (X,)).squeeze(-1)
        probs = torch.sigmoid(logits)
        y = torch.bernoulli(probs)

    log_prior = make_gaussian_prior(torch.ones(D, dtype=dtype))
    log_lik = make_bernoulli_likelihood(module, param_dict_fn, X, y)
    energy = make_energy(log_prior, log_lik)

    return energy, D, X, y


def main():
    dtype = torch.float64
    energy, D, X, y = build_target(dtype=dtype)

    print(f"Target: logistic regression, D={D}")
    print("Finding reference measure (MAP + Hessian)...")
    x_ref, Sigma_inv = find_reference_glm(energy, D, dtype=dtype, n_steps=500)
    print(f"x_ref: {x_ref}")

    sampler = GridBoomerangSampler(
        grad_target=torch.func.grad(energy),
        D=D,
        refresh_rate=0.2,
        grid_t_max_init=math.pi / 4,
        n_segments=20,
        grid_spacing=math.pi / 16,
        dtype=dtype,
    )
    sampler.preprocess(x_ref=x_ref, Sigma_inv=Sigma_inv)

    result = sampler.sample(N=2000, diagnostics=True)

    positions = result["positions"]
    print("\n=== Posterior summary ===")
    print("Mean:", positions.mean(dim=0))
    print("Std :", positions.std(dim=0))

    t_max_log = result["grid_t_max_log"]
    print(f"\ngrid_t_max: start={t_max_log[0]:.4f}, end={t_max_log[-1]:.4f}, "
          f"min={min(t_max_log):.4f}, max={max(t_max_log):.4f}")

    if result["bound_violations"] > 0:
        print(f"\n*** {result['bound_violations']} bound violations occurred during this run. ***")


if __name__ == "__main__":
    main()
