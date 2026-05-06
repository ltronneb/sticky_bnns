"""Minimal example: PDMP sampling for a Bayesian feedforward neural network.

Given training data (X, y) and a layer specification, this script builds a
fully-connected BNN with Gaussian priors on the weights and samples from
its posterior using one of four PDMP samplers.

Usage as a script:
    python sample_bnn.py

Usage as a library:
    from sample_bnn import sample_bnn

    samples = sample_bnn(
        X_train, y_train,
        layer_sizes=[X_train.shape[1], 50, 1],
        sampler="sticky_boomerang",
        n_skeleton=10_000,
    )
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from sazz.samplers.AutomaticBoomerangSampler import AutomaticBoomerangSampler
from sazz.samplers.AutomaticZigZagSampler_torch import AutomaticZigZagSampler
from sazz.samplers.StickyAutomaticBoomerangSampler import (
    StickyAutomaticBoomerangSampler,
)
from sazz.samplers.StickyAutomaticZigZagSampler_torch import (
    StickyAutomaticZigZagSampler,
)
from sazz.models.bnn_torch import ModuleGaussianPrior, ModuleGaussianLikelihood
from sazz.models.make_models import TorchTarget
from sazz.models.models_torch import BayesianModel
from sazz.models.networks import FFN
from sazz.utils.bnn_modular_utils import (
    ParamSpec, build_prior_precision, make_kappa_from_inclusion,
)
from sazz.utils.warmup import find_reference_bnn, tune_refresh_rate
from sazz.utils.sampling import (
    resample_boomerang_path, resample_boomerang_path_sticky,
    resample_zigzag_path, resample_zigzag_path_sticky,
)


SAMPLER_NAMES = ("zigzag", "sticky_zigzag", "boomerang", "sticky_boomerang")

# ===========================================================================
# Build the target
# ===========================================================================

def build_target(
    X: Tensor, y: Tensor, *,
    layer_sizes: Sequence[int],
    activation: str = "relu",
    noise_std: float = 0.3,
    prior_std_weight: float = 1.0,
    prior_std_bias: float = 1.0,
    fan_in_scaling: bool = True,
    adam_steps: int = 5_000,
    dtype=torch.float64,
):
    """Build a regression BNN target: nn.Module + Gaussian prior + Gaussian likelihood.

    Returns a TorchTarget the samplers can consume. Swap in a different
    nn.Module (e.g. a CNN) and the rest of this function works unchanged —
    everything downstream of `ParamSpec.from_module` is architecture-agnostic.
    """
    X = X.to(dtype=dtype)
    y = y.to(dtype=dtype)

    module = FFN(layer_sizes, activation=activation).to(dtype=dtype)
    spec = ParamSpec.from_module(module)
    init_beta = torch.cat([p.detach().flatten() for p in module.parameters()])

    prec = build_prior_precision(
        spec, prior_std_weight, prior_std_bias, fan_in_scaling, dtype, "cpu",
    )
    prior = ModuleGaussianPrior(prec)
    likelihood = ModuleGaussianLikelihood(module, spec, X, y, noise_std)
    model = BayesianModel(prior, likelihood)

    # MAP + diagonal Laplace as the sampler's reference / preconditioner.
    x_ref, Sigma_inv = find_reference_bnn(
        model.energy, spec.D, model=model, dtype=dtype, device="cpu",
        reference="laplace_diag", n_steps=adam_steps, lr=1e-2#, init_beta=init_beta
    )

    return TorchTarget(
        name=f"bnn_{'x'.join(map(str, layer_sizes))}_{activation}",
        D=spec.D,
        grad_target=model.grad_energy,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={"model": model, "spec": spec, "module": module},
    )


# ===========================================================================
# Build a sampler
# ===========================================================================

def build_sampler(
    name: str, target, *,
    prior_std_weight: float = 1.0,
    prior_inclusion_weight: float = 0.5,
    fan_in_scaling: bool = True,
    refresh_rate: float = 1.0,
    t_max: float = 0.1,
    gamma: float = 0.01,
):
    """Instantiate one of the four PDMP samplers for the given target.

    Sticky variants use a kappa vector derived from the spike-and-slab
    inclusion-weight formula (see make_kappa_from_inclusion).
    Boomerang samplers preprocess with the target's reference and tune
    their refresh rate from a short pilot run.
    """
    common = dict(grad_target=target.grad_target, D=target.D, thinning="pli")
    spec = target.meta["spec"]

    if name == "zigzag":
        return AutomaticZigZagSampler(**common, t_max=t_max, gamma=gamma)

    if name == "boomerang":
        s = AutomaticBoomerangSampler(**common, refresh_rate=refresh_rate)
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
        info = tune_refresh_rate(s, n_pilot=2000)
        print(f"  tuned refresh_rate: "
              f"{info['lambda_r_old']:.3f} -> {info['lambda_r_new']:.3f}")
        return s

    # Sticky variants need a kappa vector and a per-coord can_freeze mask
    kappa = make_kappa_from_inclusion(
        spec=spec,
        prior_std_weight=prior_std_weight,
        prior_inclusion_weight=prior_inclusion_weight,
        fan_in_scaling=fan_in_scaling,
    )
    can_freeze = torch.repeat_interleave(
        torch.tensor(spec.can_freeze, dtype=torch.bool),
        torch.tensor(spec.numels),
    )

    if name == "sticky_zigzag":
        return StickyAutomaticZigZagSampler(
            **common, t_max=t_max, gamma=gamma,
            kappa=kappa, can_freeze=can_freeze,
        )
    if name == "sticky_boomerang":
        s = StickyAutomaticBoomerangSampler(
            **common, refresh_rate=refresh_rate,
            kappa=kappa, can_freeze=can_freeze,
        )
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
        info = tune_refresh_rate(s, n_pilot=2000)
        print(f"  tuned refresh_rate: "
              f"{info['lambda_r_old']:.3f} -> {info['lambda_r_new']:.3f}")
        return s

    raise ValueError(f"Unknown sampler '{name}'. Choose from {SAMPLER_NAMES}.")


# ===========================================================================
# Resample the continuous-time path to discrete-time draws
# ===========================================================================

def resample_path(name: str, result: dict, x_ref: Tensor, *,
                  n_resample: int = 5_000, burnin_frac: float = 0.2):
    """Convert the continuous-time skeleton into a discrete-time chain
    of `n_resample` evenly time-spaced draws after burn-in."""
    pos = result["positions"].cpu().numpy()
    vel = result["velocities"].cpu().numpy()
    tim = result["times"].cpu().numpy()
    common = dict(N_resample=n_resample, burnin_frac=burnin_frac)
    x_ref_np = x_ref.cpu().numpy()

    if name == "zigzag":
        return resample_zigzag_path(pos, vel, tim, **common)
    if name == "sticky_zigzag":
        return resample_zigzag_path_sticky(pos, vel, tim, **common)
    if name == "boomerang":
        return resample_boomerang_path(pos, vel, tim, x_ref_np, **common)
    if name == "sticky_boomerang":
        return resample_boomerang_path_sticky(pos, vel, tim, x_ref_np, **common)
    raise ValueError(name)


# ===========================================================================
# Top-level convenience function
# ===========================================================================

def sample_bnn(
    X: Tensor, y: Tensor, *,
    layer_sizes: Sequence[int],
    activation: str = "relu",
    noise_std: float = 0.3,
    sampler: str = "sticky_boomerang",
    n_skeleton: int = 10_000,
    n_resample: int = 5_000,
    burnin_frac: float = 0.2,
    prior_std_weight: float = 1.0,
    prior_std_bias: float = 1.0,
    prior_inclusion_weight: float = 0.5,
    fan_in_scaling: bool = True,
    seed: int = 0,
):
    """Sample from the posterior of a regression BNN. Returns (samples, target).

    `samples` is a [n_resample, D] tensor of posterior draws, with parameters
    laid out in named_parameters() order. To compute predictions on new data:

        likelihood = target.meta["model"].likelihood
        preds = torch.stack([likelihood.predict(b, X_new) for b in samples])
    """
    if sampler not in SAMPLER_NAMES:
        raise ValueError(f"Unknown sampler '{sampler}'. Choose from {SAMPLER_NAMES}.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[1/3] Building target  ({sampler}, layers={list(layer_sizes)})")
    target = build_target(
        X, y, layer_sizes=layer_sizes, activation=activation,
        noise_std=noise_std,
        prior_std_weight=prior_std_weight, prior_std_bias=prior_std_bias,
        fan_in_scaling=fan_in_scaling,
    )
    print(f"      D = {target.D}")

    print(f"[2/3] Sampling {n_skeleton} skeleton events")
    s = build_sampler(
        sampler, target,
        prior_std_weight=prior_std_weight,
        prior_inclusion_weight=prior_inclusion_weight,
        fan_in_scaling=fan_in_scaling,
    )
    t0 = time.perf_counter()
    result = s.sample(N=n_skeleton, diagnostics=True)
    print(f"      skeletons done in {time.perf_counter() - t0:.1f}s")

    print(f"[3/3] Resampling path to {n_resample} draws (burn-in {burnin_frac:.0%})")
    samples_np = resample_path(
        sampler, result, target.x_ref,
        n_resample=n_resample, burnin_frac=burnin_frac,
    )
    print(f"      sampling done in {time.perf_counter() - t0:.1f}s")
    samples = torch.tensor(samples_np)

    return samples, target


# ===========================================================================
# Demo: a small synthetic regression problem
# ===========================================================================

if __name__ == "__main__":
    # Synthetic 1-D regression: y = sin(2*pi*x) + noise
    torch.manual_seed(0)
    N = 60
    x = torch.linspace(-1, 1, N).unsqueeze(-1)
    y = (2 * torch.pi * x.squeeze()).sin() + 0.1 * torch.randn(N)

    samples, target = sample_bnn(
        x, y,
        layer_sizes=[1, 50, 1],
        activation="tanh",
        noise_std=0.1,
        sampler="sticky_boomerang",
        n_skeleton=5_000,
        n_resample=2_000,
    )

    # Predictive on a fine grid
    x_grid = torch.linspace(-1.2, 1.2, 200).unsqueeze(-1)
    likelihood = target.meta["model"].likelihood
    with torch.no_grad():
        preds = torch.stack([
            likelihood.predict(b, x_grid).squeeze(-1) for b in samples
        ])
    mean = preds.mean(0)
    std  = preds.std(0)

