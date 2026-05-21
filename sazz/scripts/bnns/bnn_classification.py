"""Minimal example: PDMP sampling for a Bayesian feedforward neural network
applied to multiclass classification.

Given training data (X, y) where y is class labels in {0, ..., K-1}, and a
layer specification, this script builds a fully-connected BNN with Gaussian
priors on the weights and samples from its posterior using one of four PDMP
samplers.

The intended use is as a starting point — copy this file, adjust the data
loading, swap in a CNN if you have image inputs, change the prior, etc.

Usage as a script:
    python bnn_classification.py

Usage as a library:
    from bnn_classification import sample_bnn_classification

    samples, target = sample_bnn_classification(
        X_train, y_train,
        layer_sizes=[X_train.shape[1], 50, n_classes],
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
from sazz.samplers.AutomaticZigZagSampler import AutomaticZigZagSampler
from sazz.samplers.StickyAutomaticBoomerangSampler import (
    StickyAutomaticBoomerangSampler,
)
from sazz.samplers.StickyAutomaticZigZagSampler import (
    StickyAutomaticZigZagSampler,
)
from sazz.models.likelihoods import ModuleCategoricalLikelihood
from sazz.models.priors import ModuleGaussianPrior
from sazz.models.model import TorchTarget, BayesianModel
from sazz.models.neural_networks import FFN
from sazz.utils.bnn_utils import (
    ParamSpec, build_prior_precision, make_kappa_from_inclusion,
)
from sazz.utils.warmup import find_reference_bnn, tune_refresh_rate, warmup
from sazz.utils.sampling import (
    resample_boomerang_path, resample_boomerang_path_sticky,
    resample_zigzag_path, resample_zigzag_path_sticky,
)


SAMPLER_NAMES = ("zigzag", "sticky_zigzag", "boomerang", "sticky_boomerang")

def build_target(
    X: Tensor, y: Tensor, *,
    layer_sizes: Sequence[int],
    activation: str = "relu",
    prior_std_weight: float = 1.0,
    prior_std_bias: float = 1.0,
    fan_in_scaling: bool = True,
    adam_steps: int = 5_000,
    dtype=torch.float64,
):
    X = X.to(dtype=dtype)
    y = y.long()

    n_classes = int(y.max().item()) + 1
    if layer_sizes[-1] != n_classes:
        raise ValueError(
            f"layer_sizes[-1]={layer_sizes[-1]} must equal n_classes={n_classes} "
            f"(inferred from y.max()+1)."
        )

    module = FFN(layer_sizes, activation=activation).to(dtype=dtype)
    spec = ParamSpec.from_module(module)
    init_beta = torch.cat([p.detach().flatten() for p in module.parameters()])
    

    prec = build_prior_precision(
        spec, prior_std_weight, prior_std_bias, fan_in_scaling, dtype, "cpu",
    )
    prior = ModuleGaussianPrior(prec)
    likelihood = ModuleCategoricalLikelihood(module, spec, X, y)
    model = BayesianModel(prior, likelihood)
    # MAP + diagonal Laplace as the sampler's reference / preconditioner.
    x_ref, Sigma_inv = find_reference_bnn(
        model.energy, spec.D, model=model, dtype=dtype, device="cpu",
        reference="laplace_diag", n_steps=adam_steps, lr=1e-2#, init_beta=init_beta
    )

    return TorchTarget(
        name=f"bnn_cls_{'x'.join(map(str, layer_sizes))}_{activation}",
        D=spec.D,
        grad_target=model.grad_energy,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={"model": model, "spec": spec, "module": module,
              "n_classes": n_classes},
    )


def build_sampler(
    name: str, target, *,
    thinning: str = "pli",
    prior_std_weight: float = 1.0,
    prior_inclusion_weight: float = 0.5,
    fan_in_scaling: bool = True,
    refresh_rate: float = 1.0,
    t_max: float = 0.1,
    gamma: float = 0.01,
):
    common = dict(grad_target=target.grad_target, D=target.D, thinning=thinning)
    spec = target.meta["spec"]

    if name == "zigzag":
        return AutomaticZigZagSampler(**common, t_max=t_max, gamma=gamma)

    if name == "boomerang":
        s = AutomaticBoomerangSampler(**common, refresh_rate=0.1)
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
        info = tune_refresh_rate(s, n_pilot=2000)
        print(f"  tuned refresh_rate: "
              f"{info['lambda_r_old']:.3f} -> {info['lambda_r_new']:.3f}")
        warmup(s, n_rounds=3, n_pilot=2000, target=target)
        return s

    # Sticky variants need a kappa vector and a per-coord can_freeze mask.
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
            kappa=kappa, can_freeze=can_freeze, cold_start_threshold=1e-4
        )
    if name == "sticky_boomerang":
        s = StickyAutomaticBoomerangSampler(
            **common, refresh_rate=refresh_rate,
            kappa=kappa, can_freeze=can_freeze, cold_start_threshold=1e-4
        )
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
        info = tune_refresh_rate(s, n_pilot=2000)
        print(f"  tuned refresh_rate: "
              f"{info['lambda_r_old']:.3f} -> {info['lambda_r_new']:.3f}")
        return s

    raise ValueError(f"Unknown sampler '{name}'. Choose from {SAMPLER_NAMES}.")


def resample_path(name: str, result: dict, x_ref: Tensor, *,
                  n_resample: int = 5_000, burnin_frac: float = 0.2):
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


def sample_bnn_classification(
    X: Tensor, y: Tensor, *,
    layer_sizes: Sequence[int],
    activation: str = "relu",
    sampler: str = "sticky_boomerang",
    n_skeleton: int = 10_000,
    n_resample: int = 5_000,
    burnin_frac: float = 0.2,
    thinning: str = "pli",
    prior_std_weight: float = 1.0,
    prior_std_bias: float = 1.0,
    prior_inclusion_weight: float = 0.5,
    fan_in_scaling: bool = True,
    seed: int = 0,
):
    if sampler not in SAMPLER_NAMES:
        raise ValueError(f"Unknown sampler '{sampler}'. Choose from {SAMPLER_NAMES}.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[1/3] Building target  ({sampler}, layers={list(layer_sizes)})")
    target = build_target(
        X, y, layer_sizes=layer_sizes, activation=activation,
        prior_std_weight=prior_std_weight, prior_std_bias=prior_std_bias,
        fan_in_scaling=fan_in_scaling,
    )
    print(f"      D = {target.D},  n_classes = {target.meta['n_classes']}")

    print(f"[2/3] Sampling {n_skeleton} skeleton events")
    s = build_sampler(
        sampler, target,
        thinning=thinning,
        prior_std_weight=prior_std_weight,
        prior_inclusion_weight=prior_inclusion_weight,
        fan_in_scaling=fan_in_scaling,
    )
    t0 = time.perf_counter()
    result = s.sample(N=n_skeleton, diagnostics=False, x0=target.x_ref)
    print(f"      sampling done in {time.perf_counter() - t0:.1f}s")

    print(f"[3/3] Resampling path to {n_resample} draws (burn-in {burnin_frac:.0%})")
    samples_np = resample_path(
        sampler, result, target.x_ref,
        n_resample=n_resample, burnin_frac=burnin_frac,
    )
    samples = torch.tensor(samples_np)

    return samples, target, result


# ===========================================================================
# Demo: small synthetic classification problem
# ===========================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # Three 2-D Gaussian blobs at the corners of a triangle
    centres = torch.tensor([[ 1.5,  0.0],
                            [-0.75,  1.3],
                            [-0.75, -1.3]])
    n_per_class = 40
    X = torch.cat([centres[k] + 0.4 * torch.randn(n_per_class, 2)
                   for k in range(3)], dim=0)
    y = torch.cat([torch.full((n_per_class,), k) for k in range(3)], dim=0)

    samples, target = sample_bnn_classification(
        X, y,
        layer_sizes=[2, 16, 3],
        activation="tanh",
        sampler="sticky_boomerang",
        n_skeleton=5_000,
        n_resample=2_000,
    )

    # Predictive class probabilities on a coarse grid for a quick sanity check
    grid = torch.linspace(-3, 3, 30)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    X_grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    likelihood = target.meta["model"].likelihood
    with torch.no_grad():
        logits = torch.stack([likelihood.predict(b, X_grid) for b in samples])
    probs = torch.softmax(logits, dim=-1).mean(0)   # [N_grid, 3]
    pred_class = probs.argmax(-1)
    confidence = probs.max(-1).values

    train_acc = (
        torch.softmax(
            torch.stack([likelihood.predict(b, X) for b in samples]),
            dim=-1,
        ).mean(0).argmax(-1) == y
    ).float().mean().item()

    print(f"\nsamples: {tuple(samples.shape)}")
    print(f"posterior predictive entropy on grid: "
          f"min={(-probs * probs.clamp(min=1e-12).log()).sum(-1).min():.3f}, "
          f"max={(-probs * probs.clamp(min=1e-12).log()).sum(-1).max():.3f}")
    print(f"train accuracy: {train_acc:.3f}")
