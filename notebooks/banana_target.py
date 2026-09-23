import os
from pathlib import Path

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

if Path.cwd().name == "notebooks":
    os.chdir("..")

from sazz.models.math_targets import make_gaussian, make_banana, make_gaussian_mixture
from sazz.gpu_friendly.samplers.grid_boomerang import GridBoomerangSampler
from sazz.gpu_friendly.samplers.grid_zigzag import GridZigZagSampler
from sazz.gpu_friendly.utils.resample import resample_zigzag_path_torch, resample_boomerang_path_torch

def run_grid_boomerang(target, N=50_000, refresh_rate=1.0, n_segments=20,
                        grid_t_max_init=math.pi / 4, sigma_inv_scale=1.0,
                        dtype=torch.float64):
    """Build a GridBoomerangSampler for `target` and run it for N skeleton points.

    `sigma_inv_scale` multiplies the target's Sigma_inv. It defaults to 1.0 so
    the reference measure is the one the target defines: the scale belongs in
    ONE place, either here or in make_banana(sigma_inv_scale=...), never both.
    A hardcoded factor here silently multiplied with the target's own scale,
    turning the 1 / 0.1 / 10 sweep below into 0.1 / 0.01 / 1.0.
    """
    sampler = GridBoomerangSampler(
        grad_target=target.grad_target,
        D=target.D,
        refresh_rate=refresh_rate,
        grid_t_max_init=grid_t_max_init,
        n_segments=n_segments,
        dtype=dtype,
    )
    sampler.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv * sigma_inv_scale)
    result = sampler.sample(N=N, diagnostics=True)
    result["sampler"] = sampler
    return result

def run_grid_zigzag(target, N=50_000, n_segments=20,
                        grid_t_max_init=1.0, dtype=torch.float64):
    """Build a GridBoomerangSampler for `target` and run it for N skeleton points."""
    sampler = GridZigZagSampler(
        grad_target=target.grad_target,
        D=target.D,
        gamma=0.01,
        grid_t_max_init=grid_t_max_init,
        n_segments=n_segments,
        dtype=dtype,
    )
    result = sampler.sample(N=N, x0=target.x_ref, diagnostics=True)
    result["sampler"] = sampler
    return result

def plot_marginals(target, result, model="zigzag", max_coords=4, bins=60, burnin_frac=0.5):
    """Histogram of each coordinate vs the analytic marginal PDF, plus the
    adaptive grid_t_max trajectory over the run."""
    coords = list(target.marginal_grids.keys())[:max_coords]
    # Resamplers require torch.Tensor inputs (they read .dtype/.device off
    # them directly) -- keep these as tensors here, only convert to numpy
    # for matplotlib after resampling.
    positions = result["positions"]
    velocities = result["velocities"]
    times = result["times"]
    n = positions.shape[0]
    if model == "zigzag":
        samples = resample_zigzag_path_torch(positions, velocities, times, N_resample=10_000, burnin_frac=burnin_frac)
    elif model == "boomerang":
        samples = resample_boomerang_path_torch(positions, velocities, times, target.x_ref, N_resample=10_000, burnin_frac=burnin_frac)
    else:
        raise ValueError(f"Please provide a valid model ('zigzag' or 'boomerang'), got {model!r}")
    samples = samples.numpy()

    fig, axes = plt.subplots(1, len(coords) + 1, figsize=(3.2 * (len(coords) + 1), 3))

    for i, c in enumerate(coords):
        ax = axes[i]
        info = target.marginal_grids[c]
        ax.hist(samples[:, c], bins=bins, density=True, alpha=0.5, label="grid boomerang")
        ax.plot(info["grid"], info["pdf"], "k--", lw=1.5, label="analytic")
        ax.set_title(info["label"])
        if i == 0:
            ax.legend(fontsize=8)

    ax = axes[-1]
    ax.plot(result["grid_t_max_log"])
    ax.set_title("adaptive grid_t_max")
    ax.set_xlabel("iteration")

    fig.tight_layout()
    plt.show()

    print(f"bound_violations: {result['bound_violations']}")
    print(f"gradient_evals: {result['gradient_evals']} "
          f"({result['gradient_evals'] / n:.1f} / skeleton point)")
    
    
    
def plot_marginals_joint(target, result_list, max_coords=4, bins=60, burnin_frac=0.5):
    """Histogram of each coordinate vs the analytic marginal PDF, with
    zigzag and boomerang samples overlaid on the same axes."""
    coords = list(target.marginal_grids.keys())[:max_coords]

    fig, axes = plt.subplots(1, len(coords), figsize=(3.2 * len(coords), 3))
    if len(coords) == 1:
        axes = [axes]

    for result in result_list:
        # Resamplers require torch.Tensor inputs (they read .dtype/.device
        # off them directly) -- keep these as tensors here, only convert to
        # numpy for matplotlib after resampling.
        positions = result["positions"]
        velocities = result["velocities"]
        times = result["times"]

        if isinstance(result["sampler"], GridZigZagSampler):
            samples = resample_zigzag_path_torch(positions, velocities, times, N_resample=10_000, burnin_frac=burnin_frac)
            label = "Zigzag"
        elif isinstance(result["sampler"], GridBoomerangSampler):
            samples = resample_boomerang_path_torch(positions, velocities, times, target.x_ref, N_resample=10_000, burnin_frac=burnin_frac)
            label = "Boomerang"
        else:
            raise ValueError(f"Unrecognized sampler type: {type(result['sampler'])!r}")
        samples = samples.numpy()

        for i, c in enumerate(coords):
            axes[i].hist(samples[:, c], bins=bins, density=True, alpha=0.5, label=label)

        n = positions.shape[0]
        print(f"[{label}] bound_violations: {result['bound_violations']}")
        print(f"[{label}] gradient_evals: {result['gradient_evals']} "
              f"({result['gradient_evals'] / n:.1f} / skeleton point)")

    for i, c in enumerate(coords):
        info = target.marginal_grids[c]
        axes[i].plot(info["grid"], info["pdf"], "k--", lw=1.5, label="Analytic")
        axes[i].set_title(info["label"])
        if i == 0:
            axes[i].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fname=f"results/plots/math_targets/{target.name}.png")
    plt.show()


target_banana_original = make_banana(a=1.0, scale=2.0)

target_banana_shrink = make_banana(a=1.0, scale=2.0, sigma_inv_scale=0.1)

target_banana_bloat = make_banana(a=1.0, scale=2.0, sigma_inv_scale=10.0)

print(f"Target: {target_banana_original.Sigma_inv}  D={target_banana_original.D}")
result_banana_zigzag = run_grid_zigzag(target_banana_original, N=50_000)
result_banana_boom_original = run_grid_boomerang(target_banana_original, N=50_000, refresh_rate=1.0)
result_banana_boom_original_slow = run_grid_boomerang(target_banana_original, N=50_000, refresh_rate=0.1)
result_banana_boom_original_fast = run_grid_boomerang(target_banana_original, N=50_000, refresh_rate=10.0)

print(f"Target: {target_banana_shrink.Sigma_inv}  D={target_banana_shrink.D}")
result_banana_zigzag_shrink = run_grid_zigzag(target_banana_shrink, N=50_000)

result_banana_boom_shrink = run_grid_boomerang(target_banana_shrink, N=50_000, refresh_rate=1.0)
result_banana_boom_shrink_slow = run_grid_boomerang(target_banana_shrink, N=50_000, refresh_rate=0.1)
result_banana_boom_shrink_fast = run_grid_boomerang(target_banana_shrink, N=50_000, refresh_rate=10.0)

print(f"Target: {target_banana_bloat.Sigma_inv}  D={target_banana_bloat.D}")
result_banana_zigzag_bloat = run_grid_zigzag(target_banana_bloat, N=50_000)
result_banana_boom_bloat = run_grid_boomerang(target_banana_bloat, N=50_000, refresh_rate=1.0)
result_banana_boom_bloat_slow = run_grid_boomerang(target_banana_bloat, N=50_000, refresh_rate=0.1)
result_banana_boom_bloat_fast = run_grid_boomerang(target_banana_bloat, N=50_000, refresh_rate=10.0)