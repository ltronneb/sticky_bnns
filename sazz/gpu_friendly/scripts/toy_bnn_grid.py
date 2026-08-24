"""
Grid-bound Boomerang/ZigZag (Andral & Kamatani 2024) on the same toy 1-D BNN
regression benchmarks as sazz/scripts/bnns/toy_bnn.py -- same FFN
architecture, datasets, and per-dataset hyperparameters. Six samplers
available: "grid_zigzag" (GridZigZagSampler), "grid_sticky_zigzag"
(GridStickyZigZagSampler, sparsity via freeze/thaw), "grid_boomerang"
(GridBoomerangSampler), "grid_sticky_boomerang" (GridStickyBoomerangSampler),
"nuts" (NumPyro NUTS), and "svi" (NumPyro mean-field SVI with
TraceMeanField_ELBO, on the same model as "nuts").

Isolated from sazz/scripts/bnns/toy_bnn.py's model/sampler/warmup imports;
only reads pure utilities from the existing tree (generate_toys.GENERATORS,
sampling.resample_boomerang_path[_sticky]). Model/likelihood/prior wiring,
the reference-measure finder, and the grid samplers are gpu_friendly-tree-native.
Usage:
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid --datasets hernandez gap
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid --samplers nuts
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid --samplers grid_sticky_boomerang
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid --samplers grid_sticky_zigzag
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid --samplers grid_boomerang grid_sticky_boomerang nuts
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from sazz.scripts.bnns.generate_toys import GENERATORS, DEFAULT_SEED

from sazz.gpu_friendly.models.neural_networks import FFN
from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.priors import (
    build_fan_in_prior_precision, build_kappa_from_inclusion, build_can_freeze_mask,
)
from sazz.gpu_friendly.utils.warmup import find_reference_bnn
from sazz.gpu_friendly.utils.resample import (
    resample_zigzag_path_torch, resample_zigzag_path_sticky_torch,
    resample_boomerang_path_torch, resample_boomerang_path_sticky_torch,
)
from sazz.gpu_friendly.samplers.grid_zigzag import GridZigZagSampler
from sazz.gpu_friendly.samplers.grid_sticky_zigzag import GridStickyZigZagSampler
from sazz.gpu_friendly.samplers.grid_boomerang import GridBoomerangSampler
from sazz.gpu_friendly.samplers.grid_sticky_boomerang import GridStickyBoomerangSampler



# ===========================================================================
# Config -- mirrors sazz/scripts/bnns/toy_bnn.py's 
# ===========================================================================

# NOTE: torch.func.jvp(torch.func.grad(...)) -- the autodiff composition
# GridBoomerangSampler/GridStickyBoomerangSampler use for their rate/
# derivative closures (see grid_boomerang.py's _make_rate_and_grad_fn) --
# is currently broken on MPS (confirmed on torch 2.11.0: fails with a
# TypeError from deep inside functorch's dual-tensor machinery, reproduces
# with a minimal jvp(grad(...)) snippet with no project code involved).
# This is an upstream PyTorch/MPS bug, not something fixable here. Model
# construction, warmup/MAP-finding, and resampling all work correctly on
# MPS (verified); only the sampler's rate-closure step is blocked. CUDA is
# untested but should work since jvp(grad(...)) is a standard, well-
# supported composition there. MPS will silently pick "mps" below and then
# fail inside sampler.sample() -- if that happens, fall back to "cpu".
DEVICE = (
    "cuda" if torch.cuda.is_available()
    #else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# grid bound needs float64 for the Adam MAP + Fisher step to be numerically
# comparable to the Brent/PLI runs, which also default to float64 in
# AutomaticBoomerangSampler -- but MPS has no float64 support at all
# (torch.Tensor.to(dtype=torch.float64, device="mps") raises), so MPS runs
# fall back to float32 throughout. This is a real precision tradeoff, not
# a workaround: MAP/Fisher accuracy is somewhat reduced on MPS relative to
# CUDA/CPU float64 runs.
DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64
torch.set_default_dtype(torch.float32)  # matches old script's global default;
                                          # DTYPE above is what we actually pass

N_SKELETON  = 50_000
N_RESAMPLE  = 10_000
BURNIN_FRAC = 0.2
BASE_SEED   = 42

SIGMA_INV_SCALE = 0.1 #1.0 10.0
REFRESH_RATE = 1.0
GAMMA = 0.01

GRID_N_SEGMENTS = 60
GRID_T_MAX_INIT_ZIGZAG = 0.002
GRID_ALPHA_PLUS = 1.01
GRID_ALPHA_MINUS = 1.04
GRID_SPACING_ZIGZAG = 0.0002
GRID_T_MAX_INIT_BOOM = 2e-2
GRID_SPACING_BOOM = 3e-3
GRID_STICKY_BOOM_SPACING = GRID_SPACING_BOOM
GRID_STICKY_ZIGZAG_SPACING = GRID_SPACING_ZIGZAG

GRID_STICKY_COLD_START_THRESHOLD = None 

N_SAVE = 8_000  # matches the existing results/toy_bnns/*/split_00/*.pt files

NUTS_DRAWS  = 2_000
NUTS_WARMUP = 1_000
NUTS_CHAINS = 4
PRIOR_INCLUSION_WEIGHT = 0.1

OUT_DIR = Path("results/grid/toy_bnns")
TOY_DIR = Path("datasets/toy_1d")

TOY_DATASETS = ("hernandez", "gap", "sharp", "multiscale")
SAMPLER_NAMES = ("grid_zigzag", "grid_sticky_zigzag", "grid_boomerang", "grid_sticky_boomerang", "nuts", "svi")

DATASET_CONFIGS = {
    "hernandez": dict(
        layer_sizes=[1, 100, 1], activation="tanh",
        prior_std_weight=3.0, prior_std_bias=3.0,
        fan_in_scaling=True, adam_steps=10000,
        prior_inclusion_weight=PRIOR_INCLUSION_WEIGHT,
    ),
    "gap": dict(
        layer_sizes=[1, 100, 1], activation="tanh",
        prior_std_weight=3.0, prior_std_bias=3.0,
        fan_in_scaling=True, adam_steps=10000,
        prior_inclusion_weight=PRIOR_INCLUSION_WEIGHT,
    ),
    "sharp": dict(
        layer_sizes=[1, 100, 1], activation="tanh",
        prior_std_weight=3.0, prior_std_bias=3.0,
        fan_in_scaling=True, adam_steps=10000,
        prior_inclusion_weight=PRIOR_INCLUSION_WEIGHT,
    ),
    "multiscale": dict(
        layer_sizes=[1, 100, 1], activation="tanh",
        prior_std_weight=3.0, prior_std_bias=3.0,
        fan_in_scaling=True, adam_steps=10000,
        prior_inclusion_weight=PRIOR_INCLUSION_WEIGHT,
    ),
}


@dataclass
class BNNConfig:
    layer_sizes: list[int]
    activation: str = "tanh"
    noise_std: float = 0.3
    prior_std_weight: float = 3.0
    prior_std_bias: float = 3.0
    fan_in_scaling: bool = True
    adam_steps: int = 10000
    prior_inclusion_weight: float = PRIOR_INCLUSION_WEIGHT


# ===========================================================================
# Data loading -- identical generation path to toy_bnn.py (same GENERATORS,
# same DEFAULT_SEED), so results are directly comparable dataset-for-dataset.
# ===========================================================================

def load_toy(name: str, toy_dir: Path) -> tuple[dict[str, Any], BNNConfig]:
    path = toy_dir / f"{name}.pt"
    toy_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(DEFAULT_SEED)
    data = GENERATORS[name](rng)
    torch.save(data, path)
    cfg = BNNConfig(**DATASET_CONFIGS[name], noise_std=data["noise_std"])
    return data, cfg


# ===========================================================================
# Target builder
# ===========================================================================

def build_target(data: dict[str, Any], cfg: BNNConfig, dtype=DTYPE, device=DEVICE):
    X = data["X_train"].to(dtype=dtype, device=device)
    y = data["y_train"].to(dtype=dtype, device=device)

    module = FFN(cfg.layer_sizes, cfg.activation)
    prec = build_fan_in_prior_precision(
        module, cfg.prior_std_weight, cfg.prior_std_bias,
        cfg.fan_in_scaling, dtype=dtype, device=device,
    )

    # Fixed-noise override (BayesianModule.build defaults to learning it) --
    # the toy datasets already know their true standardized noise level.
    bm = BayesianModule.build(
        module, likelihood="gaussian", X=X, y=y,
        noise_std=cfg.noise_std, prior_precision=prec, dtype=dtype, device=device,
    )

    x_ref, Sigma_inv = find_reference_bnn(
        bm, n_steps=cfg.adam_steps, lr=1e-2, dtype=dtype, device=torch.device(device),
    )

    return bm, x_ref, Sigma_inv


# ===========================================================================
# Sampler
# ===========================================================================

def build_zigzag_sampler(bm: BayesianModule):
    sampler = GridZigZagSampler(
        grad_target=torch.func.grad(bm.energy),
        D=bm.D,
        gamma=GAMMA,
        grid_t_max_init=GRID_T_MAX_INIT_ZIGZAG,
        n_segments=GRID_N_SEGMENTS,
        grid_spacing=GRID_SPACING_ZIGZAG,
        alpha_plus=GRID_ALPHA_PLUS,
        alpha_minus=GRID_ALPHA_MINUS,
        dtype=DTYPE,
        device=bm.device,
    )
    return sampler


def build_sticky_zigzag_sampler(bm: BayesianModule, cfg: BNNConfig):
    """
    kappa/can_freeze constructed identically to build_sticky_boomerang_sampler
    (same bm.module.named_parameters()-based helpers in gpu_friendly/models/
    priors.py) -- ZigZag has no x_ref/Sigma_inv dependency, so this builder
    only needs bm/cfg, unlike its Boomerang counterpart.
    """
    kappa_net = build_kappa_from_inclusion(
        bm.module, cfg.prior_std_weight, cfg.prior_inclusion_weight,
        cfg.fan_in_scaling, dtype=DTYPE, device=bm.device,
    )
    can_freeze_net = build_can_freeze_mask(bm.module, device=bm.device)

    if bm.learns_noise:
        kappa = torch.cat([kappa_net, torch.zeros(1, dtype=DTYPE, device=bm.device)])
        can_freeze = torch.cat([can_freeze_net, torch.zeros(1, dtype=torch.bool, device=bm.device)])
    else:
        kappa = kappa_net
        can_freeze = can_freeze_net

    sampler = GridStickyZigZagSampler(
        grad_target=torch.func.grad(bm.energy),
        D=bm.D,
        kappa=kappa,
        can_freeze=can_freeze,
        cold_start_threshold=GRID_STICKY_COLD_START_THRESHOLD,
        gamma=GAMMA,
        grid_t_max_init=GRID_T_MAX_INIT_ZIGZAG,
        n_segments=GRID_N_SEGMENTS,
        grid_spacing=GRID_STICKY_ZIGZAG_SPACING,
        alpha_plus=GRID_ALPHA_PLUS,
        alpha_minus=GRID_ALPHA_MINUS,
        dtype=DTYPE,
        device=bm.device,
    )
    return sampler


def build_boomerang_sampler(bm: BayesianModule, x_ref: torch.Tensor, Sigma_inv: torch.Tensor):
    sampler = GridBoomerangSampler(
        grad_target=torch.func.grad(bm.energy),
        D=bm.D,
        refresh_rate=REFRESH_RATE,
        grid_t_max_init=GRID_T_MAX_INIT_BOOM,
        n_segments=GRID_N_SEGMENTS,
        grid_spacing=GRID_SPACING_BOOM,
        alpha_plus=GRID_ALPHA_PLUS,
        alpha_minus=GRID_ALPHA_MINUS,
        dtype=DTYPE,
        device=bm.device,
    )
    sampler.preprocess(x_ref=x_ref, Sigma_inv=Sigma_inv * SIGMA_INV_SCALE)
    return sampler


def build_sticky_boomerang_sampler(bm: BayesianModule, cfg: BNNConfig,
                               x_ref: torch.Tensor, Sigma_inv: torch.Tensor):
    """
    kappa/can_freeze are derived from the network's own module (weight vs.
    bias, fan-in), matching sazz.utils.bnn_utils.make_kappa_from_inclusion's
    math but reimplemented against bm.module.named_parameters() directly
    (see gpu_friendly/models/priors.py) -- no ParamSpec dependency.
    If bm.learns_noise, the appended log_sigma coordinate is never a
    network parameter, so it's excluded from kappa/can_freeze construction
    and explicitly padded as non-freezable (can_freeze=False, kappa
    irrelevant since it can never freeze) -- the toy datasets currently
    always use the fixed-noise path (see build_target), so this padding is
    presently inert, but keeps this builder correct if that ever changes.
    """
    kappa_net = build_kappa_from_inclusion(
        bm.module, cfg.prior_std_weight, cfg.prior_inclusion_weight,
        cfg.fan_in_scaling, dtype=DTYPE, device=bm.device,
    )
    can_freeze_net = build_can_freeze_mask(bm.module, device=bm.device)

    if bm.learns_noise:
        kappa = torch.cat([kappa_net, torch.zeros(1, dtype=DTYPE, device=bm.device)])
        can_freeze = torch.cat([can_freeze_net, torch.zeros(1, dtype=torch.bool, device=bm.device)])
    else:
        kappa = kappa_net
        can_freeze = can_freeze_net

    sampler = GridStickyBoomerangSampler(
        grad_target=torch.func.grad(bm.energy),
        D=bm.D,
        kappa=kappa,
        can_freeze=can_freeze,
        cold_start_threshold=GRID_STICKY_COLD_START_THRESHOLD,
        grid_spacing=GRID_STICKY_BOOM_SPACING,
        refresh_rate=REFRESH_RATE,
        grid_t_max_init=GRID_T_MAX_INIT_BOOM,
        n_segments=GRID_N_SEGMENTS,
        alpha_plus=GRID_ALPHA_PLUS,
        alpha_minus=GRID_ALPHA_MINUS,
        dtype=DTYPE,
        device=bm.device,
    )
    sampler.preprocess(x_ref=x_ref, Sigma_inv=Sigma_inv * SIGMA_INV_SCALE)
    return sampler



# ===========================================================================
# NUTS via NumPyro
# ===========================================================================

def run_nuts(data: dict[str, Any], cfg: BNNConfig, seed: int) -> tuple[np.ndarray, float, int]:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    layer_sizes, activation = cfg.layer_sizes, cfg.activation
    prior_std, prior_std_bias = cfg.prior_std_weight, cfg.prior_std_bias

    X = jnp.asarray(data["X_train"].cpu().numpy())
    y = jnp.asarray(data["y_train"].cpu().numpy())

    def bnn(X, y=None):
        h = X
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            scale = prior_std / jnp.sqrt(n_in if cfg.fan_in_scaling else 1)
            W = numpyro.sample(
                f"W{i}", dist.Normal(jnp.zeros((n_out, n_in)), scale).to_event(2),
            )
            b = numpyro.sample(
                f"b{i}", dist.Normal(jnp.zeros(n_out), prior_std_bias).to_event(1),
            )
            h = h @ W.T + b
            if i < len(layer_sizes) - 2:
                if activation == "tanh":
                    h = jnp.tanh(h)
                elif activation == "relu":
                    h = jnp.maximum(0.0, h)
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("y", dist.Normal(h.squeeze(-1), cfg.noise_std), obs=y)

    kernel = NUTS(bnn, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=NUTS_WARMUP, num_samples=NUTS_DRAWS,
                num_chains=NUTS_CHAINS, progress_bar=True)

    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), X=X, y=y, extra_fields=("num_steps",))
    elapsed = time.perf_counter() - t0

    # num_steps = leapfrog steps per post-warmup sample; summing across
    # samples/chains gives total gradient evaluations, the same currency
    # as the PDMP samplers' gradient_evals (see grid_boomerang.py).
    gradient_evals = int(np.asarray(mcmc.get_extra_fields()["num_steps"]).sum())

    posterior = mcmc.get_samples()
    flat = []
    for i in range(len(layer_sizes) - 1):
        W = np.asarray(posterior[f"W{i}"])  # [n, n_out, n_in]
        b = np.asarray(posterior[f"b{i}"])  # [n, n_out]
        flat.append(W.reshape(W.shape[0], -1))
        flat.append(b)
    return np.concatenate(flat, axis=1).astype(np.float64), elapsed, gradient_evals

SVI_STEPS   = 30_000
SVI_LR      = 1e-2
SVI_DRAWS   = NUTS_DRAWS * NUTS_CHAINS  # match NUTS's total posterior-draw count
SVI_RANK    = 20  # low-rank covariance factor count; D~150-300 here, so this is a strong compression


def run_svi(data: dict[str, Any], cfg: BNNConfig, seed: int) -> tuple[np.ndarray, float, int]:
    """
    SVI on the same `bnn` model as run_nuts, using AutoLowRankMultivariateNormal
    (mean + diagonal + low-rank factor covariance) as the guide -- unlike a
    mean-field guide, this lets the approximate posterior capture weight-weight
    correlation, closer in spirit to the Boomerang samplers' Gaussian reference
    measure (x_ref/Sigma_inv) than a diagonal guide would be, at O(D*rank) cost
    instead of full AutoMultivariateNormal's O(D^2).

    Uses Trace_ELBO, not TraceMeanField_ELBO: internally this guide reparameterizes
    per-site samples as Delta distributions hung off one joint "_auto_latent"
    LowRankMultivariateNormal site, so TraceMeanField_ELBO's analytic-KL fast path
    (which needs a registered analytic KL(guide_site, model_site) per site, e.g.
    Normal-vs-Normal) never applies here -- it silently falls back to the same
    single-sample surrogate Trace_ELBO already computes, just with extra
    bookkeeping overhead for no variance-reduction benefit.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
    from numpyro.optim import Adam

    layer_sizes, activation = cfg.layer_sizes, cfg.activation
    prior_std, prior_std_bias = cfg.prior_std_weight, cfg.prior_std_bias

    X = jnp.asarray(data["X_train"].cpu().numpy())
    y = jnp.asarray(data["y_train"].cpu().numpy())

    def bnn(X, y=None):
        h = X
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            scale = prior_std / jnp.sqrt(n_in if cfg.fan_in_scaling else 1)
            W = numpyro.sample(
                f"W{i}", dist.Normal(jnp.zeros((n_out, n_in)), scale).to_event(2),
            )
            b = numpyro.sample(
                f"b{i}", dist.Normal(jnp.zeros(n_out), prior_std_bias).to_event(1),
            )
            h = h @ W.T + b
            if i < len(layer_sizes) - 2:
                if activation == "tanh":
                    h = jnp.tanh(h)
                elif activation == "relu":
                    h = jnp.maximum(0.0, h)
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("y", dist.Normal(h.squeeze(-1), cfg.noise_std), obs=y)

    guide = AutoLowRankMultivariateNormal(bnn, rank=SVI_RANK)
    svi = SVI(bnn, guide, Adam(SVI_LR), Trace_ELBO())

    rng_key, sample_key = jax.random.split(jax.random.PRNGKey(seed))

    t0 = time.perf_counter()
    svi_result = svi.run(rng_key, SVI_STEPS, X=X, y=y, progress_bar=True)
    elapsed = time.perf_counter() - t0

    # One gradient evaluation per optimization step -- the SVI analogue
    # of NUTS's leapfrog-step count, same currency as the PDMP samplers'
    # gradient_evals (see grid_boomerang.py).
    gradient_evals = SVI_STEPS

    posterior = guide.sample_posterior(
        sample_key, svi_result.params, sample_shape=(SVI_DRAWS,),
    )
    flat = []
    for i in range(len(layer_sizes) - 1):
        W = np.asarray(posterior[f"W{i}"])  # [n, n_out, n_in]
        b = np.asarray(posterior[f"b{i}"])  # [n, n_out]
        flat.append(W.reshape(W.shape[0], -1))
        flat.append(b)
    return np.concatenate(flat, axis=1).astype(np.float64), elapsed, gradient_evals


# ===========================================================================
# Persistence -- same payload schema as toy_bnn.py's save_run, so results
# under results/grid/ can be loaded with the same downstream analysis code.
# ===========================================================================

def thin_to(samples: torch.Tensor, n_keep: int) -> torch.Tensor:
    n = samples.shape[0]
    if n <= n_keep:
        return samples
    idx = torch.linspace(0, n - 1, n_keep, device=samples.device).round().long()
    return samples[idx]


def split_dir(out_dir: Path, dataset: str, split_id: int) -> Path:
    return out_dir / dataset / f"split_{split_id:02d}"


def save_run(out_path: Path, *, dataset: str, split_id: int, sampler: str,
             samples: torch.Tensor, x_ref: Optional[torch.Tensor], cfg: BNNConfig,
             y_std: float, elapsed_sec: float, n_events: int,
             bound_violations: int, gradient_evals: Optional[int] = None,
             grid_t_max_log: Optional[list[float]] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "dataset":          dataset,
        "split_id":         split_id,
        "sampler":          sampler,
        "samples":          thin_to(samples, N_SAVE).cpu(),
        "x_ref":            x_ref.cpu() if x_ref is not None else None,
        "layer_sizes":      cfg.layer_sizes,
        "activation":       cfg.activation,
        "noise_std":        cfg.noise_std,
        "y_std":            y_std,
        "n_events":         n_events,
        "elapsed_sec":      elapsed_sec,
        "bound_violations": bound_violations,
        "gradient_evals":   gradient_evals,
        "grid_t_max_log":   grid_t_max_log,
    }, out_path)


# ===========================================================================
# Per-dataset run -- each runner below does its own BayesianModule build
# (MAP + Laplace), independently seeded; nuts only needs cfg/data/seed.
# ===========================================================================

def run_grid_zigzag(dataset_name: str, split_id: int, data: dict[str, Any],
                        cfg: BNNConfig, sd: Path) -> None:
    seed = BASE_SEED + split_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, x_ref, _ = build_target(data, cfg)
    print(f"  D = {bm.D}")
    

    sampler = build_zigzag_sampler(bm)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, x0=x_ref, diagnostics=True)
    elapsed = time.perf_counter() - t0

    samples = resample_zigzag_path_torch(
        result["positions"], result["velocities"], result["times"],
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations)")

    out_path = sd / "grid_zigzag.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_zigzag",
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=N_SKELETON,
        bound_violations=result["bound_violations"],
        gradient_evals=result["gradient_evals"],
        grid_t_max_log=result["grid_t_max_log"],
    )
    print(f"      saved {samples.shape[0]} samples (thinned to {N_SAVE}) -> {out_path}")


def run_grid_sticky_zigzag(dataset_name: str, split_id: int, data: dict[str, Any],
                            cfg: BNNConfig, sd: Path) -> None:
    seed = BASE_SEED + split_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, x_ref, _ = build_target(data, cfg)
    print(f"  D = {bm.D}")
    print(f"  Sticky inclusion prob = {PRIOR_INCLUSION_WEIGHT}")

    sampler = build_sticky_zigzag_sampler(bm, cfg)

    t0 = time.perf_counter()
    # x0=x_ref: same warm-start convention as plain grid_zigzag -- ZigZag has
    # no reference measure, so x_ref is used purely as an initial position
    # here (confirmed compatible when GridZigZagSampler was first built).
    result = sampler.sample(N=N_SKELETON, x0=x_ref, diagnostics=True)
    elapsed = time.perf_counter() - t0

    samples = resample_zigzag_path_sticky_torch(
        result["positions"], result["velocities"], result["times"],
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, "
          f"final sparsity {sparsity:.2f})")

    out_path = sd / "grid_sticky_zigzag.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_sticky_zigzag",
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=N_SKELETON,
        bound_violations=result["bound_violations"],
        gradient_evals=result["gradient_evals"],
        grid_t_max_log=result["grid_t_max_log"],
    )
    print(f"      saved {samples.shape[0]} samples (thinned to {N_SAVE}) -> {out_path}")


def run_grid_boomerang(dataset_name: str, split_id: int, data: dict[str, Any],
                        cfg: BNNConfig, sd: Path) -> None:
    seed = BASE_SEED + split_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, x_ref, Sigma_inv = build_target(data, cfg)
    print(f"  D = {bm.D}")
    print(f"  Sticky inclusion prob = {PRIOR_INCLUSION_WEIGHT}")

    sampler = build_boomerang_sampler(bm, x_ref, Sigma_inv)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, diagnostics=True)
    elapsed = time.perf_counter() - t0

    samples = resample_boomerang_path_torch(
        result["positions"], result["velocities"], result["times"], x_ref,
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations)")

    out_path = sd / "grid_boomerang.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_boomerang",
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=N_SKELETON,
        bound_violations=result["bound_violations"],
        gradient_evals=result["gradient_evals"],
        grid_t_max_log=result["grid_t_max_log"],
    )
    print(f"      saved {samples.shape[0]} samples (thinned to {N_SAVE}) -> {out_path}")


def run_grid_sticky_boomerang(dataset_name: str, split_id: int, data: dict[str, Any],
                               cfg: BNNConfig, sd: Path) -> None:
    seed = BASE_SEED + split_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, x_ref, Sigma_inv = build_target(data, cfg)
    print(f"  D = {bm.D}")

    sampler = build_sticky_boomerang_sampler(bm, cfg, x_ref, Sigma_inv)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, diagnostics=True)
    elapsed = time.perf_counter() - t0

    samples = resample_boomerang_path_sticky_torch(
        result["positions"], result["velocities"], result["times"], x_ref,
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, "
          f"final sparsity {sparsity:.2f})")

    out_path = sd / "grid_sticky_boomerang.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_sticky_boomerang",
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=N_SKELETON,
        bound_violations=result["bound_violations"],
        gradient_evals=result["gradient_evals"],
        grid_t_max_log=result["grid_t_max_log"],
    )
    print(f"      saved {samples.shape[0]} samples (thinned to {N_SAVE}) -> {out_path}")


def run_nuts_dataset(dataset_name: str, split_id: int, data: dict[str, Any],
                      cfg: BNNConfig, sd: Path) -> None:
    seed = BASE_SEED + split_id

    samples_np, elapsed, gradient_evals = run_nuts(data, cfg, seed)
    samples = torch.tensor(samples_np)

    print(f"      sampled {samples.shape[0]} NUTS draws in {elapsed:.1f}s "
          f"({gradient_evals} gradient evals)")

    out_path = sd / "nuts.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="nuts",
        samples=samples, x_ref=None, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=samples.shape[0], bound_violations=0,
        gradient_evals=gradient_evals,
    )
    print(f"      saved {samples.shape[0]} samples -> {out_path}")


def run_svi_dataset(dataset_name: str, split_id: int, data: dict[str, Any],
                     cfg: BNNConfig, sd: Path) -> None:
    seed = BASE_SEED + split_id

    samples_np, elapsed, gradient_evals = run_svi(data, cfg, seed)
    samples = torch.tensor(samples_np)

    print(f"      sampled {samples.shape[0]} SVI draws in {elapsed:.1f}s "
          f"({gradient_evals} gradient evals)")

    out_path = sd / "svi.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="svi",
        samples=samples, x_ref=None, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=samples.shape[0], bound_violations=0,
        gradient_evals=gradient_evals,
    )
    print(f"      saved {samples.shape[0]} samples -> {out_path}")


SAMPLER_RUNNERS = {
    "grid_zigzag": run_grid_zigzag,
    "grid_sticky_zigzag": run_grid_sticky_zigzag,
    "grid_boomerang": run_grid_boomerang,
    "grid_sticky_boomerang": run_grid_sticky_boomerang,
    "nuts": run_nuts_dataset,
    "svi": run_svi_dataset,
}


def run_dataset(dataset_name: str, split_id: int, data: dict[str, Any],
                 cfg: BNNConfig, out_dir: Path, samplers: list[str], resume: bool) -> None:
    print(f"\n--- {dataset_name.upper()} | layers={cfg.layer_sizes} | "
          f"act={cfg.activation} | noise_std={cfg.noise_std:.4f} | "
          f"seed={BASE_SEED + split_id} ---")

    sd = split_dir(out_dir, dataset_name, split_id)

    for sampler_name in samplers:
        out_path = sd / f"{sampler_name}.pt"
        if resume and out_path.exists():
            print(f"  [{sampler_name}] skipping — exists at {out_path}")
            continue
        SAMPLER_RUNNERS[sampler_name](dataset_name, split_id, data, cfg, sd)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=list(TOY_DATASETS),
                         choices=list(TOY_DATASETS))
    parser.add_argument("--samplers", nargs="+", default=list(SAMPLER_NAMES),
                         choices=list(SAMPLER_NAMES))
    parser.add_argument("--splits", nargs="+", type=int, default=[0])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--toy-dir", type=Path, default=TOY_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("\nLoading toy 1D datasets...")
    toys = {n: load_toy(n, args.toy_dir) for n in args.datasets}

    print(f"\nRunning {args.datasets} | samplers: {args.samplers} | splits: {args.splits}")
    for ds in args.datasets:
        data, cfg = toys[ds]
        for split_id in args.splits:
            run_dataset(ds, split_id, data, cfg, args.out, args.samplers, resume=args.resume)


if __name__ == "__main__":
    main()
