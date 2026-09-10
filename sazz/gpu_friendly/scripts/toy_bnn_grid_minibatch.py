"""
Minibatch ablation of sazz/gpu_friendly/scripts/toy_bnn_grid.py.

Identical model / datasets / per-dataset hyperparameters / warmup / grid
config as toy_bnn_grid.py -- the ONLY difference is that the four grid
PDMP samplers can be driven by a SUBSAMPLED gradient: with
--grad-batch-size B set, every grid-bound episode draws a fresh random
size-B minibatch from the training set (likelihood rescaled by N_full/B)
instead of using the full training set. This is the same
statistically-uncorrected subsampled-gradient scheme the larger models
(fast_mnist_cnn.py / fast_cifar_resnet.py) already use; the point of this
script is to measure how badly that minibatching breaks the grid upper
bound (bound_violations) on the toy 1-D problems, where a full-batch
reference is cheap to compute alongside it.

Samplers "grid_zigzag" / "grid_sticky_zigzag" / "grid_boomerang" /
"grid_sticky_boomerang" use the Mb* subclasses from
sazz.gpu_friendly.samplers.mb_grid_* (a thin resample_grad_batch hook on
top of the plain Grid* samplers). "nuts" and "svi" are still selectable
via --samplers for a full-batch reference in the same output tree, but
they run full-batch regardless of --grad-batch-size and are NOT in the
default sampler list here.

--grad-batch-size None (default) reproduces toy_bnn_grid.py exactly (the
Mb* subclasses become no-op passthroughs).

Usage:
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid_minibatch --grad-batch-size 16
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid_minibatch --grad-batch-size 8 --datasets hernandez gap
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid_minibatch --grad-batch-size 32 --samplers grid_sticky_boomerang
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid_minibatch --samplers nuts   # full-batch reference
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
    make_gaussian_prior,
)
from sazz.gpu_friendly.utils.warmup import find_reference_bnn
from sazz.gpu_friendly.utils.resample import (
    resample_zigzag_path_torch, resample_zigzag_path_sticky_torch,
    resample_boomerang_path_torch, resample_boomerang_path_sticky_torch,
)
from sazz.gpu_friendly.samplers.mb_grid_zigzag import MbGridZigZagSampler
from sazz.gpu_friendly.samplers.mb_grid_sticky_zigzag import MbGridStickyZigZagSampler
from sazz.gpu_friendly.samplers.mb_grid_boomerang import MbGridBoomerangSampler
from sazz.gpu_friendly.samplers.mb_grid_sticky_boomerang import MbGridStickyBoomerangSampler



# ===========================================================================
# Config -- mirrors sazz/gpu_friendly/scripts/toy_bnn_grid.py exactly
# ===========================================================================

# See toy_bnn_grid.py's identical DEVICE/DTYPE block for the full MPS-vs-CUDA
# rationale (jvp(grad(...)) broken on MPS; float64 unavailable on MPS).
DEVICE = (
    "cuda" if torch.cuda.is_available()
    #else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64
torch.set_default_dtype(torch.float32)

N_SKELETON  = 50_000
N_RESAMPLE  = 10_000
BURNIN_FRAC = 0.2
BASE_SEED   = 42

# If set (via --grad-budget), every sampler stops as soon as this many real
# gradient evaluations have been spent -- see toy_bnn_grid.py's identical
# GRAD_BUDGET for the full rationale.
GRAD_BUDGET: Optional[int] = None

# If set (via --grad-batch-size), every grid-bound episode of the four grid
# PDMP samplers draws a fresh random minibatch of this size from the
# training set (likelihood rescaled by N_full/batch_size) instead of the
# full training set. None (default) => exact full-batch, reproducing
# toy_bnn_grid.py.
GRAD_BATCH_SIZE: Optional[int] = None

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

OUT_DIR = Path("results/paper/toy_bnns_minibatch")
TOY_DIR = Path("datasets/toy_1d")

TOY_DATASETS = ("hernandez", "gap", "sharp", "multiscale")
# nuts/svi are still selectable via --samplers (full-batch reference), but
# not in the default set -- minibatching does not apply to them.
GRID_SAMPLER_NAMES = ("grid_zigzag", "grid_sticky_zigzag", "grid_boomerang", "grid_sticky_boomerang")
SAMPLER_NAMES = GRID_SAMPLER_NAMES + ("nuts", "svi")

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
# Data loading -- identical generation path to toy_bnn_grid.py
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
# Target builder -- identical to toy_bnn_grid.py
# ===========================================================================

def build_target(data: dict[str, Any], cfg: BNNConfig, dtype=DTYPE, device=DEVICE):
    X = data["X_train"].to(dtype=dtype, device=device)
    y = data["y_train"].to(dtype=dtype, device=device)

    module = FFN(cfg.layer_sizes, cfg.activation)
    prec = build_fan_in_prior_precision(
        module, cfg.prior_std_weight, cfg.prior_std_bias,
        cfg.fan_in_scaling, dtype=dtype, device=device,
    )

    bm = BayesianModule.build(
        module, likelihood="gaussian", X=X, y=y,
        noise_std=cfg.noise_std, prior_precision=prec, dtype=dtype, device=device,
    )

    x_ref, Sigma_inv = find_reference_bnn(
        bm, n_steps=cfg.adam_steps, lr=1e-2, dtype=dtype, device=torch.device(device),
    )

    return bm, x_ref, Sigma_inv


# ===========================================================================
# Minibatch gradient -- same construction as
# fast_mnist_cnn.py::build_minibatch_grad_target, specialised to the toy
# fixed-noise Gaussian likelihood.
# ===========================================================================

def build_minibatch_grad_target(bm: BayesianModule, batch_size: int):
    """
    Returns (grad_target, resample_fn).

    grad_target(x) differentiates the energy against whatever minibatch is
    CURRENTLY cached (never a fresh draw per call) -- it must stay a fixed
    function of x for one full _grid_bound episode, since the grid upper
    bound is only valid if every eager and vmapped rate evaluation within
    that episode sees the same rate function. resample_fn() -- a plain,
    non-vmapped call -- draws a new minibatch and overwrites the cache in
    place; the Mb* samplers call it once per sample() loop iteration,
    immediately before _grid_bound (one fresh minibatch per Poisson-
    thinning proposal).

    Likelihood term rescaled by N_full/batch_size (log_likelihood.single
    SUMS log-probs over whatever batch it is given) so the energy's overall
    scale stays comparable to the full-batch target this script's GAMMA /
    grid spacing were tuned against. This is a raw subsampled-gradient PDMP
    with no exactness correction -- traded for throughput, not exact.

    The toy datasets always use fixed noise (build_target passes noise_std),
    so bm.learns_noise is False and there is no log_sigma coordinate to
    special-case here -- unlike fast_mnist_cnn's helper, which asserts the
    same but for a different reason (its rescale would double-count a
    learned-sigma prior).
    """
    assert not bm.learns_noise, (
        "minibatch likelihood rescale assumes fixed noise (no log_sigma coord)"
    )
    N_full = bm.X.shape[0]
    scale = N_full / batch_size
    log_prior_fn = make_gaussian_prior(bm.prior_precision)

    idx0 = torch.randint(0, N_full, (batch_size,), device=bm.device)
    cache = {"X": bm.X[idx0], "y": bm.y[idx0]}

    def energy_minibatch(beta: torch.Tensor) -> torch.Tensor:
        log_lik_batch = bm.log_likelihood.single(beta, cache["X"], cache["y"]) * scale
        return -(log_prior_fn(beta) + log_lik_batch)

    grad_target = torch.func.grad(energy_minibatch)

    def resample_fn() -> None:
        idx = torch.randint(0, N_full, (batch_size,), device=bm.device)
        cache["X"] = bm.X[idx]
        cache["y"] = bm.y[idx]

    return grad_target, resample_fn


def _grad_target_and_resample(bm: BayesianModule):
    """Full-batch (GRAD_BATCH_SIZE None) or minibatch, chosen by the global."""
    if GRAD_BATCH_SIZE is not None:
        return build_minibatch_grad_target(bm, GRAD_BATCH_SIZE)
    return torch.func.grad(bm.energy), None


# ===========================================================================
# Sampler builders -- same config as toy_bnn_grid.py, Mb* classes with the
# resample_grad_batch hook wired from _grad_target_and_resample.
# ===========================================================================

def build_zigzag_sampler(bm: BayesianModule):
    grad_target, resample_grad_batch = _grad_target_and_resample(bm)
    sampler = MbGridZigZagSampler(
        grad_target=grad_target,
        D=bm.D,
        gamma=GAMMA,
        grid_t_max_init=GRID_T_MAX_INIT_ZIGZAG,
        n_segments=GRID_N_SEGMENTS,
        grid_spacing=GRID_SPACING_ZIGZAG,
        alpha_plus=GRID_ALPHA_PLUS,
        alpha_minus=GRID_ALPHA_MINUS,
        dtype=DTYPE,
        device=bm.device,
        resample_grad_batch=resample_grad_batch,
    )
    return sampler


def build_sticky_zigzag_sampler(bm: BayesianModule, cfg: BNNConfig):
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

    grad_target, resample_grad_batch = _grad_target_and_resample(bm)
    sampler = MbGridStickyZigZagSampler(
        grad_target=grad_target,
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
        resample_grad_batch=resample_grad_batch,
    )
    return sampler


def build_boomerang_sampler(bm: BayesianModule, x_ref: torch.Tensor, Sigma_inv: torch.Tensor):
    grad_target, resample_grad_batch = _grad_target_and_resample(bm)
    sampler = MbGridBoomerangSampler(
        grad_target=grad_target,
        D=bm.D,
        refresh_rate=REFRESH_RATE,
        grid_t_max_init=GRID_T_MAX_INIT_BOOM,
        n_segments=GRID_N_SEGMENTS,
        grid_spacing=GRID_SPACING_BOOM,
        alpha_plus=GRID_ALPHA_PLUS,
        alpha_minus=GRID_ALPHA_MINUS,
        dtype=DTYPE,
        device=bm.device,
        resample_grad_batch=resample_grad_batch,
    )
    sampler.preprocess(x_ref=x_ref, Sigma_inv=Sigma_inv * SIGMA_INV_SCALE)
    return sampler


def build_sticky_boomerang_sampler(bm: BayesianModule, cfg: BNNConfig,
                               x_ref: torch.Tensor, Sigma_inv: torch.Tensor):
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

    grad_target, resample_grad_batch = _grad_target_and_resample(bm)
    sampler = MbGridStickyBoomerangSampler(
        grad_target=grad_target,
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
        resample_grad_batch=resample_grad_batch,
    )
    sampler.preprocess(x_ref=x_ref, Sigma_inv=Sigma_inv * SIGMA_INV_SCALE)
    return sampler



# ===========================================================================
# NUTS via NumPyro -- full-batch reference, unchanged from toy_bnn_grid.py.
# Not affected by --grad-batch-size.
# ===========================================================================

def _truncate_nuts_to_budget(sample_steps: np.ndarray, num_chains: int, num_samples: int,
                              grad_budget: int) -> int:
    """See toy_bnn_grid.py's identical helper for the full rationale."""
    per_chain = sample_steps.reshape(num_chains, num_samples)
    cum_per_chain = np.cumsum(per_chain, axis=1)
    total_cum = cum_per_chain.sum(axis=0)
    fits = np.nonzero(total_cum <= grad_budget)[0]
    return int(fits[-1]) + 1 if len(fits) > 0 else 0


def run_nuts(data: dict[str, Any], cfg: BNNConfig, seed: int,
             grad_budget: Optional[int] = None) -> tuple[np.ndarray, float, int]:
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
    mcmc.warmup(jax.random.PRNGKey(seed), X=X, y=y,
                extra_fields=("num_steps",), collect_warmup=True)
    warmup_steps = int(np.asarray(mcmc.get_extra_fields()["num_steps"]).sum())
    mcmc.run(mcmc.post_warmup_state.rng_key, X=X, y=y, extra_fields=("num_steps",))
    elapsed = time.perf_counter() - t0

    sample_steps_arr = np.asarray(mcmc.get_extra_fields()["num_steps"])
    sample_steps = int(sample_steps_arr.sum())

    posterior = mcmc.get_samples(group_by_chain=(grad_budget is not None))

    if grad_budget is not None:
        k = _truncate_nuts_to_budget(sample_steps_arr, NUTS_CHAINS, NUTS_DRAWS, grad_budget)
        posterior = {name: arr[:, :k] for name, arr in posterior.items()}
        truncated_steps = sample_steps_arr.reshape(NUTS_CHAINS, NUTS_DRAWS)[:, :k].sum()
        sample_steps = int(truncated_steps)
        posterior = {name: arr.reshape((-1,) + arr.shape[2:]) for name, arr in posterior.items()}

    gradient_evals = warmup_steps + sample_steps

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
SVI_RANK    = 20


def run_svi(data: dict[str, Any], cfg: BNNConfig, seed: int) -> tuple[np.ndarray, float, int]:
    """SVI on the same `bnn` model as run_nuts -- see toy_bnn_grid.py::run_svi
    for the full guide/ELBO rationale. Full-batch, unchanged here."""
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
# Persistence -- same payload schema as toy_bnn_grid.py's save_run, plus a
# grad_batch_size field so a minibatch run and its full-batch reference are
# distinguishable in the same output tree.
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
             grid_t_max_log: Optional[list[float]] = None,
             grad_budget: Optional[int] = "unset",
             grad_batch_size: Optional[int] = "unset") -> None:
    # See toy_bnn_grid.py::save_run's "unset" sentinel comment -- same idea,
    # extended to grad_batch_size.
    if grad_budget == "unset":
        grad_budget = GRAD_BUDGET
    if grad_batch_size == "unset":
        grad_batch_size = GRAD_BATCH_SIZE
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
        "grad_budget":      grad_budget,
        "grad_batch_size":  grad_batch_size,
    }, out_path)


# ===========================================================================
# Per-dataset runners -- identical to toy_bnn_grid.py's, except the four
# grid runners build Mb* samplers (via the builders above) and pass
# grad_batch_size through to save_run.
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
    result = sampler.sample(N=N_SKELETON, x0=x_ref, diagnostics=True, grad_budget=GRAD_BUDGET)
    elapsed = time.perf_counter() - t0

    samples = resample_zigzag_path_torch(
        result["positions"], result["velocities"], result["times"],
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    n_events = result["positions"].shape[0]
    print(f"      sampled {n_events} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations)")

    out_path = sd / "grid_zigzag.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_zigzag",
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=n_events,
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
    result = sampler.sample(N=N_SKELETON, x0=x_ref, diagnostics=True, grad_budget=GRAD_BUDGET)
    elapsed = time.perf_counter() - t0

    samples = resample_zigzag_path_sticky_torch(
        result["positions"], result["velocities"], result["times"],
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    sparsity = float(result["frozen_mask_final"].float().mean())
    n_events = result["positions"].shape[0]
    print(f"      sampled {n_events} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, "
          f"final sparsity {sparsity:.2f})")

    out_path = sd / "grid_sticky_zigzag.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_sticky_zigzag",
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=n_events,
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
    result = sampler.sample(N=N_SKELETON, diagnostics=True, grad_budget=GRAD_BUDGET)
    elapsed = time.perf_counter() - t0

    samples = resample_boomerang_path_torch(
        result["positions"], result["velocities"], result["times"], x_ref,
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    n_events = result["positions"].shape[0]
    print(f"      sampled {n_events} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations)")

    out_path = sd / "grid_boomerang.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_boomerang",
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=n_events,
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
    result = sampler.sample(N=N_SKELETON, diagnostics=True, grad_budget=GRAD_BUDGET)
    elapsed = time.perf_counter() - t0

    samples = resample_boomerang_path_sticky_torch(
        result["positions"], result["velocities"], result["times"], x_ref,
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    sparsity = float(result["frozen_mask_final"].float().mean())
    n_events = result["positions"].shape[0]
    print(f"      sampled {n_events} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, "
          f"final sparsity {sparsity:.2f})")

    out_path = sd / "grid_sticky_boomerang.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_sticky_boomerang",
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=n_events,
        bound_violations=result["bound_violations"],
        gradient_evals=result["gradient_evals"],
        grid_t_max_log=result["grid_t_max_log"],
    )
    print(f"      saved {samples.shape[0]} samples (thinned to {N_SAVE}) -> {out_path}")


def run_nuts_dataset(dataset_name: str, split_id: int, data: dict[str, Any],
                      cfg: BNNConfig, sd: Path) -> None:
    seed = BASE_SEED + split_id

    samples_np, elapsed, gradient_evals = run_nuts(data, cfg, seed, grad_budget=GRAD_BUDGET)
    samples = torch.tensor(samples_np)

    print(f"      sampled {samples.shape[0]} NUTS draws in {elapsed:.1f}s "
          f"({gradient_evals} gradient evals)")

    out_path = sd / "nuts.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="nuts",
        samples=samples, x_ref=None, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=samples.shape[0], bound_violations=0,
        gradient_evals=gradient_evals,
        grad_batch_size=None,  # NUTS is always full-batch here
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
        grad_batch_size=None,  # SVI is always full-batch here
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
          f"seed={BASE_SEED + split_id} | grad_batch_size={GRAD_BATCH_SIZE} ---")

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
    global N_SKELETON, N_RESAMPLE, GRAD_BUDGET, GRAD_BATCH_SIZE

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=list(TOY_DATASETS),
                         choices=list(TOY_DATASETS))
    parser.add_argument("--samplers", nargs="+", default=list(GRID_SAMPLER_NAMES),
                         choices=list(SAMPLER_NAMES),
                         help="Default: the four grid PDMP samplers (the ones minibatching "
                              "applies to). nuts/svi are still selectable for a full-batch "
                              "reference but are excluded from the default.")
    parser.add_argument("--splits", nargs="+", type=int, default=[0])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--toy-dir", type=Path, default=TOY_DIR)
    parser.add_argument("--n-skeleton", type=int, default=N_SKELETON,
                         help="Overrides the module-level N_SKELETON default (see "
                              "toy_bnn_grid.py's identical flag).")
    parser.add_argument("--n-resample", type=int, default=N_RESAMPLE,
                         help="Overrides the module-level N_RESAMPLE default.")
    parser.add_argument("--grad-budget", type=int, default=None,
                         help="If set, every grid sampler stops as soon as this many real "
                              "gradient evaluations have been spent (see toy_bnn_grid.py's "
                              "identical flag). --n-skeleton still acts as an outer safety cap.")
    parser.add_argument("--grad-batch-size", type=int, default=None,
                         help="If set, every grid-bound episode of the four grid PDMP samplers "
                              "is served from a fresh random minibatch of this size (drawn from "
                              "the training set, likelihood rescaled by N_full/batch_size) "
                              "instead of the full training set -- a statistically uncorrected "
                              "subsampled-gradient PDMP, matching what fast_mnist_cnn.py / "
                              "fast_cifar_resnet.py do. None (default) preserves exact "
                              "full-batch behavior, reproducing toy_bnn_grid.py. For an "
                              "apples-to-apples ablation, run toy_bnn_grid.py (or this script "
                              "with --grad-batch-size unset) for the full-batch reference and "
                              "this script with --grad-batch-size B for the minibatch arm, "
                              "same --datasets/--splits.")
    args = parser.parse_args()

    N_SKELETON = args.n_skeleton
    N_RESAMPLE = args.n_resample
    GRAD_BUDGET = args.grad_budget
    GRAD_BATCH_SIZE = args.grad_batch_size

    args.out.mkdir(parents=True, exist_ok=True)

    print("\nLoading toy 1D datasets...")
    toys = {n: load_toy(n, args.toy_dir) for n in args.datasets}

    print(f"\nRunning {args.datasets} | samplers: {args.samplers} | splits: {args.splits} "
          f"| grad_batch_size: {GRAD_BATCH_SIZE}")
    for ds in args.datasets:
        data, cfg = toys[ds]
        for split_id in args.splits:
            run_dataset(ds, split_id, data, cfg, args.out, args.samplers, resume=args.resume)


if __name__ == "__main__":
    main()
