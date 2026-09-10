"""
Negative-MAP (mirror-mode) ablation of
sazz/gpu_friendly/scripts/toy_bnn_grid.py, built on top of
toy_bnn_grid_minibatch.py so it can be run alongside BOTH the full-batch
and the minibatched toys (--grad-batch-size carries over unchanged).

Idea: the toy nets are single-hidden-layer tanh FFNs, layer_sizes
[1, 100, 1]. tanh is odd, so negating the FIRST layer's weights AND biases
(W0, b0) together with the SECOND layer's weights (W1) -- but NOT the
second layer's bias (b1) -- maps the network to a DIFFERENT parameter
vector that computes the EXACT SAME function:

    h1  = tanh(W0 x + b0)
    out = W1 h1 + b1

    tanh(-(W0 x + b0)) = -tanh(W0 x + b0) = -h1     (tanh odd)
    (-W1)(-h1) + b1    =  W1 h1 + b1    = out        (b1 untouched)

So energy(x_ref_neg) == energy(x_ref) to floating point, and x_ref_neg is
a genuine, distinct posterior mode (the "mirror" mode) -- equally good,
not the sign-flipped-output function you'd get by ALSO negating b1.

This script finds x_ref exactly as toy_bnn_grid.py does (Adam from a
random init -> whichever of the two mirror modes that seed lands near),
then flips it to x_ref_neg and starts every grid sampler there instead:
  * ZigZag / sticky ZigZag: x0 = x_ref_neg.
  * Boomerang / sticky Boomerang: preprocess(x_ref = x_ref_neg, ...).
Sigma_inv is REUSED UNCHANGED: it is the diagonal empirical-Fisher +
prior precision (find_reference_bnn), and both terms are invariant under a
per-coordinate sign flip (d/dbeta_i of a flipped coord flips sign, so
(d/dbeta_i)^2 -- and the fan-in prior precision, which depends only on
layer shape -- are identical at the mirror mode).

Only single-hidden-layer FFNs are supported (asserted): for a deeper net
the flip pattern is more involved (alternating weight layers, interior
biases flip, the final bias does not), and all four toys are [1, 100, 1]
anyway. Fixed noise (noise_std passed in build_target) => no log_sigma
coordinate => nothing noise-related to flip.

--negate-map (default) does the flip; --no-negate-map turns it off, in
which case this script is exactly toy_bnn_grid_minibatch.py.
--grad-batch-size None (default) => full batch (Mb* samplers are no-op
passthroughs), so the default invocation is "toy_bnn_grid.py started from
the mirror mode".

Usage:
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid_negative_map
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid_negative_map --grad-batch-size 16
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid_negative_map --no-negate-map   # == minibatch script
    python -m sazz.gpu_friendly.scripts.toy_bnn_grid_negative_map --samplers grid_sticky_boomerang
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
# full training set. None (default) => exact full-batch. Identical semantics
# to toy_bnn_grid_minibatch.py.
GRAD_BATCH_SIZE: Optional[int] = None

# If True (default; --no-negate-map disables), every grid sampler is started
# from the MIRROR MAP (x_ref with W0/b0/W1 negated, b1 kept) instead of the
# raw x_ref find_reference_bnn returns. See the module docstring.
NEGATE_MAP: bool = True

# Tolerance for the "energy(x_ref_neg) == energy(x_ref)" invariance check
# done once per grid run before sampling. float64 CPU path: a few 1e-7 of
# relative slop is normal (100-wide tanh sum, cancellation); float32 CUDA
# would need this looser.
NEGATE_MAP_ENERGY_RTOL = 1e-5

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

OUT_DIR = Path("results/paper/toy_bnns_negative_map")
TOY_DIR = Path("datasets/toy_1d")

TOY_DATASETS = ("hernandez", "gap", "sharp", "multiscale")
# nuts/svi are still selectable via --samplers (full-batch reference), but
# not in the default set -- neither minibatching nor "start from the mirror
# MAP" applies to them (they explore the full posterior regardless of init).
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
# Negative / mirror MAP -- the one thing this script adds.
# ===========================================================================

def build_negate_mask(bm: BayesianModule) -> torch.Tensor:
    """
    Boolean [D] mask, True where beta_i must be sign-flipped to move from
    x_ref to the mirror mode of a single-hidden-layer tanh FFN.

    FFN stores its Linears as `layers.0`, `layers.1`, ... (see
    neural_networks.FFN). For layer_sizes [n_in, H, n_out] there are
    exactly two Linears:
        layers.0.weight [H, n_in], layers.0.bias [H]   -> FLIP (W0, b0)
        layers.1.weight [n_out, H], layers.1.bias [n_out] -> FLIP W1, KEEP b1

    So the mask is True for every parameter EXCEPT layers.1.bias. Asserts
    the two-Linear structure -- deeper nets need a different flip pattern
    (see module docstring) and none of the toys are deeper.
    """
    names = [n for n, _ in bm.module.named_parameters()]
    linear_idxs = sorted({
        int(n.split(".")[1]) for n in names if n.startswith("layers.")
    })
    assert linear_idxs == [0, 1], (
        f"negate-map only supports single-hidden-layer FFNs (2 Linears); "
        f"got layers {linear_idxs} from {names}"
    )
    last_bias = f"layers.{linear_idxs[-1]}.bias"

    mask = torch.zeros(bm.D, dtype=torch.bool, device=bm.device)
    idx = 0
    for name, p in bm.module.named_parameters():
        n = p.numel()
        if name != last_bias:
            mask[idx: idx + n] = True
        idx += n
    assert idx == bm.D or (bm.learns_noise and idx == bm.D - 1), (
        f"parameter flattening covered {idx} of D={bm.D}"
    )
    # A trailing learned-noise coordinate (not present for the fixed-noise
    # toys) would be left un-flipped by the loop above, which is correct.
    return mask


def negate_map(bm: BayesianModule, x_ref: torch.Tensor) -> torch.Tensor:
    """x_ref with the mirror-mode coordinates sign-flipped; energy unchanged.
    Verifies the invariance and raises if it does not hold to
    NEGATE_MAP_ENERGY_RTOL (a real guard: catches an architecture whose
    flip pattern this helper does not actually implement)."""
    mask = build_negate_mask(bm)
    x_neg = torch.where(mask, -x_ref, x_ref)

    with torch.no_grad():
        e_ref = float(bm.energy(x_ref))
        e_neg = float(bm.energy(x_neg))
    denom = max(abs(e_ref), 1.0)
    rel = abs(e_neg - e_ref) / denom
    if rel > NEGATE_MAP_ENERGY_RTOL:
        raise AssertionError(
            f"negate_map broke energy invariance: energy(x_ref)={e_ref:.8g}, "
            f"energy(x_ref_neg)={e_neg:.8g}, rel diff {rel:.2e} > "
            f"{NEGATE_MAP_ENERGY_RTOL:.0e}. The sign-flip pattern in "
            f"build_negate_mask does not match this architecture."
        )
    print(f"  negate-map: energy(x_ref)={e_ref:.6g}  energy(x_ref_neg)={e_neg:.6g}  "
          f"(rel diff {rel:.2e}); flipping {int(mask.sum())}/{bm.D} coords")
    return x_neg


def maybe_negate_map(bm: BayesianModule, x_ref: torch.Tensor) -> torch.Tensor:
    """x_ref_neg if NEGATE_MAP, else x_ref unchanged."""
    if not NEGATE_MAP:
        return x_ref
    return negate_map(bm, x_ref)


# ===========================================================================
# Minibatch gradient -- identical to toy_bnn_grid_minibatch.py.
# ===========================================================================

def build_minibatch_grad_target(bm: BayesianModule, batch_size: int):
    """
    Returns (grad_target, resample_fn). See
    toy_bnn_grid_minibatch.py::build_minibatch_grad_target for the full
    rationale -- this is a verbatim copy (kept local so this script stays
    standalone, same convention as the *_grid.py scripts).
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
# Sampler builders -- same config as toy_bnn_grid.py; the caller passes in
# the (possibly negated) x_ref so preprocess() anchors the Boomerang
# reference measure at the mirror mode.
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
# Not affected by --grad-batch-size or --negate-map.
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
# Persistence -- same payload schema as toy_bnn_grid.py's save_run, plus
# grad_batch_size and negate_map fields so a mirror-MAP run is
# distinguishable from its full-batch / positive-MAP references.
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
             grad_batch_size: Optional[int] = "unset",
             negate_map: Optional[bool] = "unset") -> None:
    # See toy_bnn_grid.py::save_run's "unset" sentinel comment -- same idea,
    # extended to grad_batch_size / negate_map.
    if grad_budget == "unset":
        grad_budget = GRAD_BUDGET
    if grad_batch_size == "unset":
        grad_batch_size = GRAD_BATCH_SIZE
    if negate_map == "unset":
        negate_map = NEGATE_MAP
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
        "negate_map":       negate_map,
    }, out_path)


# ===========================================================================
# Per-dataset runners -- identical to toy_bnn_grid_minibatch.py's, except
# each grid runner computes x_ref_run = maybe_negate_map(bm, x_ref) right
# after build_target and starts the sampler there (x0 for ZigZag,
# preprocess anchor for Boomerang, and the x_ref stored in save_run).
# ===========================================================================

def run_grid_zigzag(dataset_name: str, split_id: int, data: dict[str, Any],
                        cfg: BNNConfig, sd: Path) -> None:
    seed = BASE_SEED + split_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, x_ref, _ = build_target(data, cfg)
    print(f"  D = {bm.D}")
    x_ref_run = maybe_negate_map(bm, x_ref)

    sampler = build_zigzag_sampler(bm)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, x0=x_ref_run, diagnostics=True, grad_budget=GRAD_BUDGET)
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
        samples=samples, x_ref=x_ref_run, cfg=cfg, y_std=data["y_std"],
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
    x_ref_run = maybe_negate_map(bm, x_ref)

    sampler = build_sticky_zigzag_sampler(bm, cfg)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, x0=x_ref_run, diagnostics=True, grad_budget=GRAD_BUDGET)
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
        samples=samples, x_ref=x_ref_run, cfg=cfg, y_std=data["y_std"],
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
    x_ref_run = maybe_negate_map(bm, x_ref)

    sampler = build_boomerang_sampler(bm, x_ref_run, Sigma_inv)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, diagnostics=True, grad_budget=GRAD_BUDGET)
    elapsed = time.perf_counter() - t0

    samples = resample_boomerang_path_torch(
        result["positions"], result["velocities"], result["times"], x_ref_run,
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    n_events = result["positions"].shape[0]
    print(f"      sampled {n_events} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations)")

    out_path = sd / "grid_boomerang.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_boomerang",
        samples=samples, x_ref=x_ref_run, cfg=cfg, y_std=data["y_std"],
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
    x_ref_run = maybe_negate_map(bm, x_ref)

    sampler = build_sticky_boomerang_sampler(bm, cfg, x_ref_run, Sigma_inv)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, diagnostics=True, grad_budget=GRAD_BUDGET)
    elapsed = time.perf_counter() - t0

    samples = resample_boomerang_path_sticky_torch(
        result["positions"], result["velocities"], result["times"], x_ref_run,
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
        samples=samples, x_ref=x_ref_run, cfg=cfg, y_std=data["y_std"],
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
        negate_map=False,      # NUTS explores the full posterior regardless of init
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
        negate_map=False,      # SVI fits the full posterior regardless of init
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
          f"seed={BASE_SEED + split_id} | grad_batch_size={GRAD_BATCH_SIZE} | "
          f"negate_map={NEGATE_MAP} ---")

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
    global N_SKELETON, N_RESAMPLE, GRAD_BUDGET, GRAD_BATCH_SIZE, NEGATE_MAP

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=list(TOY_DATASETS),
                         choices=list(TOY_DATASETS))
    parser.add_argument("--samplers", nargs="+", default=list(GRID_SAMPLER_NAMES),
                         choices=list(SAMPLER_NAMES),
                         help="Default: the four grid PDMP samplers. nuts/svi are still "
                              "selectable for a full-batch reference but are excluded from "
                              "the default (starting from the mirror MAP is meaningless for "
                              "them).")
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
                              "instead of the full training set. Identical semantics to "
                              "toy_bnn_grid_minibatch.py -- lets this ablation run alongside "
                              "both the full-batch and minibatched toys. None (default) = "
                              "exact full batch.")
    neg = parser.add_mutually_exclusive_group()
    neg.add_argument("--negate-map", dest="negate_map", action="store_true", default=True,
                     help="(default) Start every grid sampler from the MIRROR MAP: x_ref with "
                          "the first layer's weights+biases and the second layer's weights "
                          "sign-flipped (the second layer's bias is kept). tanh's odd "
                          "symmetry makes this an exact energy-preserving map to a distinct, "
                          "equally-good posterior mode.")
    neg.add_argument("--no-negate-map", dest="negate_map", action="store_false",
                     help="Disable the flip -- start from the raw x_ref, making this script "
                          "identical to toy_bnn_grid_minibatch.py.")
    args = parser.parse_args()

    N_SKELETON = args.n_skeleton
    N_RESAMPLE = args.n_resample
    GRAD_BUDGET = args.grad_budget
    GRAD_BATCH_SIZE = args.grad_batch_size
    NEGATE_MAP = args.negate_map

    args.out.mkdir(parents=True, exist_ok=True)

    print("\nLoading toy 1D datasets...")
    toys = {n: load_toy(n, args.toy_dir) for n in args.datasets}

    print(f"\nRunning {args.datasets} | samplers: {args.samplers} | splits: {args.splits} "
          f"| grad_batch_size: {GRAD_BATCH_SIZE} | negate_map: {NEGATE_MAP}")
    for ds in args.datasets:
        data, cfg = toys[ds]
        for split_id in args.splits:
            run_dataset(ds, split_id, data, cfg, args.out, args.samplers, resume=args.resume)


if __name__ == "__main__":
    main()
