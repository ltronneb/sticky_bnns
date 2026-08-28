"""
Grid-bound Boomerang (Andral & Kamatani 2024) on UCI regression BNN benchmarks 
Activation is tanh by default, as this is smooth
.

Six samplers: "grid_zigzag", "grid_sticky_zigzag", "grid_boomerang",
"grid_sticky_boomerang", "nuts", "nuts_horseshoe"

Usage:
    python -m sazz.gpu_friendly.scripts.uci_bnn_grid
    python -m sazz.gpu_friendly.scripts.uci_bnn_grid --datasets boston
    python -m sazz.gpu_friendly.scripts.uci_bnn_grid --datasets boston --splits 0
    python -m sazz.gpu_friendly.scripts.uci_bnn_grid --datasets boston --splits 0 --samplers grid_boomerang
    python -m sazz.gpu_friendly.scripts.uci_bnn_grid --samplers nuts nuts_horseshoe
    
python3 -m sazz.gpu_friendly.scripts.uci_bnn_grid --datasets boston energy naval --samplers nuts nuts_horseshoe
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

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
# Config
# ===========================================================================
DEVICE = (
    "cuda" if torch.cuda.is_available()
    #else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64
torch.set_default_dtype(torch.float32)


N_SKELETON  = 100_000
N_RESAMPLE  = 50_000
BURNIN_FRAC = 0.2
BASE_SEED   = 42

# If set (via --grad-budget), every sampler stops as soon as this many real
# gradient evaluations have been spent, instead of running to N_SKELETON
# skeleton events / NUTS_DRAWS samples. None (default) reproduces today's
# exact event-count-driven behavior in every runner below.
GRAD_BUDGET: Optional[int] = None

SIGMA_INV_SCALE = 0.1 #1.0, 10.0
REFRESH_RATE = 1.0
GAMMA = 0.01

GRID_N_SEGMENTS = 60
GRID_ALPHA_PLUS = 1.01
GRID_ALPHA_MINUS = 1.04

GRID_T_MAX_INIT_ZIGZAG = 0.002
GRID_SPACING_ZIGZAG = 0.0002

GRID_T_MAX_INIT_BOOM = 2e-2
GRID_SPACING_BOOM = 3e-3 #math.pi / 128

GRID_STICKY_BOOM_SPACING = GRID_SPACING_BOOM
GRID_STICKY_ZIGZAG_SPACING = GRID_SPACING_ZIGZAG


GRID_STICKY_COLD_START_THRESHOLD = None

NUTS_DRAWS  = 2_000
NUTS_WARMUP = 1_000
NUTS_CHAINS = 4
N_SAVE      = NUTS_DRAWS * NUTS_CHAINS

# Three architecture variants for a size-sensitivity comparison across the UCI tables 
# -- "small" matches Izmailov et al.'s 1x50 shape 
# "deep_narrow" keeps that width but adds depth, 
# "deep_wide" mirrors Goan et al.'s 3-hidden-layer shape
# (512-256-128) scaled down by half for tractability. 
# Selected via --hidden-variant; 
HIDDEN_VARIANTS = {
    "small":       [50],
    "deep_narrow": [50, 50, 50],
    "deep_wide":   [256, 128, 64],
}
DEFAULT_HIDDEN_VARIANT = "small"

OUT_DIR = Path("results/grid/uci_bnn")

UCI_DATASETS  = ("boston", "naval", "energy", "yacht", "concrete")
SAMPLER_NAMES = ("grid_zigzag", "grid_sticky_zigzag", "grid_boomerang", "grid_sticky_boomerang", "nuts", "nuts_horseshoe")


@dataclass
class BNNConfig:
    layer_sizes: list[int]
    activation: str = "tanh"
    prior_sigma_scale: float = 0.3
    prior_std_weight: float = 1.0
    prior_std_bias: float = 1.0
    fan_in_scaling: bool = True
    adam_steps: int = 20000
    prior_inclusion_weight: float = 0.3  # sticky-only: spike-and-slab inclusion prob for kappa


def configs_for(input_dims: dict[str, int], hidden: list[int],
                 prior_inclusion_weight: float = 0.3) -> dict[str, BNNConfig]:
    cfgs: dict[str, BNNConfig] = {}
    for name in UCI_DATASETS:
        if name in input_dims:
            cfgs[name] = BNNConfig(
                layer_sizes=[input_dims[name], *hidden, 1],
                prior_sigma_scale=0.01 if name == "naval" else 0.3,
                prior_inclusion_weight=prior_inclusion_weight,
            )
    return cfgs


# ===========================================================================
# Data loading -- same sources as sazz/scripts/bnns/uci_bnn.py, independent
# implementation (no shared code with that file).
# ===========================================================================

def load_raw_datasets(datasets: tuple[str, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if "boston" in datasets:
        from sklearn.datasets import fetch_openml
        boston = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
        raw["boston"] = (
            boston.data.values.astype(float),
            boston.target.values.astype(float),
        )

    if "naval" in datasets:
        df_naval = pd.read_csv("datasets/naval_data.txt", sep=r"\s+", header=None)
        raw["naval"] = (
            df_naval.iloc[:, :16].values.astype(float),
            df_naval.iloc[:, 16].values.astype(float),
        )

    if "energy" in datasets:
        df_energy = pd.read_excel("datasets/energy_data.xlsx")
        raw["energy"] = (
            df_energy.iloc[:, :8].values.astype(float),
            df_energy.iloc[:, 8].values.astype(float),
        )

    if "yacht" in datasets:
        df_yacht = pd.read_csv("datasets/yacht_hydrodynamics.data", sep=r"\s+", header=None)
        raw["yacht"] = (
            df_yacht.iloc[:, :6].values.astype(float),
            df_yacht.iloc[:, 6].values.astype(float),
        )

    if "concrete" in datasets:
        # Legacy .xls (not .xlsx) -- needs the xlrd engine, not openpyxl
        # (energy_data.xlsx above uses openpyxl's default automatically).
        # pip install xlrd if this raises ImportError.
        df_concrete = pd.read_excel(
            "datasets/concrete+compressive+strength 3/Concrete_Data.xls", engine="xlrd",
        )
        raw["concrete"] = (
            df_concrete.iloc[:, :8].values.astype(float),
            df_concrete.iloc[:, 8].values.astype(float),
        )

    return raw


def make_split(X: np.ndarray, y: np.ndarray, seed: int, dtype, device,
                test_frac: float = 0.1) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_frac, random_state=seed)
    x_mean, x_std = X_tr.mean(0), X_tr.std(0)
    x_std[x_std == 0] = 1.0
    y_mean, y_std = y_tr.mean(), y_tr.std()

    return {
        "X_train": torch.tensor((X_tr - x_mean) / x_std, dtype=dtype, device=device),
        "y_train": torch.tensor((y_tr - y_mean) / y_std, dtype=dtype, device=device),
        "X_test":  torch.tensor((X_te - x_mean) / x_std, dtype=dtype, device=device),
        "y_test":  torch.tensor((y_te - y_mean) / y_std, dtype=dtype, device=device),
        "y_std":   float(y_std),
    }


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

    # noise_std omitted -> learned (default), with a HalfNormal(prior_sigma_scale)
    # prior on sigma; find_reference_bnn's log_sigma curvature patch (warmup.py)
    # picks this up automatically via bm.learns_noise/bm.prior_sigma_scale.
    bm = BayesianModule.build(
        module, likelihood="gaussian", X=X, y=y,
        prior_precision=prec, prior_sigma_scale=cfg.prior_sigma_scale,
        dtype=dtype, device=device,
    )

    x_ref, Sigma_inv = find_reference_bnn(
        bm, n_steps=cfg.adam_steps, lr=1e-2, dtype=dtype, device=torch.device(device),
    )

    return bm, x_ref, Sigma_inv


# ===========================================================================
# Samplers
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
    kappa/can_freeze constructed identically to build_sticky_boomerang_sampler.
    ZigZag has no x_ref/Sigma_inv dependency, so unlike its Boomerang
    counterpart this builder only needs bm/cfg -- x_ref is still used by the
    caller as sample()'s x0 warm-start, but that's independent of construction.
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
# NUTS / NUTS-HS via NumPyro -- ported from sazz/scripts/bnns/uci_bnn.py's
# run_nuts/run_nuts_horseshoe, kept close to the original. Initializes NUTS
# chains from x_ref (the grid pipeline's own MAP estimate) so PDMP and NUTS
# start from the same point, same as the old script did.
# ===========================================================================

def _truncate_nuts_to_budget(sample_steps: np.ndarray, num_chains: int, num_samples: int,
                              grad_budget: int) -> int:
    """Given the flat per-draw leapfrog step counts from
    get_extra_fields()["num_steps"] (shape (num_chains*num_samples,),
    row-major -- chain 0's num_samples draws, then chain 1's, etc.,
    confirmed to match get_samples(group_by_chain=False)'s own ordering by
    direct comparison), find the largest k such that keeping the first k
    draws FROM EVERY CHAIN (not a global flat prefix) stays within
    grad_budget.

    grad_budget here governs the SAMPLING phase only, not warmup -- same
    convention as the Goan TF/TFP budget-chunked loop (see run_tf_bps/
    run_tf_boomerang's grad_budget docstring): warmup's job is adaptation,
    not posterior coverage, and the fixed-budget comparison's fairness
    claim is about the *sampling* phase's gradient cost. gradient_evals in
    the returned/saved result still reports warmup_steps + the truncated
    sampling cost (the true total spend), it's just that warmup itself is
    never truncated/capped by grad_budget.

    Truncating per-chain-equally (not as one flat prefix across all
    chains-concatenated) keeps every returned chain the same length --
    a ragged "chain 0 got 50 draws, chain 1 got 3" result would be an
    awkward, arguably-wrong thing to hand downstream code that assumes a
    rectangular [n_chains, n_samples, ...] or flattened [n_total, ...]
    posterior array.

    Returns k (number of post-warmup draws to keep PER CHAIN), in
    [0, num_samples]. k=num_samples means the budget was never binding
    (the full ceiling run already fit) -- this is the "trim is a no-op"
    case verified separately.
    """
    per_chain = sample_steps.reshape(num_chains, num_samples)
    cum_per_chain = np.cumsum(per_chain, axis=1)  # [num_chains, num_samples]
    # Total spend if we keep k draws from every chain = sum over chains of
    # each chain's own cumulative cost at k -- NOT necessarily k times a
    # single chain's cost, since per-draw leapfrog counts vary chain to
    # chain (dynamic tree-doubling, no shared trajectory).
    total_cum = cum_per_chain.sum(axis=0)  # [num_samples], summed across chains
    fits = np.nonzero(total_cum <= grad_budget)[0]
    return int(fits[-1]) + 1 if len(fits) > 0 else 0


def run_nuts(data: dict[str, Any], cfg: BNNConfig, seed: int,
             x_ref: Optional[torch.Tensor] = None,
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
        # Learned noise: same HalfNormal(prior_sigma_scale) prior on sigma
        # used by the PDMP samplers (BayesianModule.build), so NUTS and PDMP
        # are sampling the identical model, not a fixed-noise approximation
        # of it.
        sigma = numpyro.sample("sigma", dist.HalfNormal(cfg.prior_sigma_scale))
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("y", dist.Normal(h.squeeze(-1), sigma), obs=y)

    init_params = None
    if x_ref is not None:
        x0 = x_ref.cpu().numpy()
        init_params = {}
        offset = 0
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            W_size = n_out * n_in
            W = jnp.array(x0[offset:offset + W_size].reshape(n_out, n_in))
            init_params[f"W{i}"] = jnp.broadcast_to(W, (NUTS_CHAINS,) + W.shape)
            offset += W_size
            b = jnp.array(x0[offset:offset + n_out])
            init_params[f"b{i}"] = jnp.broadcast_to(b, (NUTS_CHAINS,) + b.shape)
            offset += n_out
        # x_ref's trailing coordinate is log_sigma (BayesianModule.build
        # appends it when noise is learned); NumPyro samples sigma directly
        # on the positive reals.
        sigma0 = jnp.exp(jnp.array(x0[offset]))
        init_params["sigma"] = jnp.broadcast_to(sigma0, (NUTS_CHAINS,))

    kernel = NUTS(bnn, target_accept_prob=0.9, adapt_mass_matrix=True)
    mcmc = MCMC(kernel, num_warmup=NUTS_WARMUP, num_samples=NUTS_DRAWS,
                num_chains=NUTS_CHAINS, progress_bar=True)

    # Split into warmup()-then-run() (NumPyro's documented two-step pattern)
    # instead of one mcmc.run() call, so warmup-phase leapfrog steps are
    # visible via collect_warmup=True -- collect_warmup is a kwarg of
    # MCMC.warmup(), NOT of MCMC.__init__()/run() (verified against the
    # installed NumPyro 0.20.1's actual signatures). Passing
    # mcmc.post_warmup_state.rng_key to run() (NOT a fresh/repeated
    # PRNGKey(seed) call) is required for this to be a faithful
    # decomposition of a single mcmc.run() call: run() always re-derives
    # its sampling-phase key from whatever key it's given, so re-passing
    # the ORIGINAL seed would silently reuse/collide the RNG stream and
    # produce a genuinely different (not just re-labeled) trajectory --
    # confirmed by direct comparison: re-passing PRNGKey(seed) here gives
    # posterior samples that differ by up to ~0.6 in raw units from the
    # single-run() baseline, while post_warmup_state.rng_key gives
    # bit-identical samples. This is NumPyro's own documented pattern (see
    # MCMC.post_warmup_state's docstring example: "mcmc.run(mcmc.post_warmup_state.rng_key)").
    t0 = time.perf_counter()
    mcmc.warmup(jax.random.PRNGKey(seed), X=X, y=y, init_params=init_params,
                extra_fields=("num_steps",), collect_warmup=True)
    warmup_steps = int(np.asarray(mcmc.get_extra_fields()["num_steps"]).sum())
    mcmc.run(mcmc.post_warmup_state.rng_key, X=X, y=y, init_params=init_params,
             extra_fields=("num_steps",))
    elapsed = time.perf_counter() - t0

    # num_steps = leapfrog steps per sample; NUTS/HMC evaluates the
    # potential-energy gradient once per leapfrog step, so summing across
    # all samples/chains (warmup AND post-warmup) gives total gradient
    # evaluations -- the same currency as the PDMP samplers' gradient_evals,
    # for a fair cost-per-gradient-eval comparison across sampler families.
    # get_extra_fields() reflects whichever of warmup()/run() ran most
    # recently, so warmup_steps was captured right after warmup() above,
    # before run() overwrites it with the post-warmup fields.
    sample_steps_arr = np.asarray(mcmc.get_extra_fields()["num_steps"])
    sample_steps = int(sample_steps_arr.sum())

    posterior = mcmc.get_samples(group_by_chain=(grad_budget is not None))

    if grad_budget is not None:
        # "generous ceiling, then truncate" (v1; see module docstring for
        # why NUTS doesn't get a chunked-resumption budget loop like the
        # grid/Goan samplers -- dynamic tree-doubling means per-sample
        # gradient cost isn't knob-controllable ahead of time). NUTS_DRAWS
        # is the generous ceiling; find the largest per-chain draw count k
        # whose real summed leapfrog cost stays within grad_budget, then
        # trim every chain to k draws before flattening. If the ceiling
        # run never actually needed grad_budget's worth of draws, k ==
        # NUTS_DRAWS and this is a no-op (verified separately).
        k = _truncate_nuts_to_budget(sample_steps_arr, NUTS_CHAINS, NUTS_DRAWS, grad_budget)
        posterior = {name: arr[:, :k] for name, arr in posterior.items()}
        truncated_steps = sample_steps_arr.reshape(NUTS_CHAINS, NUTS_DRAWS)[:, :k].sum()
        sample_steps = int(truncated_steps)
        # Flatten [n_chains, k, ...] -> [n_chains*k, ...], same row-major
        # chain-then-sample ordering get_samples(group_by_chain=False)
        # itself uses (confirmed by direct comparison).
        posterior = {name: arr.reshape((-1,) + arr.shape[2:]) for name, arr in posterior.items()}

    gradient_evals = warmup_steps + sample_steps

    flat = []
    for i in range(len(layer_sizes) - 1):
        W = np.asarray(posterior[f"W{i}"])  # [n, n_out, n_in]
        b = np.asarray(posterior[f"b{i}"])  # [n, n_out]
        flat.append(W.reshape(W.shape[0], -1))
        flat.append(b)
    flat.append(np.log(np.asarray(posterior["sigma"]))[:, None])  # log_sigma, matches x_ref's convention
    return np.concatenate(flat, axis=1).astype(np.float64), elapsed, gradient_evals


def run_nuts_horseshoe(data: dict[str, Any], cfg: BNNConfig, seed: int,
                        x_ref: Optional[torch.Tensor] = None,
                        grad_budget: Optional[int] = None) -> tuple[np.ndarray, float, int]:
    """
    NUTS with a horseshoe prior on weights, half-Cauchy on biases -- see
    sazz/scripts/bnns/uci_bnn.py's run_nuts_horseshoe for the full math
    rationale. Output flatten convention matches run_nuts: [W0, b0, W1,
    b1, ...] with each W flattened row-major in (n_out, n_in). Auxiliary
    horseshoe latents (tau, lambda) are not included in the flat vector.
    """
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
            tau = numpyro.sample(f"tau{i}", dist.HalfCauchy(prior_std))
            lam = numpyro.sample(
                f"lam{i}", dist.HalfCauchy(jnp.ones((n_out, n_in))).to_event(2),
            )
            W_raw = numpyro.sample(
                f"W{i}_raw", dist.Normal(jnp.zeros((n_out, n_in)), 1.0).to_event(2),
            )
            W = numpyro.deterministic(f"W{i}", tau * lam * W_raw)
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
        # Learned noise -- same HalfNormal(prior_sigma_scale) prior as the
        # PDMP samplers and plain NUTS, so every sampler in this script
        # targets the identical model.
        sigma = numpyro.sample("sigma", dist.HalfNormal(cfg.prior_sigma_scale))
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("y", dist.Normal(h.squeeze(-1), sigma), obs=y)

    # Init from x_ref: tau=lam=1 so W = 1*1*W_raw = x_ref_W at start; tau/lam
    # adapt freely during warmup, horseshoe regularization kicks in from there.
    init_params = None
    if x_ref is not None:
        x0 = x_ref.cpu().numpy()
        init_params = {}
        offset = 0
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            W_size = n_out * n_in
            W_raw = jnp.array(x0[offset:offset + W_size].reshape(n_out, n_in))
            init_params[f"tau{i}"] = jnp.ones(NUTS_CHAINS)
            init_params[f"lam{i}"] = jnp.ones((NUTS_CHAINS, n_out, n_in))
            init_params[f"W{i}_raw"] = jnp.broadcast_to(W_raw, (NUTS_CHAINS,) + W_raw.shape)
            offset += W_size
            b = jnp.array(x0[offset:offset + n_out])
            init_params[f"b{i}"] = jnp.broadcast_to(b, (NUTS_CHAINS,) + b.shape)
            offset += n_out
        sigma0 = jnp.exp(jnp.array(x0[offset]))
        init_params["sigma"] = jnp.broadcast_to(sigma0, (NUTS_CHAINS,))

    kernel = NUTS(bnn, target_accept_prob=0.95, adapt_mass_matrix=True)
    mcmc = MCMC(kernel, num_warmup=NUTS_WARMUP, num_samples=NUTS_DRAWS,
                num_chains=NUTS_CHAINS, progress_bar=True)

    # See run_nuts's identical warmup()-then-run() split above for why --
    # collect_warmup=True makes warmup-phase leapfrog steps visible via
    # get_extra_fields(), which a single mcmc.run() call would silently omit.
    # run() takes post_warmup_state.rng_key (NOT a fresh PRNGKey(seed) call)
    # -- required for bit-identical posterior samples vs. a single run()
    # call, verified directly; see run_nuts's comment for the full story.
    t0 = time.perf_counter()
    mcmc.warmup(jax.random.PRNGKey(seed), X=X, y=y, init_params=init_params,
                extra_fields=("num_steps",), collect_warmup=True)
    warmup_steps = int(np.asarray(mcmc.get_extra_fields()["num_steps"]).sum())
    mcmc.run(mcmc.post_warmup_state.rng_key, X=X, y=y, init_params=init_params,
             extra_fields=("num_steps",))
    elapsed = time.perf_counter() - t0

    sample_steps_arr = np.asarray(mcmc.get_extra_fields()["num_steps"])
    sample_steps = int(sample_steps_arr.sum())

    posterior = mcmc.get_samples(group_by_chain=(grad_budget is not None))

    if grad_budget is not None:
        # See run_nuts's identical "generous ceiling, then truncate" logic
        # and _truncate_nuts_to_budget's docstring for the full rationale.
        k = _truncate_nuts_to_budget(sample_steps_arr, NUTS_CHAINS, NUTS_DRAWS, grad_budget)
        posterior = {name: arr[:, :k] for name, arr in posterior.items()}
        truncated_steps = sample_steps_arr.reshape(NUTS_CHAINS, NUTS_DRAWS)[:, :k].sum()
        sample_steps = int(truncated_steps)
        posterior = {name: arr.reshape((-1,) + arr.shape[2:]) for name, arr in posterior.items()}

    gradient_evals = warmup_steps + sample_steps

    flat = []
    for i in range(len(layer_sizes) - 1):
        W = np.asarray(posterior[f"W{i}"])  # deterministic tau*lam*W_raw
        b = np.asarray(posterior[f"b{i}"])
        flat.append(W.reshape(W.shape[0], -1))
        flat.append(b)
    flat.append(np.log(np.asarray(posterior["sigma"]))[:, None])
    return np.concatenate(flat, axis=1).astype(np.float64), elapsed, gradient_evals


# ===========================================================================
# Persistence
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
             grad_budget: Optional[int] = "unset") -> None:
    # "unset" sentinel (not None): lets a caller explicitly pass
    # grad_budget=None to record "definitely no budget," distinct from
    # "caller didn't say, fall back to the module-level GRAD_BUDGET global"
    # -- every runner in this script already reads GRAD_BUDGET implicitly
    # for the sample()/run_nuts(...) call itself, so defaulting here to the
    # same global (rather than requiring every one of the ~10 call sites
    # to pass it explicitly) keeps this provenance field accurate with a
    # single point of truth, not ten places that could drift out of sync.
    if grad_budget == "unset":
        grad_budget = GRAD_BUDGET
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "dataset":           dataset,
        "split_id":          split_id,
        "sampler":           sampler,
        "samples":           thin_to(samples, N_SAVE).cpu(),
        "x_ref":             x_ref.cpu() if x_ref is not None else None,
        "layer_sizes":       cfg.layer_sizes,
        "activation":        cfg.activation,
        "learned_noise":     True,  # noise is learned for every sampler in this script
        "prior_sigma_scale": cfg.prior_sigma_scale,
        "y_std":             y_std,
        "n_events":          n_events,
        "elapsed_sec":       elapsed_sec,
        "bound_violations":  bound_violations,
        "gradient_evals":    gradient_evals,
        "grid_t_max_log":    grid_t_max_log,
        "grad_budget":       grad_budget,  # the REQUESTED budget (provenance); gradient_evals is the actual spend
    }, out_path)


# ===========================================================================
# Per-(dataset, split) runners -- one BayesianModule build (MAP + Laplace)
# is shared by grid_boomerang/grid_sticky_boomerang (they need x_ref/
# Sigma_inv); NUTS/NUTS-HS also reuse the same x_ref for chain init, but
# only build the target once per split_id if needed (see run_split).
# ===========================================================================

def run_grid_zigzag(dataset_name: str, split_id: int, data: dict[str, Any],
                        cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    sampler = build_zigzag_sampler(bm)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, x0=x_ref, diagnostics=True, grad_budget=GRAD_BUDGET)
    elapsed = time.perf_counter() - t0

    samples = resample_zigzag_path_torch(
        result["positions"], result["velocities"], result["times"],
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    n_events = result["positions"].shape[0]  # actual events produced -- may be < N_SKELETON if grad_budget stopped early
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
                            cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    sampler = build_sticky_zigzag_sampler(bm, cfg)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, x0=x_ref, diagnostics=True, grad_budget=GRAD_BUDGET)
    elapsed = time.perf_counter() - t0

    samples = resample_zigzag_path_sticky_torch(
        result["positions"], result["velocities"], result["times"],
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    sparsity = float(result["frozen_mask_final"].float().mean())
    n_events = result["positions"].shape[0]  # actual events produced -- may be < N_SKELETON if grad_budget stopped early
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
                        cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    sampler = build_boomerang_sampler(bm, x_ref, Sigma_inv)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, diagnostics=True, grad_budget=GRAD_BUDGET)
    elapsed = time.perf_counter() - t0

    samples = resample_boomerang_path_torch(
        result["positions"], result["velocities"], result["times"], x_ref,
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    n_events = result["positions"].shape[0]  # actual events produced -- may be < N_SKELETON if grad_budget stopped early
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
                               cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    sampler = build_sticky_boomerang_sampler(bm, cfg, x_ref, Sigma_inv)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, diagnostics=True, grad_budget=GRAD_BUDGET)
    elapsed = time.perf_counter() - t0

    samples = resample_boomerang_path_sticky_torch(
        result["positions"], result["velocities"], result["times"], x_ref,
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    sparsity = float(result["frozen_mask_final"].float().mean())
    n_events = result["positions"].shape[0]  # actual events produced -- may be < N_SKELETON if grad_budget stopped early
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
                      cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    seed = BASE_SEED + split_id
    samples_np, elapsed, gradient_evals = run_nuts(data, cfg, seed, x_ref=x_ref, grad_budget=GRAD_BUDGET)
    samples = torch.tensor(samples_np)

    print(f"      sampled {samples.shape[0]} NUTS draws in {elapsed:.1f}s "
          f"({gradient_evals} gradient evals)")

    out_path = sd / "nuts.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="nuts",
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=samples.shape[0], bound_violations=0,
        gradient_evals=gradient_evals,
    )
    print(f"      saved {samples.shape[0]} samples -> {out_path}")


def run_nuts_horseshoe_dataset(dataset_name: str, split_id: int, data: dict[str, Any],
                                cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    seed = BASE_SEED + split_id
    samples_np, elapsed, gradient_evals = run_nuts_horseshoe(data, cfg, seed, x_ref=x_ref, grad_budget=GRAD_BUDGET)
    samples = torch.tensor(samples_np)

    print(f"      sampled {samples.shape[0]} NUTS-HS draws in {elapsed:.1f}s "
          f"({gradient_evals} gradient evals)")

    out_path = sd / "nuts_horseshoe.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="nuts_horseshoe",
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
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
    "nuts_horseshoe": run_nuts_horseshoe_dataset,
}


def run_split(dataset_name: str, split_id: int, data: dict[str, Any],
              cfg: BNNConfig, out_dir: Path, samplers: list[str], resume: bool) -> None:
    print(f"\n--- {dataset_name.upper()} split {split_id:02d} | "
          f"layers={cfg.layer_sizes} | act={cfg.activation} | "
          f"noise=learned (HalfNormal scale={cfg.prior_sigma_scale:.4f}) | "
          f"seed={BASE_SEED + split_id} ---")

    sd = split_dir(out_dir, dataset_name, split_id)

    pending = [
        s for s in samplers
        if not (resume and (sd / f"{s}.pt").exists())
    ]
    for s in samplers:
        if s not in pending:
            print(f"  [{s}] skipping — exists at {sd / f'{s}.pt'}")

    if not pending:
        return

    # One MAP/Laplace build per split, shared by every sampler that runs
    # this split (all six samplers use x_ref -- Boomerang-family PDMP for
    # the reference measure, ZigZag-family PDMP purely as an x0 warm-start,
    # NUTS/NUTS-HS for chain initialization).
    seed = BASE_SEED + split_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, x_ref, Sigma_inv = build_target(data, cfg)
    print(f"  D = {bm.D}")

    for sampler_name in pending:
        print(f"  [{sampler_name}]")
        torch.manual_seed(seed)
        np.random.seed(seed)
        SAMPLER_RUNNERS[sampler_name](dataset_name, split_id, data, cfg, sd, bm, x_ref, Sigma_inv)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    global N_SKELETON, N_RESAMPLE, GRAD_BUDGET

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=["boston"],
                         choices=list(UCI_DATASETS))
    parser.add_argument("--samplers", nargs="+", default=list(SAMPLER_NAMES),
                         choices=list(SAMPLER_NAMES))
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--hidden-variant", choices=list(HIDDEN_VARIANTS),
                         default=DEFAULT_HIDDEN_VARIANT,
                         help="Architecture size for this run -- see HIDDEN_VARIANTS. "
                              "Results are written under <out>/<hidden-variant>/... so "
                              "runs at different sizes never collide on disk.")
    parser.add_argument("--n-skeleton", type=int, default=N_SKELETON,
                         help="Overrides the module-level N_SKELETON default -- e.g. "
                              "lower this for larger hidden-variants (more D) to keep "
                              "per-split wall time down.")
    parser.add_argument("--n-resample", type=int, default=N_RESAMPLE,
                         help="Overrides the module-level N_RESAMPLE default -- independent "
                              "of --n-skeleton.")
    parser.add_argument("--grad-budget", type=int, default=None,
                         help="If set, every sampler stops as soon as this many real gradient "
                              "evaluations have been spent, instead of running to "
                              "--n-skeleton skeleton events / NUTS_DRAWS samples. "
                              "--n-skeleton still acts as an outer safety cap (raise it if "
                              "the budget is large enough to need more events than the "
                              "default allows). For a fair cross-family comparison, run "
                              "this script and goan_scripts/uci_bnn_tf_grid.py with the "
                              "SAME --grad-budget value, same --datasets/--splits/"
                              "--hidden-variant. Since .pt filenames (grid_zigzag.pt, "
                              "nuts.pt, ...) don't encode run mode, budget runs should "
                              "generally use a distinct --out directory rather than "
                              "overwrite/compare in place against event-count-driven runs.")
    parser.add_argument("--prior-inclusion-weight", type=float, default=0.3,
                         help="Sticky-only spike-and-slab prior inclusion probability (BNNConfig."
                              "prior_inclusion_weight, fed to build_kappa_from_inclusion). Lower "
                              "values shrink kappa (thaw rate), so coordinates that freeze stay "
                              "frozen longer -- reduces the effective active-D the sticky samplers "
                              "spend skeleton events resolving. Only affects grid_sticky_zigzag/"
                              "grid_sticky_boomerang; plain grid_zigzag/grid_boomerang/NUTS ignore it.")
    args = parser.parse_args()

    N_SKELETON = args.n_skeleton
    N_RESAMPLE = args.n_resample
    GRAD_BUDGET = args.grad_budget

    hidden = HIDDEN_VARIANTS[args.hidden_variant]
    # "small" keeps writing to the original flat <out>/<dataset>/... layout
    # (matches the pre-existing boston/energy results already on disk under
    # results/grid/uci_bnn/<dataset>/...) -- every other variant gets its
    # own <out>/<variant>/<dataset>/... subtree so the three architectures
    # never collide.
    out_dir = args.out if args.hidden_variant == DEFAULT_HIDDEN_VARIANT else args.out / args.hidden_variant
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw UCI datasets...")
    raw = load_raw_datasets(tuple(args.datasets))
    missing = [d for d in args.datasets if d not in raw]
    if missing:
        print(f"  (skipping {missing} -- data file(s) not found)")
    datasets_to_run = [d for d in args.datasets if d in raw]

    cfgs = configs_for({n: X.shape[1] for n, (X, _) in raw.items()}, hidden,
                        prior_inclusion_weight=args.prior_inclusion_weight)

    print(f"\nRunning {datasets_to_run} | samplers: {args.samplers} | splits: {args.splits} | "
          f"hidden_variant={args.hidden_variant} ({hidden}) | "
          f"N_SKELETON={N_SKELETON} N_RESAMPLE={N_RESAMPLE} "
          f"prior_inclusion_weight={args.prior_inclusion_weight}"
          f"Sigma inverse scale={SIGMA_INV_SCALE}")
    for ds in datasets_to_run:
        X, y = raw[ds]
        cfg = cfgs[ds]
        for split_id in args.splits:
            data = make_split(X, y, seed=BASE_SEED + split_id, dtype=DTYPE, device=DEVICE)
            run_split(ds, split_id, data, cfg, out_dir, list(args.samplers), resume=args.resume)


if __name__ == "__main__":
    main()
