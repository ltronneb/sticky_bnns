"""
    python -m sazz.scripts.toy_bnn --datasets hernandez
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sazz.samplers.AutomaticBoomerangSampler import AutomaticBoomerangSampler
from sazz.samplers.StickyAutomaticBoomerangSampler import (
    StickyAutomaticBoomerangSampler,
)
from sazz.samplers.AutomaticZigZagSampler import AutomaticZigZagSampler
from sazz.samplers.StickyAutomaticZigZagSampler import (
    StickyAutomaticZigZagSampler,
)
from sazz.models.priors import ModuleGaussianPrior
from sazz.models.likelihoods import ModuleGaussianLikelihood, ModuleGaussianLikelihoodLearnedNoise
from sazz.models.model import BayesianModel, TorchTarget
from sazz.utils.bnn_utils import (
    ParamSpec, build_prior_precision, make_kappa_from_inclusion,
)
from sazz.models.neural_networks import FFN
from sazz.utils.warmup import find_reference_bnn, tune_refresh_rate
from sazz.scripts.bnns.generate_toys import GENERATORS, DEFAULT_SEED
from sazz.utils.sampling import (
    resample_boomerang_path, resample_boomerang_path_sticky,
    resample_zigzag_path, resample_zigzag_path_sticky,
)


# ===========================================================================
# Config
# ===========================================================================

torch.set_default_dtype(torch.float64)

N_SKELETON  = 10_000
N_RESAMPLE  = 5_000
BURNIN_FRAC = 0.2
BASE_SEED   = 42

T_MAX_ZZ    = 0.1
GAMMA_ZZ    = 0.01

NUTS_DRAWS  = 2_000
NUTS_WARMUP = 1_000
NUTS_CHAINS = 4

# Saved chain length, shared by PDMPs (after path resampling + thinning) and
# NUTS (NUTS_DRAWS * NUTS_CHAINS by construction). Keeps storage modest and
# Monte-Carlo error comparable across samplers.
N_SAVE      = NUTS_DRAWS * NUTS_CHAINS

HIDDEN      = [50]

OUT_DIR     = Path("results/toy_bnns")
TOY_DIR     = Path("datasets/toy_1d")

PDMP_SAMPLER_NAMES = ("zigzag", "sticky_zigzag", "boomerang", "sticky_boomerang")
ALL_SAMPLER_NAMES  = PDMP_SAMPLER_NAMES + ("nuts", "nuts_horseshoe")
TOY_DATASETS       = ("hernandez", "gap", "sharp", "multiscale")

# Prior scales and architecture — edit here to change hyperparameters.
# noise_std is data-dependent and filled in at load time from the .pt file.
DATASET_CONFIGS = {
    "hernandez": dict(
        layer_sizes=[1, 100, 1], activation="relu",
        prior_std_weight=3.0, prior_std_bias=3.0,
        fan_in_scaling=True, adam_steps=3000,
        prior_inclusion_weight=[0.5, 0.5],
    ),
    "gap": dict(
        layer_sizes=[1, 100, 1], activation="tanh",
        prior_std_weight=3.0, prior_std_bias=3.0,
        fan_in_scaling=True, adam_steps=3000,
        prior_inclusion_weight=[0.5, 0.5],
    ),
    "sharp": dict(
        layer_sizes=[1, 100, 1], activation="tanh",
        prior_std_weight=3.0, prior_std_bias=3.0,
        fan_in_scaling=True, adam_steps=3000,
        prior_inclusion_weight=[0.5, 0.5],
    ),
    "multiscale": dict(
        layer_sizes=[1, 100, 1], activation="tanh",
        prior_std_weight=5.0, prior_std_bias=5.0,
        fan_in_scaling=True, adam_steps=3000,
        prior_inclusion_weight=[0.5, 0.5],
    ),
}


@dataclass
class BNNConfig:
    layer_sizes: list[int]
    activation: str = "relu"
    noise_std: float = 0.3
    prior_std_weight: float = 5.0
    prior_std_bias: float = 5.0
    fan_in_scaling: bool = True
    adam_steps: int = 5000
    prior_inclusion_weight: list[float] = field(default_factory=list)


# ===========================================================================
# Data loading
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
# target builder
# ===========================================================================

def build_target(data: dict[str, Any], cfg: BNNConfig,
                 dtype=torch.float64, device="cpu") -> TorchTarget:
    """Build a v2 regression BNN target via ParamSpec + functional_call."""
    X = data["X_train"].to(dtype=dtype, device=device)
    y = data["y_train"].to(dtype=dtype, device=device)

    module = FFN(cfg.layer_sizes, cfg.activation)
    spec = ParamSpec.from_module(module)

    prec = build_prior_precision(
        spec, cfg.prior_std_weight, cfg.prior_std_bias,
        cfg.fan_in_scaling, dtype, device,
    )
    prior = ModuleGaussianPrior(prec)
    likelihood = ModuleGaussianLikelihood(module, spec, X, y, cfg.noise_std)
    model = BayesianModel(prior, likelihood)

    x_ref, Sigma_inv = find_reference_bnn(
        model.energy, spec.D, model=model, dtype=dtype, device=device,
        reference="laplace_diag", n_steps=cfg.adam_steps, lr=1e-2,
    )

    return TorchTarget(
        name=f"bnn_v2_{'x'.join(map(str, cfg.layer_sizes))}_{cfg.activation}",
        D=spec.D,
        grad_target=model.grad_energy,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={"model": model, "spec": spec, "module": module,
              "layer_sizes": cfg.layer_sizes},
    )


def build_target_learned_noise(data, cfg, prior_sigma_scale=1.0,
                               dtype=torch.float64, device="cpu"):
    """v2 regression target with a learned noise scale.

    The sampled vector has length base_spec.D + 1, with log_sigma at the
    end. The prior on log_sigma is HalfNormal(prior_sigma_scale) on sigma,
    transformed to the log scale — encoded inside the likelihood class.
    The corresponding entry in the prior precision vector is therefore 0.
    """
    X = data["X_train"].to(dtype=dtype, device=device)
    y = data["y_train"].to(dtype=dtype, device=device)

    module = FFN(cfg.layer_sizes, cfg.activation)
    base_spec = ParamSpec.from_module(module)
    extended_spec = base_spec.with_extra_scalar("log_sigma", can_freeze=False)

    # Build precision for the EXTENDED spec (so prec has length D+1), then
    # zero the last entry — log_sigma's prior lives in the likelihood, not
    # in ModuleGaussianPrior.
    prec = build_prior_precision(
        extended_spec, cfg.prior_std_weight, cfg.prior_std_bias,
        cfg.fan_in_scaling, dtype, device,
    )
    prec[-1] = 0.0

    prior = ModuleGaussianPrior(prec)
    likelihood = ModuleGaussianLikelihoodLearnedNoise(
        module, base_spec, X, y, prior_sigma_scale=prior_sigma_scale,
    )
    model = BayesianModel(prior, likelihood)

    x_ref, Sigma_inv = find_reference_bnn(
        model.energy, extended_spec.D, model=model, dtype=dtype, device=device,
        reference="laplace_diag", n_steps=cfg.adam_steps, lr=1e-2,
    )
    sigma_map = x_ref[-1].exp()
    prior_curvature = 2.0 * sigma_map ** 2 / prior_sigma_scale ** 2
    if Sigma_inv.dim() == 1:
        Sigma_inv[-1] = Sigma_inv[-1] + prior_curvature
    else:
        Sigma_inv[-1, -1] = Sigma_inv[-1, -1] + prior_curvature

    return TorchTarget(
        name=f"bnn_v2ln_{'x'.join(map(str, cfg.layer_sizes))}_{cfg.activation}",
        D=extended_spec.D,
        grad_target=model.grad_energy,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={"model": model, "spec": extended_spec, "module": module,
              "base_spec": base_spec, "layer_sizes": cfg.layer_sizes},
    )


# ===========================================================================
# PDMP samplers
# ===========================================================================

def build_pdmp_sampler(name: str, target: TorchTarget, cfg: BNNConfig):
    """Instantiate one of the four PDMP samplers, all with PLI thinning."""
    common = dict(grad_target=target.grad_target, D=target.D, thinning="pli")
    spec = target.meta["spec"]

    if name == "zigzag":
        return AutomaticZigZagSampler(**common, t_max=T_MAX_ZZ, gamma=GAMMA_ZZ)

    if name == "sticky_zigzag":
        kappa = make_kappa_from_inclusion(
            spec=spec,
            prior_std_weight=cfg.prior_std_weight,
            prior_inclusion_weight=cfg.prior_inclusion_weight,
            fan_in_scaling=cfg.fan_in_scaling,
        )
        can_freeze = torch.repeat_interleave(
            torch.tensor(spec.can_freeze, dtype=torch.bool),
            torch.tensor(spec.numels),
        )
        return StickyAutomaticZigZagSampler(
            **common, t_max=T_MAX_ZZ, gamma=GAMMA_ZZ, kappa=kappa, 
            can_freeze=can_freeze
        )

    if name == "boomerang":
        s = AutomaticBoomerangSampler(**common, refresh_rate=0.1)
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
        info = tune_refresh_rate(s, n_pilot=200)
        print(f"      tuned refresh_rate: "
              f"{info['lambda_r_old']:.3f} -> {info['lambda_r_new']:.3f}")
        return s

    if name == "sticky_boomerang":
        kappa = make_kappa_from_inclusion(
            spec=spec,
            prior_std_weight=cfg.prior_std_weight,
            prior_inclusion_weight=cfg.prior_inclusion_weight,
            fan_in_scaling=cfg.fan_in_scaling,
        )
        can_freeze = torch.repeat_interleave(
            torch.tensor(spec.can_freeze, dtype=torch.bool),
            torch.tensor(spec.numels),
        )
        s = StickyAutomaticBoomerangSampler(
            **common, refresh_rate=1.0, kappa=kappa,
            can_freeze=can_freeze
        )
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
        info = tune_refresh_rate(s, n_pilot=200)
        print(f"      tuned refresh_rate: "
              f"{info['lambda_r_old']:.3f} -> {info['lambda_r_new']:.3f}")
        return s

    raise ValueError(f"Unknown PDMP sampler: {name}")


def resample_pdmp(name: str, result: dict, x_ref_np: np.ndarray) -> np.ndarray:
    pos = result["positions"].cpu().numpy()
    vel = result["velocities"].cpu().numpy()
    tim = result["times"].cpu().numpy()
    common = dict(N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC)

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
# NUTS via NumPyro. The flatten convention ([W0, b0, W1, b1, ...] row-major
# within each W) matches what ParamSpec.from_module(build_ffn_module(...))
# produces, so the resulting samples slot into the same downstream code.
# ===========================================================================

def run_nuts(data: dict[str, Any], cfg: BNNConfig, seed: int,
            learned_noise: bool = False, prior_sigma_scale: float = 1.0,
             ) -> tuple[np.ndarray, float]:
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
                f"W{i}",
                dist.Normal(jnp.zeros((n_out, n_in)), scale).to_event(2),
            )
            b = numpyro.sample(
                f"b{i}",
                dist.Normal(jnp.zeros(n_out), prior_std_bias).to_event(1),
            )
            h = h @ W.T + b
            if i < len(layer_sizes) - 2:
                if activation == "tanh":
                    h = jnp.tanh(h)
                elif activation == "relu":
                    h = jnp.maximum(0.0, h)
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
        if learned_noise:
            sigma = numpyro.sample("sigma", dist.HalfNormal(prior_sigma_scale))
        else:
            sigma = cfg.noise_std
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("y", dist.Normal(h.squeeze(-1), sigma), obs=y)

    kernel = NUTS(bnn, target_accept_prob=0.9)
    mcmc = MCMC(kernel,
                num_warmup=NUTS_WARMUP, num_samples=NUTS_DRAWS,
                num_chains=NUTS_CHAINS, progress_bar=True)

    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), X=X, y=y)
    elapsed = time.perf_counter() - t0

    posterior = mcmc.get_samples()
    flat = []
    for i in range(len(layer_sizes) - 1):
        W = np.asarray(posterior[f"W{i}"])  # [n, n_out, n_in]
        b = np.asarray(posterior[f"b{i}"])  # [n, n_out]
        flat.append(W.reshape(W.shape[0], -1))
        flat.append(b)
    if learned_noise:
        flat.append(np.asarray(posterior["sigma"])[:, None])
    return np.concatenate(flat, axis=1).astype(np.float64), elapsed

def run_nuts_horseshoe(data: dict[str, Any], cfg: BNNConfig, seed: int,
                      learned_noise: bool = False, prior_sigma_scale: float = 1.0,
                       ) -> tuple[np.ndarray, float]:
    """NUTS with a horseshoe prior on weights, half-Cauchy on biases.

    Per-weight prior:  W_ij ~ N(0, (tau * lambda_ij)^2)
        tau         ~ HalfCauchy(prior_std_weight)   (global shrinkage, per layer)
        lambda_ij   ~ HalfCauchy(1)                  (local shrinkage, per weight)

    Biases keep the same Gaussian prior as the standard NUTS run for
    comparability — putting horseshoe on biases rarely helps and slows mixing.

    Output flatten convention matches run_nuts: [W0, b0, W1, b1, ...] with
    each W flattened row-major in (n_out, n_in). Auxiliary horseshoe latents
    (tau, lambda) are NOT included in the flat vector — only the weights and
    biases that the BNN target consumes.
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
            # Per-layer global shrinkage
            tau = numpyro.sample(f"tau{i}", dist.HalfCauchy(prior_std))
            # Per-weight local shrinkage
            lam = numpyro.sample(
                f"lam{i}",
                dist.HalfCauchy(jnp.ones((n_out, n_in))).to_event(2),
            )
            # Non-centred parameterisation: sample a unit Normal then scale.
            # Critical for horseshoe — the funnel between tau, lam, and W is
            # what makes the centred form mix terribly.
            W_raw = numpyro.sample(
                f"W{i}_raw",
                dist.Normal(jnp.zeros((n_out, n_in)), 1.0).to_event(2),
            )
            W = numpyro.deterministic(f"W{i}", tau * lam * W_raw)

            b = numpyro.sample(
                f"b{i}",
                dist.Normal(jnp.zeros(n_out), prior_std_bias).to_event(1),
            )

            h = h @ W.T + b
            if i < len(layer_sizes) - 2:
                if activation == "tanh":
                    h = jnp.tanh(h)
                elif activation == "relu":
                    h = jnp.maximum(0.0, h)
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
        if learned_noise:
            sigma = numpyro.sample("sigma", dist.HalfNormal(prior_sigma_scale))
        else:
            sigma = cfg.noise_std
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("y", dist.Normal(h.squeeze(-1), sigma), obs=y)

    kernel = NUTS(bnn, target_accept_prob=0.95)   # tighter step than vanilla
    mcmc = MCMC(kernel,
                num_warmup=NUTS_WARMUP, num_samples=NUTS_DRAWS,
                num_chains=NUTS_CHAINS, progress_bar=True)

    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), X=X, y=y)
    elapsed = time.perf_counter() - t0

    posterior = mcmc.get_samples()
    flat = []
    for i in range(len(layer_sizes) - 1):
        # Pull the deterministic W (= tau * lam * W_raw), not W_raw
        W = np.asarray(posterior[f"W{i}"])  # [n, n_out, n_in]
        b = np.asarray(posterior[f"b{i}"])  # [n, n_out]
        flat.append(W.reshape(W.shape[0], -1))
        flat.append(b)
    if learned_noise:
        flat.append(np.asarray(posterior["sigma"])[:, None])
    return np.concatenate(flat, axis=1).astype(np.float64), elapsed

# ===========================================================================
# Persistence
# ===========================================================================

def thin_to(samples: torch.Tensor, n_keep: int) -> torch.Tensor:
    n = samples.shape[0]
    if n <= n_keep:
        return samples
    idx = torch.linspace(0, n - 1, n_keep).round().long()
    return samples[idx]


def split_dir(out_dir: Path, dataset: str, split_id: int) -> Path:
    return out_dir / dataset / f"split_{split_id:02d}"


def save_run(out_path: Path, *, dataset: str, split_id: int, sampler: str,
             samples: torch.Tensor, target: TorchTarget | None, cfg: BNNConfig,
             y_std: float, elapsed_sec: float, n_events: int) -> None:
    """Persist a sampler run. Same payload schema for PDMP and NUTS."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "dataset":      dataset,
        "split_id":     split_id,
        "sampler":      sampler,
        "samples":      thin_to(samples, N_SAVE).cpu(),
        "x_ref":        target.x_ref.cpu() if target is not None else None,
        "layer_sizes":  cfg.layer_sizes,
        "activation":   cfg.activation,
        "noise_std":    cfg.noise_std,
        "y_std":        y_std,
        "n_events":     n_events,        # PDMP: skeleton events; NUTS: draws
        "elapsed_sec":  elapsed_sec,
    }, out_path)


# ===========================================================================
# Per-(dataset, split) loop
# ===========================================================================

def run_split(dataset_name: str, split_id: int, data: dict[str, Any],
              cfg: BNNConfig, out_dir: Path,
              samplers: tuple[str, ...], resume: bool,#) -> None:
              learned_noise: bool = False) -> None:
    suffix = "_ln" if learned_noise else ""
    print(f"\n--- {dataset_name.upper()} split {split_id:02d} | "
          f"layers={cfg.layer_sizes} | act={cfg.activation} | "
          f"noise={'learned' if learned_noise else f'fixed ({cfg.noise_std})'} | "
          f"seed={BASE_SEED + split_id} ---")

    needs_pdmp = any(s in PDMP_SAMPLER_NAMES for s in samplers)
    if needs_pdmp:
        target = (build_target_learned_noise(data, cfg)
                  if learned_noise else build_target(data, cfg))
    else:
        target = None

    if target is not None:
        print(f"  D = {target.D}")

    seed = BASE_SEED + split_id
    sd = split_dir(out_dir, dataset_name, split_id)

    for name in samplers:
        out_path = sd / f"{name}{suffix}.pt"   # <-- suffix here
        if resume and out_path.exists():
            print(f"  [{name}{suffix}] skipping — exists at {out_path}")
            continue

        print(f"  [{name}]")
        torch.manual_seed(seed); np.random.seed(seed)

        if name == "nuts":
            try:
                samples_np, elapsed = run_nuts(data, cfg, seed)
            except ImportError as e:
                print(f"      NUTS dependency missing ({e}); skipping.")
                continue
            samples = torch.tensor(samples_np)
            n_events = samples_np.shape[0]
        elif name == "nuts_horseshoe":
            try:
                samples_np, elapsed = run_nuts_horseshoe(data, cfg, seed)
            except ImportError as e:
                print(f"      NumPyro missing ({e}); skipping.")
                continue
            samples = torch.tensor(samples_np)
            n_events = samples_np.shape[0]
        else:
            sampler = build_pdmp_sampler(name, target, cfg)
            t0 = time.perf_counter()
            result = sampler.sample(N=N_SKELETON, diagnostics=True)
            samples = torch.tensor(
                resample_pdmp(name, result, target.x_ref.cpu().numpy())
            )
            elapsed = time.perf_counter() - t0
            print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s")
            n_events = N_SKELETON

        save_run(out_path,
                 dataset=dataset_name, split_id=split_id, sampler=name,
                 samples=samples, target=target, cfg=cfg,
                 y_std=data["y_std"], elapsed_sec=elapsed, n_events=n_events)
        print(f"      saved {samples.shape[0]} samples (thinned to {N_SAVE}) "
              f"-> {out_path}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=list(TOY_DATASETS))
    parser.add_argument("--samplers", nargs="+", default=None,
                        choices=list(ALL_SAMPLER_NAMES))
    parser.add_argument("--learned-noise", action="store_true",
                    help="Use the half-normal-on-sigma learned noise model "
                         "instead of fixed cfg.noise_std.")
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--include-nuts", action="store_true")
    parser.add_argument("--include-nuts-hs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--toy-dir", type=Path, default=TOY_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.samplers is None:
        samplers = list(PDMP_SAMPLER_NAMES) + (["nuts"] if args.include_nuts else []) + (["nuts_horseshoe"] if args.include_nuts_hs else [])
    else:
        samplers = list(args.samplers)

    toy_to_run = [d for d in args.datasets if d in TOY_DATASETS]

    print("\nLoading toy 1D datasets...")
    directory = Path(f"{args.out}")
    try:
        toys = {n: load_toy(n, args.toy_dir) for n in toy_to_run}
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Generate the toys first with: "
                f"python -m sazz.scripts.generate_toys --names {' '.join(toy_to_run)}")
        return
    print(f"\nRunning {toy_to_run} (single split each) | "
            f"samplers {samplers}")
    for ds in toy_to_run:
        data, cfg = toys[ds]
        run_split(ds, 0, data, cfg, directory, tuple(samplers),
                    resume=args.resume, learned_noise=args.learned_noise)
            


if __name__ == "__main__":
    main()
