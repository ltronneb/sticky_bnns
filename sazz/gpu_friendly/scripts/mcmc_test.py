"""
NUTS chain-behavior probe for the "small" UCI regression BNN -- for
eyeballing how the MCMC chains move, NOT for a benchmark table.

Unlike uci_bnn_grid.py's run_nuts (4 chains, all initialized from the same
MAP estimate x_ref), this runs FIVE independent single-chain NUTS runs,
each started from its OWN random draw from the prior -- no MAP reference
anywhere. The five chains therefore explore from five genuinely different
points, which is what makes cross-chain disagreement / multimodality /
label-switching visible.

Everything about the target (data loading, split construction, BNNConfig,
the NumPyro `bnn` model with a learned-sigma HalfNormal prior) is taken
from uci_bnn_grid.py so this is the identical posterior that script's NUTS
runner samples -- only the initialization and the chain bookkeeping differ.

Per (dataset, split) it writes one file:
    results/mcmc_test/<dataset>/split_<NN>/nuts_chains.pt
containing:
    "samples"        : float64 [n_chains, n_draws, D]  -- chains kept SEPARATE
                       (NOT flattened), flatten convention [W0,b0,W1,b1,...,log_sigma]
                       matching uci_bnn_grid.py's run_nuts output
    "init_points"    : float64 [n_chains, D]  -- the prior draw each chain started from
    "warmup_samples" : float64 [n_chains, n_warmup, D] or None (--collect-warmup)
    "diverging"      : bool [n_chains, n_draws]  -- per-draw divergence flag
    "num_steps"      : int  [n_chains, n_draws]  -- leapfrog steps / draw
    "accept_prob"    : float64 [n_chains, n_draws]
    plus layer_sizes / activation / prior_* / y_std / seeds / elapsed provenance.

Usage:
    python -m sazz.gpu_friendly.scripts.mcmc_test
    python -m sazz.gpu_friendly.scripts.mcmc_test --datasets boston energy --splits 0
    python -m sazz.gpu_friendly.scripts.mcmc_test --n-draws 2000 --n-warmup 1000 --collect-warmup
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sazz.gpu_friendly.scripts.uci_bnn_grid import (
    DTYPE, BASE_SEED,
    UCI_DATASETS,
    BNNConfig, configs_for,
    load_raw_datasets, make_split,
)

# "small" == Izmailov et al.'s 1x50 shape; this probe is deliberately fixed
# to it (the whole point is a network small enough to inspect chain-by-chain).
HIDDEN_VARIANTS = {
    "small":       [50],
    "medium":      [50, 50],
    "deep_narrow": [50, 50, 50],
    "deep_wide":   [256, 128, 64],
}
HIDDEN_VARIANT = "medium"

N_CHAINS = 5
N_DRAWS = 3_000
N_WARMUP = 1_000
TARGET_ACCEPT = 0.9

OUT_DIR = Path("results/mcmc_test")


def _prior_init_point(cfg: BNNConfig, rng: np.random.Generator) -> np.ndarray:
    """One draw from the model's own prior, laid out in the flat
    [W0, b0, W1, b1, ..., log_sigma] order run_nuts uses. Weights ~
    Normal(0, prior_std / sqrt(fan_in)), biases ~ Normal(0, prior_std_bias),
    sigma ~ HalfNormal(prior_sigma_scale) stored as log_sigma. This is a
    genuine prior sample, so the five chains start spread across the prior
    the same way independent NumPyro init_strategy=init_to_sample draws
    would -- just made explicit here so we can save exactly where each
    chain began."""
    layer_sizes = cfg.layer_sizes
    parts = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        scale = cfg.prior_std_weight / np.sqrt(n_in if cfg.fan_in_scaling else 1.0)
        parts.append(rng.normal(0.0, scale, size=n_out * n_in))
        parts.append(rng.normal(0.0, cfg.prior_std_bias, size=n_out))
    sigma = np.abs(rng.normal(0.0, cfg.prior_sigma_scale))
    parts.append(np.array([np.log(max(sigma, 1e-6))]))
    return np.concatenate(parts).astype(np.float64)


def _unflatten_init(x0: np.ndarray, cfg: BNNConfig) -> dict:
    """Flat [W0,b0,...,log_sigma] -> NumPyro init_params dict for ONE chain
    (no leading chain axis -- single-chain runs). Inverse of run_nuts's
    flatten. sigma is stored as log_sigma in x0 (matches x_ref convention);
    NumPyro samples sigma on the positive reals, so exp() it back."""
    import jax.numpy as jnp

    layer_sizes = cfg.layer_sizes
    init = {}
    offset = 0
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w_size = n_out * n_in
        init[f"W{i}"] = jnp.array(x0[offset:offset + w_size].reshape(n_out, n_in))
        offset += w_size
        init[f"b{i}"] = jnp.array(x0[offset:offset + n_out])
        offset += n_out
    init["sigma"] = jnp.exp(jnp.array(x0[offset]))
    return init


def _flatten_posterior(posterior: dict, layer_sizes: list[int]) -> np.ndarray:
    """[W0,b0,W1,b1,...,log_sigma], same as run_nuts. posterior arrays here
    are [n_draws, ...] (single chain)."""
    flat = []
    for i in range(len(layer_sizes) - 1):
        W = np.asarray(posterior[f"W{i}"])  # [n, n_out, n_in]
        b = np.asarray(posterior[f"b{i}"])  # [n, n_out]
        flat.append(W.reshape(W.shape[0], -1))
        flat.append(b)
    flat.append(np.log(np.asarray(posterior["sigma"]))[:, None])
    return np.concatenate(flat, axis=1).astype(np.float64)


def _build_bnn_model(cfg: BNNConfig, X_np: np.ndarray, y_np: np.ndarray):
    """The identical NumPyro model uci_bnn_grid.py::run_nuts builds -- fan-in
    scaled Normal weight prior, Normal bias prior, learned sigma with a
    HalfNormal(prior_sigma_scale) prior. Kept in lockstep with that file."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    layer_sizes, activation = cfg.layer_sizes, cfg.activation
    prior_std, prior_std_bias = cfg.prior_std_weight, cfg.prior_std_bias
    X = jnp.asarray(X_np)
    y = jnp.asarray(y_np)

    def bnn(X, y=None):
        h = X
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            scale = prior_std / jnp.sqrt(n_in if cfg.fan_in_scaling else 1)
            W = numpyro.sample(f"W{i}", dist.Normal(jnp.zeros((n_out, n_in)), scale).to_event(2))
            b = numpyro.sample(f"b{i}", dist.Normal(jnp.zeros(n_out), prior_std_bias).to_event(1))
            h = h @ W.T + b
            if i < len(layer_sizes) - 2:
                if activation == "tanh":
                    h = jnp.tanh(h)
                elif activation == "relu":
                    h = jnp.maximum(0.0, h)
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
        sigma = numpyro.sample("sigma", dist.HalfNormal(cfg.prior_sigma_scale))
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("y", dist.Normal(h.squeeze(-1), sigma), obs=y)

    return bnn, X, y


def run_chains(data: dict[str, Any], cfg: BNNConfig, base_seed: int,
               n_chains: int, n_draws: int, n_warmup: int,
               collect_warmup: bool) -> dict:
    import jax
    from numpyro.infer import MCMC, NUTS

    X_np = data["X_train"].cpu().numpy()
    y_np = data["y_train"].cpu().numpy()
    bnn, X, y = _build_bnn_model(cfg, X_np, y_np)

    # One RNG for the init points (reproducible from base_seed), separate
    # from the per-chain JAX PRNGKeys used for the sampler itself.
    init_rng = np.random.default_rng(base_seed)

    all_samples, all_warmup, all_init = [], [], []
    all_diverging, all_num_steps, all_accept = [], [], []

    t0 = time.perf_counter()
    for c in range(n_chains):
        x0 = _prior_init_point(cfg, init_rng)
        all_init.append(x0)
        init_params = _unflatten_init(x0, cfg)

        # Distinct seed per chain -- the chains must be independent runs,
        # not the same trajectory relabeled.
        chain_seed = base_seed + 1 + c
        kernel = NUTS(bnn, target_accept_prob=TARGET_ACCEPT, adapt_mass_matrix=True)
        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_draws,
                    num_chains=1, progress_bar=True)

        extra = ("num_steps", "diverging", "accept_prob")
        if collect_warmup:
            # warmup()-then-run() split so warmup draws are retrievable;
            # run() takes post_warmup_state.rng_key (NOT a fresh
            # PRNGKey(chain_seed)) so it's a faithful decomposition of a
            # single run() -- same reasoning as uci_bnn_grid.py::run_nuts.
            mcmc.warmup(jax.random.PRNGKey(chain_seed), X=X, y=y, init_params=init_params,
                        extra_fields=extra, collect_warmup=True)
            warmup_post = mcmc.get_samples()
            all_warmup.append(_flatten_posterior(warmup_post, cfg.layer_sizes))
            mcmc.run(mcmc.post_warmup_state.rng_key, X=X, y=y, init_params=init_params,
                     extra_fields=extra)
        else:
            mcmc.run(jax.random.PRNGKey(chain_seed), X=X, y=y, init_params=init_params,
                     extra_fields=extra)

        posterior = mcmc.get_samples()
        fields = mcmc.get_extra_fields()
        all_samples.append(_flatten_posterior(posterior, cfg.layer_sizes))
        all_diverging.append(np.asarray(fields["diverging"]).astype(bool))
        all_num_steps.append(np.asarray(fields["num_steps"]).astype(np.int64))
        all_accept.append(np.asarray(fields["accept_prob"]).astype(np.float64))
        print(f"      chain {c}: {int(all_diverging[-1].sum())} divergences, "
              f"mean accept={all_accept[-1].mean():.3f}, "
              f"mean leapfrog={all_num_steps[-1].mean():.1f}")

    elapsed = time.perf_counter() - t0

    return {
        "samples": np.stack(all_samples),                       # [n_chains, n_draws, D]
        "init_points": np.stack(all_init),                      # [n_chains, D]
        "warmup_samples": np.stack(all_warmup) if collect_warmup else None,
        "diverging": np.stack(all_diverging),                   # [n_chains, n_draws]
        "num_steps": np.stack(all_num_steps),                   # [n_chains, n_draws]
        "accept_prob": np.stack(all_accept),                    # [n_chains, n_draws]
        "elapsed_sec": elapsed,
    }


def save_run(out_path: Path, *, dataset: str, split_id: int, cfg: BNNConfig,
             y_std: float, base_seed: int, n_chains: int, n_draws: int,
             n_warmup: int, result: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "dataset": dataset,
        "split_id": split_id,
        "sampler": "nuts_multistart",
        "n_chains": n_chains,
        "n_draws": n_draws,
        "n_warmup": n_warmup,
        "base_seed": base_seed,
        "chain_seeds": [base_seed + 1 + c for c in range(n_chains)],
        "target_accept_prob": TARGET_ACCEPT,
        "init_from_map": False,
        "samples": torch.tensor(result["samples"]),
        "init_points": torch.tensor(result["init_points"]),
        "warmup_samples": (torch.tensor(result["warmup_samples"])
                           if result["warmup_samples"] is not None else None),
        "diverging": torch.tensor(result["diverging"]),
        "num_steps": torch.tensor(result["num_steps"]),
        "accept_prob": torch.tensor(result["accept_prob"]),
        "layer_sizes": cfg.layer_sizes,
        "activation": cfg.activation,
        "learned_noise": True,
        "prior_sigma_scale": cfg.prior_sigma_scale,
        "prior_std_weight": cfg.prior_std_weight,
        "prior_std_bias": cfg.prior_std_bias,
        "fan_in_scaling": cfg.fan_in_scaling,
        "y_std": y_std,
        "elapsed_sec": result["elapsed_sec"],
    }, out_path)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=list(UCI_DATASETS),
                        choices=list(UCI_DATASETS))
    parser.add_argument("--splits", nargs="+", type=int, default=[0])
    parser.add_argument("--n-chains", type=int, default=N_CHAINS)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--collect-warmup", action="store_true",
                        help="Also keep the warmup draws (lets the notebook watch "
                             "the chains move from their prior init through adaptation).")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--resume", action="store_true",
                        help="Skip a (dataset, split) whose nuts_chains.pt already exists.")
    args = parser.parse_args()

    hidden = HIDDEN_VARIANTS[HIDDEN_VARIANT]
    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading raw UCI datasets...")
    raw = load_raw_datasets(tuple(args.datasets))
    missing = [d for d in args.datasets if d not in raw]
    if missing:
        print(f"  (skipping {missing} -- data file(s) not found)")
    datasets_to_run = [d for d in args.datasets if d in raw]

    cfgs = configs_for({n: X.shape[1] for n, (X, _) in raw.items()}, hidden)

    print(f"\nMULTISTART NUTS (no MAP) | hidden_variant={HIDDEN_VARIANT} ({hidden}) | "
          f"datasets={datasets_to_run} | splits={args.splits} | "
          f"{args.n_chains} chains x {args.n_draws} draws (+{args.n_warmup} warmup) | "
          f"target_accept={TARGET_ACCEPT}")

    for ds in datasets_to_run:
        X, y = raw[ds]
        cfg = cfgs[ds]
        for split_id in args.splits:
            out_path = args.out / ds / f"split_{split_id:02d}" / "nuts_chains.pt"
            if args.resume and out_path.exists():
                print(f"\n--- {ds.upper()} split {split_id:02d} -- skipping, exists at {out_path}")
                continue

            print(f"\n--- {ds.upper()} split {split_id:02d} | layers={cfg.layer_sizes} | "
                  f"act={cfg.activation} | noise=learned (HalfNormal scale={cfg.prior_sigma_scale:.4f}) ---")

            data = make_split(X, y, seed=BASE_SEED + split_id, dtype=DTYPE, device="cpu")
            result = run_chains(
                data, cfg, base_seed=BASE_SEED + split_id,
                n_chains=args.n_chains, n_draws=args.n_draws, n_warmup=args.n_warmup,
                collect_warmup=args.collect_warmup,
            )
            save_run(
                out_path, dataset=ds, split_id=split_id, cfg=cfg, y_std=data["y_std"],
                base_seed=BASE_SEED + split_id, n_chains=args.n_chains,
                n_draws=args.n_draws, n_warmup=args.n_warmup, result=result,
            )
            print(f"      saved {args.n_chains} chains x {args.n_draws} draws -> {out_path}")


if __name__ == "__main__":
    main()
