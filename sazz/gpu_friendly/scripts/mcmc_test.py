"""
NUTS chain-behavior probe for the "small" UCI regression BNN -- for
eyeballing how the MCMC chains move, NOT for a benchmark table.

Two init modes:

  --init prior  (default): FIVE independent single-chain NUTS runs, each
    started from its OWN random draw from the prior -- no MAP reference
    anywhere. The chains explore from genuinely different points, which is
    what makes cross-chain disagreement / multimodality / label-switching
    visible. NOTE: a prior draw for a multi-hidden-layer net is far from
    EVERY mode, and warmup's adaptation can funnel most chains into one
    basin regardless of where they started -- so "chains agree" here is
    weak evidence, not proof of unimodality.

  --init flipped-map: a targeted probe of the tanh(-z) = -tanh(z) sign-flip
    symmetry. Runs a MAP (via uci_bnn_grid.build_target) once, then starts
    n_chains chains at sign-flipped copies of it: chain 0 at the raw MAP,
    each other chain with a different random subset of hidden layers fully
    sign-flipped (W[k] row-negated, b[k] negated, W[k+1] column-negated --
    predictions exactly invariant). If a flipped chain STAYS in its flipped
    configuration instead of migrating back to the raw-MAP signs, that mode
    is a real, non-communicating basin -- the decisive test prior-init
    can't reliably give. tanh activation only (relu has no such symmetry).

Unlike uci_bnn_grid.py's run_nuts (4 chains, all from the same MAP x_ref),
every mode here keeps chains SEPARATE and single-chain.

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
    python -m sazz.gpu_friendly.scripts.mcmc_test --init flipped-map --datasets boston --splits 0
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
    load_raw_datasets, make_split, build_target,
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


def _flat_slices(layer_sizes: list[int]) -> list[dict]:
    """Byte-offset map of the flat [W0,b0,W1,b1,...,log_sigma] vector: one
    dict per Linear layer with 'W'/'b' as (start, stop) index pairs and the
    (n_out, n_in) shape. The trailing single log_sigma coord is not listed."""
    slices = []
    offset = 0
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        w0, w1 = offset, offset + n_out * n_in
        b0, b1 = w1, w1 + n_out
        slices.append({"W": (w0, w1), "b": (b0, b1), "shape": (n_out, n_in)})
        offset = b1
    return slices


def _sign_flip_map(x_map: np.ndarray, cfg: BNNConfig, flip_layers: list[int]) -> np.ndarray:
    """Apply the tanh(-z) = -tanh(z) sign-flip symmetry to a flat MAP vector.

    For each HIDDEN layer k in flip_layers (0-indexed among Linear layers;
    the output layer -- the last one -- is never eligible, it has no tanh
    after it): negate W[k]'s rows and b[k] (flips every unit of layer k's
    pre-activation), and negate W[k+1]'s columns (undoes it downstream).
    The network output is exactly unchanged; only the weight-space
    coordinates move to the mirror mode. log_sigma is untouched.

    Requires cfg.activation == 'tanh' (relu/elu have no such symmetry)."""
    if cfg.activation != "tanh":
        raise ValueError(
            f"sign-flip symmetry needs tanh activation; cfg.activation={cfg.activation!r}"
        )
    n_linear = len(cfg.layer_sizes) - 1
    out = x_map.copy()
    slices = _flat_slices(cfg.layer_sizes)
    for k in flip_layers:
        if not (0 <= k < n_linear - 1):
            raise ValueError(
                f"flip layer {k} out of range -- hidden Linear layers are 0..{n_linear - 2} "
                f"(layer {n_linear - 1} is the output layer, no tanh)"
            )
        wk0, wk1 = slices[k]["W"]
        bk0, bk1 = slices[k]["b"]
        out[wk0:wk1] = -out[wk0:wk1]     # W[k] all entries (rows) negated
        out[bk0:bk1] = -out[bk0:bk1]     # b[k] negated
        # W[k+1] columns: reshape, negate every column, write back.
        w1_0, w1_1 = slices[k + 1]["W"]
        n_out_next, n_in_next = slices[k + 1]["shape"]
        Wnext = out[w1_0:w1_1].reshape(n_out_next, n_in_next).copy()
        Wnext[:, :] = -Wnext           # n_in_next == this layer's n_out, so all cols
        out[w1_0:w1_1] = Wnext.reshape(-1)
    return out


def _flipped_map_init_points(x_map: np.ndarray, cfg: BNNConfig, n_chains: int,
                              rng: np.random.Generator) -> tuple[np.ndarray, list[list[int]]]:
    """n_chains start points for --init flipped-map: chain 0 is the raw MAP,
    each later chain flips a random non-empty subset of the hidden layers.
    Returns (init_points [n_chains, D], per-chain flipped-layer lists)."""
    n_hidden = len(cfg.layer_sizes) - 2  # Linear layers minus the output layer
    if n_hidden < 1:
        raise ValueError(
            f"flipped-map needs >=1 hidden layer; layer_sizes={cfg.layer_sizes}"
        )
    hidden_idx = list(range(n_hidden))
    points, flips = [x_map.copy()], [[]]
    for _ in range(1, n_chains):
        # random non-empty subset of hidden layers
        mask = rng.integers(0, 2, size=n_hidden).astype(bool)
        if not mask.any():
            mask[rng.integers(0, n_hidden)] = True
        chosen = [hidden_idx[i] for i in range(n_hidden) if mask[i]]
        points.append(_sign_flip_map(x_map, cfg, chosen))
        flips.append(chosen)
    return np.stack(points), flips


def _unflatten_init(x0: np.ndarray, cfg: BNNConfig) -> dict:
    """Flat [W0,b0,...,log_sigma] -> NumPyro init_params dict for ONE chain
    (no leading chain axis -- single-chain runs). Inverse of run_nuts's
    flatten.

    NumPyro's model-based init_params are UNCONSTRAINED latent values
    (verified against NumPyro 0.20.1: passing v as init_params["sigma"]
    yields a first draw of exp(v)). sigma has a HalfNormal prior, so its
    unconstrained coordinate is log_sigma -- which is exactly what x0[offset]
    already stores (see _prior_init_point / x_ref convention). So pass it
    through directly; exponentiating here (the old bug) started every chain
    at sigma = exp(log_sigma_MAP) instead of sigma_MAP."""
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
    init["sigma"] = jnp.array(x0[offset])  # already log_sigma == the unconstrained coord
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
               collect_warmup: bool, init_mode: str = "prior") -> dict:
    """init_mode: 'prior' -- each chain from its own prior draw (default);
    'flipped-map' -- chain 0 at a MAP, each other chain at that MAP with a
    random subset of hidden layers sign-flipped (see module docstring /
    _flipped_map_init_points). Returns, additionally, 'flip_layers' (list
    per chain; empty for chain 0 / all prior-mode chains) and 'init_mode'."""
    import jax
    from numpyro.infer import MCMC, NUTS

    X_np = data["X_train"].cpu().numpy()
    y_np = data["y_train"].cpu().numpy()
    bnn, X, y = _build_bnn_model(cfg, X_np, y_np)

    # One RNG for the init points (reproducible from base_seed), separate
    # from the per-chain JAX PRNGKeys used for the sampler itself.
    init_rng = np.random.default_rng(base_seed)

    if init_mode == "flipped-map":
        # MAP once (build_target does Adam + Laplace; we only need x_ref).
        _bm, x_map, _Sig = build_target(data, cfg)
        x_map = x_map.detach().cpu().numpy().astype(np.float64)
        init_points, flip_layers = _flipped_map_init_points(x_map, cfg, n_chains, init_rng)
        print(f"      flipped-map: chain 0 = raw MAP; flips per chain = {flip_layers}")
    elif init_mode == "prior":
        init_points = np.stack([_prior_init_point(cfg, init_rng) for _ in range(n_chains)])
        flip_layers = [[] for _ in range(n_chains)]
    else:
        raise ValueError(f"init_mode must be 'prior' or 'flipped-map'; got {init_mode!r}")

    all_samples, all_warmup, all_init = [], [], []
    all_diverging, all_num_steps, all_accept = [], [], []

    t0 = time.perf_counter()
    for c in range(n_chains):
        x0 = init_points[c]
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
        "init_mode": init_mode,
        "flip_layers": flip_layers,                             # list[list[int]], per chain
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
        "init_mode": result["init_mode"],                 # "prior" | "flipped-map"
        "init_from_map": result["init_mode"] == "flipped-map",
        "flip_layers": result["flip_layers"],            # list[list[int]], per chain
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
                             "the chains move from their init through adaptation).")
    parser.add_argument("--init", choices=["prior", "flipped-map"], default="prior",
                        help="'prior' (default): each chain from its own prior draw. "
                             "'flipped-map': chain 0 at a MAP, each other chain at that "
                             "MAP with a random subset of hidden layers sign-flipped "
                             "(tanh symmetry) -- a targeted probe of whether the mirror "
                             "modes are real non-communicating basins.")
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

    # sigma_inv_scale is irrelevant here (NUTS only, no Boomerang reference);
    # pass HIDDEN_VARIANT just to satisfy configs_for's signature.
    cfgs = configs_for({n: X.shape[1] for n, (X, _) in raw.items()}, hidden,
                        hidden_variant=HIDDEN_VARIANT)

    tag = "MULTISTART NUTS (prior init, no MAP)" if args.init == "prior" \
        else "NUTS sign-flip probe (chain 0 = MAP, others = sign-flipped MAP)"
    print(f"\n{tag} | hidden_variant={HIDDEN_VARIANT} ({hidden}) | "
          f"datasets={datasets_to_run} | splits={args.splits} | "
          f"{args.n_chains} chains x {args.n_draws} draws (+{args.n_warmup} warmup) | "
          f"target_accept={TARGET_ACCEPT}")

    for ds in datasets_to_run:
        X, y = raw[ds]
        cfg = cfgs[ds]
        for split_id in args.splits:
            fname = "nuts_chains.pt" if args.init == "prior" else "nuts_chains_flipped_map.pt"
            out_path = args.out / ds / f"split_{split_id:02d}" / fname
            if args.resume and out_path.exists():
                print(f"\n--- {ds.upper()} split {split_id:02d} -- skipping, exists at {out_path}")
                continue

            print(f"\n--- {ds.upper()} split {split_id:02d} | layers={cfg.layer_sizes} | "
                  f"act={cfg.activation} | noise=learned (HalfNormal scale={cfg.prior_sigma_scale:.4f}) ---")

            data = make_split(X, y, seed=BASE_SEED + split_id, dtype=DTYPE, device="cpu")
            result = run_chains(
                data, cfg, base_seed=BASE_SEED + split_id,
                n_chains=args.n_chains, n_draws=args.n_draws, n_warmup=args.n_warmup,
                collect_warmup=args.collect_warmup, init_mode=args.init,
            )
            save_run(
                out_path, dataset=ds, split_id=split_id, cfg=cfg, y_std=data["y_std"],
                base_seed=BASE_SEED + split_id, n_chains=args.n_chains,
                n_draws=args.n_draws, n_warmup=args.n_warmup, result=result,
            )
            print(f"      saved {args.n_chains} chains x {args.n_draws} draws -> {out_path}")


if __name__ == "__main__":
    main()
