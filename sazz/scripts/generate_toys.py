"""Generate toy 1D datasets for BNN benchmarking, persist to disk.

Each toy dataset is generated once with a fixed seed, preprocessed
(standardised inputs and targets), bundled with a `BNNConfig` describing
the architecture/prior/noise model that should be used for it, and
written to `datasets/toy_1d/<name>.pt`. Subsequent runs of `uci_bnn.py`
load these `.pt` files directly — no RNG involved on the runner side, so
results are reproducible regardless of unrelated code changes.

The on-disk schema for a toy `.pt` matches what `make_split` returns for
UCI datasets, plus:
  - "config":         BNNConfig instance used to build the BNN target
  - "y_test_clean":   noise-free test targets (standardised)
  - "y_test_clean_raw", "x_train_raw", "y_train_raw", "x_test_raw"
  - "noise_std_true", "x_mean", "x_std", "y_mean"

Datasets:
  hernandez   PBP / HL&A 2015 toy: y = x^3 + N(0, 9), 20 training points on [-4, 4]
  gap         y = sin(1.5x) + 0.3x with a gap in [-1.5, 1.5]
  sharp       tanh + sharp Gaussian bump near x=0.5
  multiscale  superposition of slow and fast sinusoids

Usage:
    # Generate everything (overwrites if --force is set)
    python -m sazz.scripts.generate_toys

    # Generate only one
    python -m sazz.scripts.generate_toys --names hernandez

    # Force re-generation
    python -m sazz.scripts.generate_toys --force

    # Custom output directory
    python -m sazz.scripts.generate_toys --out datasets/my_toys
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sazz.scripts.uci_bnn import BNNConfig

TOY_DIR = Path("datasets/toy_1d")
TOY_NAMES = ("hernandez", "gap", "sharp", "multiscale")
DEFAULT_SEED = 42


# ===========================================================================
# Standardisation
# ===========================================================================

def _standardise_1d(x_train, y_train, x_test, y_test_noisy,
                    y_test_clean, noise_std_true) -> dict[str, Any]:
    """Standardise a 1D dataset using training-set statistics."""
    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()

    x_train_s = (x_train - x_mean) / x_std
    x_test_s  = (x_test  - x_mean) / x_std
    y_train_s = (y_train - y_mean) / y_std
    y_test_s  = (y_test_noisy - y_mean) / y_std
    y_test_clean_s = (y_test_clean - y_mean) / y_std

    return {
        "X_train": torch.tensor(x_train_s.reshape(-1, 1), dtype=torch.float64),
        "y_train": torch.tensor(y_train_s, dtype=torch.float64),
        "X_test":  torch.tensor(x_test_s.reshape(-1, 1), dtype=torch.float64),
        "y_test":  torch.tensor(y_test_s, dtype=torch.float64),
        "y_test_clean":     torch.tensor(y_test_clean_s, dtype=torch.float64),
        "y_test_clean_raw": y_test_clean,
        "x_train_raw": x_train, "y_train_raw": y_train,
        "x_test_raw":  x_test,  "y_test_raw":  y_test_noisy,
        "x_mean":  float(x_mean), "x_std": float(x_std),
        "y_mean":  float(y_mean), "y_std":  float(y_std),
        "noise_std_true": float(noise_std_true),
        "n_train": int(x_train.shape[0]),
        "n_test":  int(x_test.shape[0]),
    }


# ===========================================================================
# Per-toy generators
# ===========================================================================

def _gen_hernandez(rng: np.random.Generator) -> dict[str, Any]:
    """PBP / HL&A 2015 toy: y = x^3 + N(0, 9), 20 train points on [-4, 4]."""
    true_f = lambda x: x ** 3
    noise = 3.0
    x_train = rng.uniform(-4.0, 4.0, size=20)
    y_train = true_f(x_train) + rng.normal(0, noise, size=20)
    x_test = np.linspace(-6.0, 6.0, 300)
    y_test_clean = true_f(x_test)
    y_test_noisy = y_test_clean + rng.normal(0, noise, size=300)
    data = _standardise_1d(x_train, y_train, x_test, y_test_noisy,
                           y_test_clean, noise)

    # HL&A 2015 toy uses [1, 100, 1] ReLU. Noise on standardised scale is
    # noise_true / y_std — which we know exactly now that we have the data.
    data["config"] = BNNConfig(
        layer_sizes=[1, 100, 1],
        activation="relu",
        noise_std=noise / data["y_std"],
        prior_std_weight=1.0,
        prior_std_bias=1.0,
        fan_in_scaling=True,
        adam_steps=3000,
        prior_inclusion_weight=[0.5, 0.5],
    )
    return data


def _gen_gap(rng: np.random.Generator) -> dict[str, Any]:
    """sin(1.5x) + 0.3x with a gap in [-1.5, 1.5]."""
    true_f = lambda x: np.sin(1.5 * x) + 0.3 * x
    noise = 0.1
    x_train = np.concatenate([
        rng.uniform(-3.0, -1.5, size=30),
        rng.uniform( 1.5,  3.0, size=30),
    ])
    y_train = true_f(x_train) + rng.normal(0, noise, size=x_train.shape)
    x_test = np.linspace(-4.0, 4.0, 300)
    y_test_clean = true_f(x_test)
    y_test_noisy = y_test_clean + rng.normal(0, noise, size=300)
    data = _standardise_1d(x_train, y_train, x_test, y_test_noisy,
                           y_test_clean, noise)
    data["config"] = BNNConfig(
        layer_sizes=[1, 100, 1],
        activation="tanh",
        noise_std=noise / data["y_std"],
        prior_std_weight=1.0,
        prior_std_bias=1.0,
        fan_in_scaling=True,
        adam_steps=3000,
        prior_inclusion_weight=[0.5, 0.5],
    )
    return data


def _gen_sharp(rng: np.random.Generator) -> dict[str, Any]:
    """tanh + sharp Gaussian bump."""
    true_f = lambda x: 0.3 * np.tanh(x) + 1.5 * np.exp(-((x - 0.5) ** 2) / 0.05)
    noise = 0.1
    x_train = np.concatenate([
        rng.uniform(-3.0, 3.0, size=40),
        rng.uniform( 0.0, 1.0, size=20),
    ])
    y_train = true_f(x_train) + rng.normal(0, noise, size=x_train.shape)
    x_test = np.linspace(-4.0, 4.0, 300)
    y_test_clean = true_f(x_test)
    y_test_noisy = y_test_clean + rng.normal(0, noise, size=300)
    data = _standardise_1d(x_train, y_train, x_test, y_test_noisy,
                           y_test_clean, noise)
    data["config"] = BNNConfig(
        layer_sizes=[1, 100, 1],
        activation="tanh",
        noise_std=noise / data["y_std"],
        prior_std_weight=1.0,
        prior_std_bias=1.0,
        fan_in_scaling=True,
        adam_steps=3000,
        prior_inclusion_weight=[0.5, 0.5],
    )
    return data


def _gen_multiscale(rng: np.random.Generator) -> dict[str, Any]:
    """sin(0.5x) + 0.3 sin(4x), 80 train points uniform on [-3, 3]."""
    true_f = lambda x: np.sin(0.5 * x) + 0.3 * np.sin(4.0 * x)
    noise = 0.1
    x_train = rng.uniform(-3.0, 3.0, size=80)
    y_train = true_f(x_train) + rng.normal(0, noise, size=80)
    x_test = np.linspace(-4.0, 4.0, 300)
    y_test_clean = true_f(x_test)
    y_test_noisy = y_test_clean + rng.normal(0, noise, size=300)
    data = _standardise_1d(x_train, y_train, x_test, y_test_noisy,
                           y_test_clean, noise)
    data["config"] = BNNConfig(
        layer_sizes=[1, 100, 1],
        activation="tanh",
        noise_std=noise / data["y_std"],
        prior_std_weight=1.0,
        prior_std_bias=1.0,
        fan_in_scaling=True,
        adam_steps=3000,
        prior_inclusion_weight=[0.5, 0.5],
    )
    return data


GENERATORS = {
    "hernandez":  _gen_hernandez,
    "gap":        _gen_gap,
    "sharp":      _gen_sharp,
    "multiscale": _gen_multiscale,
}


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--names", nargs="+", default=list(TOY_NAMES),
        choices=list(TOY_NAMES),
    )
    parser.add_argument(
        "--out", type=Path, default=TOY_DIR,
        help=f"Output directory. Default: {TOY_DIR}",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"RNG seed. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing toy .pt files.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for name in args.names:
        path = args.out / f"{name}.pt"
        if path.exists() and not args.force:
            print(f"  {name}: skipping (exists at {path}; pass --force to overwrite)")
            continue

        rng = np.random.default_rng(args.seed)
        data = GENERATORS[name](rng)
        torch.save(data, path)
        print(f"  {name}: saved -> {path}  "
              f"(n_train={data['n_train']}, n_test={data['n_test']}, "
              f"y_std={data['y_std']:.3f}, "
              f"noise_std_standardised={data['config'].noise_std:.4f})")


if __name__ == "__main__":
    main()
