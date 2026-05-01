from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from sazz.samplers.AutomaticBoomerangSampler import AutomaticBoomerangSampler
from sazz.samplers.StickyAutomaticBoomerangSampler import (
    StickyAutomaticBoomerangSampler,
)
from sazz.samplers.AutomaticZigZagSampler_torch import AutomaticZigZagSampler
from sazz.samplers.StickyAutomaticZigZagSampler_torch import (
    StickyAutomaticZigZagSampler,
)
from sazz.models.bnn_torch import make_bnn_regression, predict_regression
from sazz.utils.bnn_utils import make_kappa_from_inclusion
from sazz.utils.warmup import tune_refresh_rate
from sazz.utils.sampling import (
    resample_boomerang_path,
    resample_boomerang_path_sticky,
    resample_zigzag_path,
    resample_zigzag_path_sticky,
)


# ===========================================================================
# Config
# ===========================================================================

torch.set_default_dtype(torch.float64)

N_SKELETON   = 100_000   # skeleton events drawn per sampler
N_RESAMPLE   = 50_000    # uniform-in-time resamples used for metrics
N_THIN_SAVE  = 4_000     # thinned samples persisted to disk for reproducibility
BURNIN_FRAC  = 0.2
SEED         = 42

# ZigZag dynamics knobs (Boomerang has its own internal dynamics)
T_MAX_ZZ     = 0.1
GAMMA_ZZ     = 0.01

# A common BNN architecture across datasets keeps the comparison clean. Boston
# is 13-dim input; naval/energy override input dim from data.
HIDDEN       = [50]

OUT_DIR      = Path("results/uci_bnn")


# ===========================================================================
# Dataset loading
# ===========================================================================

def load_datasets() -> dict[str, dict[str, Any]]:
    """Load Boston / Naval / Energy with HL&A-style preprocessing.

    Returns a dict keyed by name with torch tensors for X/y train/test plus
    the y standard deviation for back-transforming RMSE to original units.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    boston_raw = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
    X_boston = boston_raw.data.values.astype(float)
    y_boston = boston_raw.target.values.astype(float)

    df_naval = pd.read_csv(
        "benchmarks_august/datasets/naval_data.txt", sep=r"\s+", header=None
    )
    X_naval = df_naval.iloc[:, :16].values.astype(float)
    y_naval = df_naval.iloc[:, 16].values.astype(float)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df_naval), 1000, replace=False)
    X_naval, y_naval = X_naval[idx], y_naval[idx]

    df_energy = pd.read_excel("benchmarks_august/datasets/energy_data.xlsx")
    X_energy = df_energy.iloc[:, :8].values.astype(float)
    y_energy = df_energy.iloc[:, 8].values.astype(float)

    raw = [
        ("boston", X_boston, y_boston),
        ("naval",  X_naval,  y_naval),
        ("energy", X_energy, y_energy),
    ]

    out: dict[str, dict[str, Any]] = {}
    for name, X, y in raw:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.1, random_state=SEED
        )
        x_mean, x_std = X_tr.mean(axis=0), X_tr.std(axis=0)
        x_std[x_std == 0] = 1.0  # naval has constant columns
        y_mean, y_std = y_tr.mean(), y_tr.std()

        X_tr = (X_tr - x_mean) / x_std
        X_te = (X_te - x_mean) / x_std
        y_tr = (y_tr - y_mean) / y_std
        y_te = (y_te - y_mean) / y_std

        out[name] = {
            "X_train": torch.tensor(X_tr),
            "y_train": torch.tensor(y_tr),
            "X_test":  torch.tensor(X_te),
            "y_test":  torch.tensor(y_te),
            "y_std":   float(y_std),
        }
        print(f"  {name:8s}  N={X.shape[0]:>6d}  D={X.shape[1]:>2d}  "
              f"train={X_tr.shape[0]}  test={X_te.shape[0]}")
    return out


# ===========================================================================
# Per-dataset BNN target config
# ===========================================================================

@dataclass
class BNNConfig:
    """Per-dataset target hyperparameters."""
    layer_sizes: list[int]
    activation: str = "tanh"
    noise_std: float = 0.1
    prior_std_weight: float = 1.0
    prior_std_bias: float = 1.0
    fan_in_scaling: bool = True
    adam_steps: int = 5000
    # Sticky inclusion priors per layer (length = len(layer_sizes) - 1)
    prior_inclusion_weight: list[float] = field(default_factory=list)


def configs_for(datasets: dict[str, dict[str, Any]]) -> dict[str, BNNConfig]:
    """Per-dataset BNN configs sharing the same hidden architecture."""
    return {
        "boston": BNNConfig(
            layer_sizes=[datasets["boston"]["X_train"].shape[1], *HIDDEN, 1],
            activation="relu",
            prior_std_weight=1.0,
            prior_std_bias=1.0,
            noise_std=0.3,
            prior_inclusion_weight=[0.7, 0.7],
        ),
        "naval": BNNConfig(
            layer_sizes=[datasets["naval"]["X_train"].shape[1], *HIDDEN, 1],
            activation="relu",
            prior_std_weight=1.0,
            prior_std_bias=1.0,
            noise_std=0.01,
            prior_inclusion_weight=[0.7, 0.7],
        ),
        "energy": BNNConfig(
            layer_sizes=[datasets["energy"]["X_train"].shape[1], *HIDDEN, 1],
            activation="relu",
            prior_std_weight=1.0,
            prior_std_bias=1.0,
            noise_std=0.3,
            prior_inclusion_weight=[0.7, 0.7],
        ),
    }


def build_target(data: dict[str, Any], cfg: BNNConfig):
    """Build a Laplace-referenced regression BNN target."""
    return make_bnn_regression(
        data["X_train"], data["y_train"],
        layer_sizes=cfg.layer_sizes,
        activation=cfg.activation,
        prior_std_weight=torch.sqrt(torch.tensor(cfg.prior_std_weight ** 2)),
        prior_std_bias=cfg.prior_std_bias,
        fan_in_scaling=cfg.fan_in_scaling,
        noise_std=cfg.noise_std,
        covariance_reference="laplace_diag",
        adam_steps=cfg.adam_steps,
    )


# ===========================================================================
# Sampler factory
# ===========================================================================

# Names used for files / metrics rows. The order here is the order we'll run.
SAMPLER_NAMES = ("zigzag", "sticky_zigzag", "boomerang", "sticky_boomerang")


def build_sampler(name: str, target, cfg: BNNConfig):
    """Instantiate one of the four PDMP samplers, all with PLI thinning."""
    common = dict(grad_target=target.grad_target, D=target.D, thinning="pli")

    if name == "zigzag":
        s = AutomaticZigZagSampler(**common, t_max=T_MAX_ZZ, gamma=GAMMA_ZZ)
        # ZigZag does not need a reference covariance; nothing to preprocess.

    elif name == "sticky_zigzag":
        kappa = make_kappa_from_inclusion(
            layer_sizes=cfg.layer_sizes,
            prior_std_weight=cfg.prior_std_weight,
            prior_inclusion_weight=cfg.prior_inclusion_weight,
            fan_in_scaling=cfg.fan_in_scaling,
        )
        s = StickyAutomaticZigZagSampler(
            **common, t_max=T_MAX_ZZ, gamma=GAMMA_ZZ, kappa=kappa
        )

    elif name == "boomerang":
        s = AutomaticBoomerangSampler(**common, refresh_rate=0.1)
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
        info = tune_refresh_rate(s, n_pilot=200)
        print(f"      tuned refresh_rate: "
              f"{info['lambda_r_old']:.3f} -> {info['lambda_r_new']:.3f}")

    elif name == "sticky_boomerang":
        kappa = make_kappa_from_inclusion(
            layer_sizes=cfg.layer_sizes,
            prior_std_weight=cfg.prior_std_weight,
            prior_inclusion_weight=cfg.prior_inclusion_weight,
            fan_in_scaling=cfg.fan_in_scaling,
        )
        s = StickyAutomaticBoomerangSampler(
            **common, refresh_rate=1.0, kappa=kappa
        )
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
        info = tune_refresh_rate(s, n_pilot=200)
        print(f"      tuned refresh_rate: "
              f"{info['lambda_r_old']:.3f} -> {info['lambda_r_new']:.3f}")

    else:
        raise ValueError(f"Unknown sampler name: {name}")

    return s


def resample_for(name: str, result: dict, x_ref_np: np.ndarray) -> np.ndarray:
    """Pick the right path-resampler for each sampler family."""
    pos = result["positions"].cpu().numpy()
    vel = result["velocities"].cpu().numpy()
    tim = result["times"].cpu().numpy()

    if name == "zigzag":
        return resample_zigzag_path(
            pos, vel, tim, N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC
        )
    if name == "sticky_zigzag":
        return resample_zigzag_path_sticky(
            pos, vel, tim, N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC
        )
    if name == "boomerang":
        return resample_boomerang_path(
            pos, vel, tim, x_ref_np,
            N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
        )
    if name == "sticky_boomerang":
        return resample_boomerang_path_sticky(
            pos, vel, tim, x_ref_np,
            N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
        )
    raise ValueError(name)


# ===========================================================================
# Metrics
# ===========================================================================

def gaussian_nll(y_true: torch.Tensor, mean: torch.Tensor,
                 pred_std: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Mean predictive log-likelihood under N(mean, pred_std^2 + noise_std^2)."""
    total_var = pred_std ** 2 + noise_std ** 2
    total_std = total_var.sqrt()
    return (
        -0.5 * ((y_true - mean) / total_std) ** 2
        - total_std.log()
        - 0.5 * math.log(2 * math.pi)
    ).mean()


def ess_per_coord(samples: torch.Tensor, max_lag: int | None = None) -> torch.Tensor:
    """Initial-positive-sequence ESS per coordinate.

    Geyer's IPS truncation: sum pairs of autocovariances and stop at the
    first negative pair. Returns a tensor of shape [D].
    """
    x = samples - samples.mean(0, keepdim=True)
    n, d = x.shape
    var = (x ** 2).mean(0)               # [D]
    if max_lag is None:
        max_lag = min(n - 1, 1000)

    # rho[0] == 1 by construction; we accumulate sum starting from 1.
    rho_sum = torch.zeros(d, dtype=samples.dtype)
    # Pairs (1,2), (3,4), ... per Geyer
    prev_pair = torch.full((d,), float("inf"), dtype=samples.dtype)
    active = torch.ones(d, dtype=torch.bool)

    k = 1
    while k + 1 <= max_lag:
        # Lag-k autocovariance, normalized
        c_k   = (x[:n - k]     * x[k:]).mean(0)     / var.clamp(min=1e-30)
        c_kp1 = (x[:n - k - 1] * x[k + 1:]).mean(0) / var.clamp(min=1e-30)
        pair = c_k + c_kp1

        # Once a coord's pair goes <= 0 or is non-decreasing, freeze it.
        kill = active & ((pair <= 0) | (pair >= prev_pair))
        active = active & ~kill
        rho_sum = rho_sum + torch.where(active, pair, torch.zeros_like(pair))
        prev_pair = torch.where(active, pair, prev_pair)
        if not active.any():
            break
        k += 2

    tau = 1.0 + 2.0 * rho_sum            # integrated autocorrelation time
    return n / tau.clamp(min=1.0)


def evaluate(samples_t: torch.Tensor, target, data: dict[str, Any],
             noise_std: float) -> dict[str, float]:
    """Compute test metrics + chain ESS summaries."""
    X_test = data["X_test"]
    y_test = data["y_test"]

    mean_pred, std_pred = predict_regression(samples_t, X_test, target)

    rmse_std = ((mean_pred - y_test) ** 2).mean().sqrt()
    nll      = gaussian_nll(y_test, mean_pred, std_pred, noise_std)
    pred_std_mean = std_pred.mean()

    ess = ess_per_coord(samples_t)
    return {
        "rmse_std":      float(rmse_std),
        "rmse_orig":     float(rmse_std) * data["y_std"],
        "nll":           float(nll),
        "pred_std_mean": float(pred_std_mean),
        "ess_min":       float(ess.min()),
        "ess_median":    float(ess.median()),
        "ess_mean":      float(ess.mean()),
    }


# ===========================================================================
# Persistence
# ===========================================================================

def thin_to(samples_t: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Evenly-spaced thin to at most n_keep samples."""
    n = samples_t.shape[0]
    if n <= n_keep:
        return samples_t
    idx = torch.linspace(0, n - 1, n_keep).round().long()
    return samples_t[idx]


def save_run(out_dir: Path, dataset: str, sampler: str,
             samples_t: torch.Tensor, target, cfg: BNNConfig,
             metrics: dict[str, float], n_skeleton_events: int,
             elapsed_sec: float) -> Path:
    """Persist thinned samples + everything needed to re-evaluate later."""
    path = out_dir / f"{dataset}_{sampler}.pt"
    payload = {
        "dataset":          dataset,
        "sampler":          sampler,
        "samples":          thin_to(samples_t, N_THIN_SAVE).cpu(),
        "x_ref":            target.x_ref.cpu(),
        "layer_sizes":      cfg.layer_sizes,
        "activation":       cfg.activation,
        "noise_std":        cfg.noise_std,
        "y_std":            metrics.get("_y_std"),  # set by caller
        "n_skeleton":       n_skeleton_events,
        "n_resample":       N_RESAMPLE,
        "burnin_frac":      BURNIN_FRAC,
        "elapsed_sec":      elapsed_sec,
        "metrics":          {k: v for k, v in metrics.items()
                              if not k.startswith("_")},
    }
    torch.save(payload, path)
    return path


# ===========================================================================
# Main loop
# ===========================================================================

def run_one(dataset_name: str, data: dict[str, Any], cfg: BNNConfig,
            out_dir: Path, samplers_to_run: tuple[str, ...]) -> list[dict]:
    """Build the target once, run each sampler, evaluate, persist."""
    print(f"\n=== {dataset_name.upper()} | "
          f"layers={cfg.layer_sizes} | act={cfg.activation} | "
          f"noise_std={cfg.noise_std} ===")

    target = build_target(data, cfg)
    x_ref_np = target.x_ref.cpu().numpy()
    print(f"  D = {target.D}")

    # MAP baseline (just x_ref)
    map_sample = target.x_ref.unsqueeze(0)
    map_metrics = evaluate(map_sample, target, data, cfg.noise_std)
    print(f"  [MAP]              "
          f"rmse_std={map_metrics['rmse_std']:.4f}  nll={map_metrics['nll']:.4f}")

    rows: list[dict] = [{
        "dataset": dataset_name, "sampler": "map",
        **map_metrics,
        "n_skeleton": 0, "elapsed_sec": 0.0,
    }]

    for name in samplers_to_run:
        print(f"  [{name}]")
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        sampler = build_sampler(name, target, cfg)
        t0 = time.perf_counter()
        result = sampler.sample(N=N_SKELETON, diagnostics=True)
        elapsed = time.perf_counter() - t0
        print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s")

        samples_np = resample_for(name, result, x_ref_np)
        samples_t = torch.tensor(samples_np)
        metrics = evaluate(samples_t, target, data, cfg.noise_std)

        print(f"      rmse_std={metrics['rmse_std']:.4f}  "
              f"nll={metrics['nll']:.4f}  "
              f"pred_std={metrics['pred_std_mean']:.4f}  "
              f"ess_med={metrics['ess_median']:.1f}")

        # Persist
        metrics_for_save = {**metrics, "_y_std": data["y_std"]}
        path = save_run(out_dir, dataset_name, name,
                        samples_t, target, cfg,
                        metrics_for_save, N_SKELETON, elapsed)
        print(f"      saved -> {path}")

        rows.append({
            "dataset": dataset_name, "sampler": name,
            **metrics,
            "n_skeleton": N_SKELETON, "elapsed_sec": elapsed,
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["boston", "naval", "energy"],
        choices=["boston", "naval", "energy"],
    )
    parser.add_argument(
        "--samplers", nargs="+", default=list(SAMPLER_NAMES),
        choices=list(SAMPLER_NAMES),
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    datasets = load_datasets()
    cfgs = configs_for(datasets)

    all_rows: list[dict] = []
    for name in args.datasets:
        all_rows.extend(
            run_one(name, datasets[name], cfgs[name], args.out,
                    tuple(args.samplers))
        )

    df = pd.DataFrame(all_rows)
    csv_path = args.out / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote metrics -> {csv_path}")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
