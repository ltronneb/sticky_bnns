"""UCI BNN benchmarks — *v2* modular pipeline (ParamSpec + nn.Module + functional_call).

Drop-in replacement for `uci_bnn.py` that runs the same samplers on the same
datasets, but uses the modular v2 pipeline:

  * Likelihood is an nn.Module (FFN built from layer_sizes), evaluated via
    torch.func.functional_call instead of the layer_sizes + unflatten_params
    forward loop.
  * Prior precision and kappa-from-inclusion are built from a ParamSpec
    derived from the module rather than from layer_sizes directly.

Everything else — sampler classes, bounding code, path resampling, NUTS,
metrics — is identical to the v1 runner.

Output goes to results/uci_bnn_v2/ so v1 results are not overwritten.

Usage is the same as uci_bnn.py, with --out defaulting to the v2 folder:

    python -m sazz.scripts.module_bnn --datasets hernandez
    python -m sazz.scripts.module_bnn --datasets boston --splits 0
    python -m sazz.scripts.module_bnn --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from sazz.samplers.AutomaticBoomerangSampler import AutomaticBoomerangSampler
from sazz.samplers.StickyAutomaticBoomerangSampler import (
    StickyAutomaticBoomerangSampler,
)
from sazz.samplers.AutomaticZigZagSampler_torch import AutomaticZigZagSampler
from sazz.samplers.StickyAutomaticZigZagSampler_torch import (
    StickyAutomaticZigZagSampler,
)
from sazz.models.make_models import TorchTarget
from sazz.models.bnn_torch import ModuleGaussianLikelihood, ModuleGaussianPrior
from sazz.models.models_torch import BayesianModel
from sazz.utils.warmup import find_reference_bnn, tune_refresh_rate
from sazz.utils.sampling import (
    resample_boomerang_path,
    resample_boomerang_path_sticky,
    resample_zigzag_path,
    resample_zigzag_path_sticky,
)

# v2 utilities — the only difference from uci_bnn.py at the import level
from sazz.utils.bnn_modular_utils import (
    ParamSpec,
    build_prior_precision,
    make_kappa_from_inclusion,
    build_ffn_module,
)


# ===========================================================================
# Config — identical to uci_bnn.py except OUT_DIR
# ===========================================================================

torch.set_default_dtype(torch.float64)

N_SKELETON   = 100_000
N_RESAMPLE   = 50_000
N_THIN_SAVE  = 4_000
BURNIN_FRAC  = 0.2
BASE_SEED    = 42

T_MAX_ZZ     = 0.1
GAMMA_ZZ     = 0.01

NUTS_WARMUP  = 1_000
NUTS_DRAWS   = 2_000
NUTS_CHAINS  = 4

HIDDEN       = [50]

# Different output directory so v1 results are not overwritten.
OUT_DIR      = Path("results/uci_bnn_v2")
TOY_DIR      = Path("datasets/toy_1d")

PDMP_SAMPLER_NAMES = ("zigzag", "sticky_zigzag", "boomerang", "sticky_boomerang")
ALL_SAMPLER_NAMES  = PDMP_SAMPLER_NAMES + ("nuts",)

UCI_DATASETS = ("boston", "naval", "energy")
TOY_DATASETS = ("hernandez", "gap", "sharp", "multiscale")
ALL_DATASETS = UCI_DATASETS + TOY_DATASETS


# ===========================================================================
# Data loading — identical to uci_bnn.py
# ===========================================================================

def load_raw_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load Boston / Naval / Energy as (X, y) numpy arrays before splitting."""
    from sklearn.datasets import fetch_openml

    boston_raw = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
    X_boston = boston_raw.data.values.astype(float)
    y_boston = boston_raw.target.values.astype(float)

    df_naval = pd.read_csv("datasets/naval_data.txt", sep=r"\s+", header=None)
    X_naval = df_naval.iloc[:, :16].values.astype(float)
    y_naval = df_naval.iloc[:, 16].values.astype(float)

    df_energy = pd.read_excel("datasets/energy_data.xlsx")
    X_energy = df_energy.iloc[:, :8].values.astype(float)
    y_energy = df_energy.iloc[:, 8].values.astype(float)

    return {
        "boston": (X_boston, y_boston),
        "naval":  (X_naval,  y_naval),
        "energy": (X_energy, y_energy),
    }


def make_split(name: str, X: np.ndarray, y: np.ndarray, seed: int,
               test_frac: float = 0.1) -> dict[str, Any]:
    """Apply HL&A-style preprocessing for one (dataset, seed) split."""
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=seed
    )
    x_mean, x_std = X_tr.mean(axis=0), X_tr.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean, y_std = y_tr.mean(), y_tr.std()

    X_tr = (X_tr - x_mean) / x_std
    X_te = (X_te - x_mean) / x_std
    y_tr = (y_tr - y_mean) / y_std
    y_te = (y_te - y_mean) / y_std

    return {
        "X_train": torch.tensor(X_tr),
        "y_train": torch.tensor(y_tr),
        "X_test":  torch.tensor(X_te),
        "y_test":  torch.tensor(y_te),
        "y_std":   float(y_std),
        "n_train": X_tr.shape[0],
        "n_test":  X_te.shape[0],
    }


def load_toy(name: str, toy_dir: Path = None) -> tuple[dict[str, Any], "BNNConfig"]:
    """Load a pre-generated toy dataset and its inline BNNConfig."""
    toy_dir = toy_dir or TOY_DIR
    path = toy_dir / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Toy dataset '{name}' not found at {path}. "
            f"Run: python -m sazz.scripts.generate_toys --names {name}"
        )
    data = torch.load(path, weights_only=False)
    cfg = data.pop("config")
    return data, cfg


# ===========================================================================
# Per-dataset BNN target config — identical to uci_bnn.py
# ===========================================================================

@dataclass
class BNNConfig:
    layer_sizes: list[int]
    activation: str = "relu"
    noise_std: float = 0.3
    prior_std_weight: float = 1.0
    prior_std_bias: float = 1.0
    fan_in_scaling: bool = True
    adam_steps: int = 5000
    prior_inclusion_weight: list[float] = field(default_factory=list)


def configs_for(input_dims: dict[str, int]) -> dict[str, BNNConfig]:
    cfgs: dict[str, BNNConfig] = {}
    if "boston" in input_dims:
        cfgs["boston"] = BNNConfig(
            layer_sizes=[input_dims["boston"], *HIDDEN, 1],
            noise_std=0.3, prior_inclusion_weight=[0.7, 0.7],
        )
    if "naval" in input_dims:
        cfgs["naval"] = BNNConfig(
            layer_sizes=[input_dims["naval"], *HIDDEN, 1],
            noise_std=0.01, prior_inclusion_weight=[0.7, 0.7],
        )
    if "energy" in input_dims:
        cfgs["energy"] = BNNConfig(
            layer_sizes=[input_dims["energy"], *HIDDEN, 1],
            noise_std=0.3, prior_inclusion_weight=[0.7, 0.7],
        )
    return cfgs


# ===========================================================================
# v2 target builder — replaces make_bnn_regression
# ===========================================================================

def build_target(data: dict[str, Any], cfg: BNNConfig,
                 dtype=torch.float64, device="cpu"):
    """Build a v2 TorchTarget for a regression BNN.

    Equivalent to v1's `make_bnn_regression` but evaluates the forward pass
    via torch.func.functional_call on an nn.Module instead of unflatten_params.
    """
    X = data["X_train"].to(dtype=dtype, device=device)
    y = data["y_train"].to(dtype=dtype, device=device)

    module = build_ffn_module(cfg.layer_sizes, cfg.activation).to(dtype=dtype)
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


@torch.no_grad()
def predict_regression(samples: Tensor, X_test: Tensor, target):
    """v2 predictive: forward each sample through functional_call."""
    likelihood = target.meta["model"].likelihood
    X_test = X_test.to(dtype=likelihood.X.dtype, device=likelihood.X.device)
    preds = torch.stack([
        likelihood.predict(beta, X_test).squeeze(-1) for beta in samples
    ])
    return preds.mean(0), preds.std(0)


# ===========================================================================
# PDMP sampler factory — uses v2 make_kappa_from_inclusion (takes a ParamSpec)
# ===========================================================================

def build_pdmp_sampler(name: str, target, cfg: BNNConfig):
    """Same logic as uci_bnn.build_pdmp_sampler, but kappas are built from
    target.meta['spec'] instead of from cfg.layer_sizes."""
    common = dict(grad_target=target.grad_target, D=target.D, thinning="pli")
    spec = target.meta["spec"]

    if name == "zigzag":
        s = AutomaticZigZagSampler(**common, t_max=T_MAX_ZZ, gamma=GAMMA_ZZ)

    elif name == "sticky_zigzag":
        kappa = make_kappa_from_inclusion(
            spec=spec,
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
            spec=spec,
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
        raise ValueError(f"Unknown PDMP sampler: {name}")

    return s


def resample_pdmp(name: str, result: dict, x_ref_np: np.ndarray) -> np.ndarray:
    """Identical to uci_bnn.resample_pdmp."""
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
# NUTS — identical to uci_bnn.run_nuts. The NUTS flatten convention
# ([W0, b0, W1, b1, ...] row-major within each W) matches what
# ParamSpec.from_module(build_ffn_module(...)) produces, so no changes needed.
# ===========================================================================

def run_nuts(data: dict[str, Any], cfg: BNNConfig, seed: int,
             num_warmup: int = NUTS_WARMUP,
             num_samples: int = NUTS_DRAWS,
             num_chains: int = NUTS_CHAINS) -> tuple[np.ndarray, float]:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    layer_sizes = cfg.layer_sizes
    activation = cfg.activation
    prior_std = cfg.prior_std_weight
    prior_std_bias = cfg.prior_std_bias
    fan_in_scaling = cfg.fan_in_scaling
    noise_std = cfg.noise_std

    X = jnp.asarray(data["X_train"].cpu().numpy())
    y = jnp.asarray(data["y_train"].cpu().numpy())

    def bnn(X, y=None):
        h = X
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            eff_fan_in = n_in if fan_in_scaling else 1
            scale = prior_std / jnp.sqrt(eff_fan_in)
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
        mean = h.squeeze(-1)
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("y", dist.Normal(mean, noise_std), obs=y)

    kernel = NUTS(bnn, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=True)

    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), X=X, y=y)
    elapsed = time.perf_counter() - t0

    posterior = mcmc.get_samples()
    flat_parts = []
    for i in range(len(layer_sizes) - 1):
        W = np.asarray(posterior[f"W{i}"])  # [n_samples, n_out, n_in]
        b = np.asarray(posterior[f"b{i}"])  # [n_samples, n_out]
        flat_parts.append(W.reshape(W.shape[0], -1))
        flat_parts.append(b)

    samples = np.concatenate(flat_parts, axis=1).astype(np.float64)
    return samples, elapsed


# ===========================================================================
# Metrics — identical to uci_bnn.py
# ===========================================================================

def gaussian_log_lik(y_true: Tensor, mean: Tensor,
                     pred_std: Tensor, noise_std: float) -> Tensor:
    total_var = pred_std ** 2 + noise_std ** 2
    total_std = total_var.sqrt()
    return (
        -0.5 * ((y_true - mean) / total_std) ** 2
        - total_std.log()
        - 0.5 * math.log(2 * math.pi)
    ).mean()


def ess_per_coord(samples: Tensor, max_lag: int | None = None) -> Tensor:
    x = samples - samples.mean(0, keepdim=True)
    n, d = x.shape
    var = (x ** 2).mean(0)
    if max_lag is None:
        max_lag = min(n - 1, 1000)

    rho_sum = torch.zeros(d, dtype=samples.dtype)
    prev_pair = torch.full((d,), float("inf"), dtype=samples.dtype)
    active = torch.ones(d, dtype=torch.bool)

    k = 1
    while k + 1 <= max_lag:
        c_k   = (x[:n - k]     * x[k:]).mean(0)     / var.clamp(min=1e-30)
        c_kp1 = (x[:n - k - 1] * x[k + 1:]).mean(0) / var.clamp(min=1e-30)
        pair = c_k + c_kp1
        kill = active & ((pair <= 0) | (pair >= prev_pair))
        active = active & ~kill
        rho_sum = rho_sum + torch.where(active, pair, torch.zeros_like(pair))
        prev_pair = torch.where(active, pair, prev_pair)
        if not active.any():
            break
        k += 2

    tau = 1.0 + 2.0 * rho_sum
    return n / tau.clamp(min=1.0)


def evaluate(samples_t: Tensor, target, data: dict[str, Any],
             noise_std: float) -> dict[str, float]:
    mean_pred, std_pred = predict_regression(samples_t, data["X_test"], target)
    rmse_std = ((mean_pred - data["y_test"]) ** 2).mean().sqrt()
    log_lik  = gaussian_log_lik(data["y_test"], mean_pred, std_pred, noise_std)
    ess = ess_per_coord(samples_t)
    return {
        "rmse_std":      float(rmse_std),
        "rmse_orig":     float(rmse_std) * data["y_std"],
        "log_lik":       float(log_lik),
        "nll_orig":      -float(log_lik) + math.log(data["y_std"]),
        "pred_std_mean": float(std_pred.mean()),
        "ess_min":       float(ess.min()),
        "ess_median":    float(ess.median()),
        "ess_mean":      float(ess.mean()),
    }


# ===========================================================================
# Persistence — identical structure to uci_bnn.py, just lives under OUT_DIR
# ===========================================================================

def split_dir(out_dir: Path, dataset: str, split_id: int) -> Path:
    return out_dir / dataset / f"split_{split_id:02d}"


def thin_to(samples_t: Tensor, n_keep: int) -> Tensor:
    n = samples_t.shape[0]
    if n <= n_keep:
        return samples_t
    idx = torch.linspace(0, n - 1, n_keep).round().long()
    return samples_t[idx]


def save_run(out_dir: Path, dataset: str, split_id: int, sampler: str,
             samples_t: Tensor, target, cfg: BNNConfig,
             metrics: dict[str, float], n_skeleton_events: int,
             elapsed_sec: float, y_std: float) -> Path:
    sd = split_dir(out_dir, dataset, split_id)
    sd.mkdir(parents=True, exist_ok=True)
    path = sd / f"{sampler}.pt"
    payload = {
        "dataset":     dataset,
        "split_id":    split_id,
        "sampler":     sampler,
        "samples":     thin_to(samples_t, N_THIN_SAVE).cpu(),
        "x_ref":       (target.x_ref.cpu() if target is not None else None),
        "layer_sizes": cfg.layer_sizes,
        "activation":  cfg.activation,
        "noise_std":   cfg.noise_std,
        "y_std":       y_std,
        "n_skeleton":  n_skeleton_events,
        "n_resample":  N_RESAMPLE,
        "burnin_frac": BURNIN_FRAC,
        "elapsed_sec": elapsed_sec,
        "metrics":     metrics,
        "pipeline":    "v2",   # marker so v1/v2 payloads are distinguishable
    }
    torch.save(payload, path)
    return path


# ===========================================================================
# Per-(dataset, split) loop — same shape as uci_bnn.run_split
# ===========================================================================

def run_split(dataset_name: str, split_id: int, data: dict[str, Any],
              cfg: BNNConfig, out_dir: Path,
              samplers_to_run: tuple[str, ...],
              resume: bool) -> list[dict]:
    print(f"\n--- {dataset_name.upper()} split {split_id:02d} | "
          f"layers={cfg.layer_sizes} | act={cfg.activation} | "
          f"noise_std={cfg.noise_std} | seed={BASE_SEED + split_id} | "
          f"pipeline=v2 ---")

    needs_pdmp = any(s in PDMP_SAMPLER_NAMES for s in samplers_to_run)
    target = build_target(data, cfg) if needs_pdmp else None
    x_ref_np = target.x_ref.cpu().numpy() if target is not None else None

    if target is not None:
        print(f"  D = {target.D}")

    rows: list[dict] = []

    if target is not None:
        map_sample = target.x_ref.unsqueeze(0)
        map_metrics = evaluate(map_sample, target, data, cfg.noise_std)
        rows.append({
            "dataset": dataset_name, "split_id": split_id, "sampler": "map",
            **map_metrics, "n_skeleton": 0, "elapsed_sec": 0.0,
        })
        print(f"  [map]              "
              f"rmse_orig={map_metrics['rmse_orig']:.4f}  "
              f"log_lik={map_metrics['log_lik']:.4f}")

    seed = BASE_SEED + split_id
    sd = split_dir(out_dir, dataset_name, split_id)

    for name in samplers_to_run:
        out_path = sd / f"{name}.pt"
        if resume and out_path.exists():
            print(f"  [{name}] skipping — exists at {out_path}")
            payload = torch.load(out_path, weights_only=False)
            rows.append({
                "dataset": dataset_name, "split_id": split_id, "sampler": name,
                **payload["metrics"],
                "n_skeleton": payload.get("n_skeleton", 0),
                "elapsed_sec": payload.get("elapsed_sec", 0.0),
            })
            continue

        print(f"  [{name}]")
        torch.manual_seed(seed)
        np.random.seed(seed)

        if name == "nuts":
            try:
                samples_np, elapsed = run_nuts(data, cfg, seed)
            except ImportError as e:
                print(f"      NUTS dependency missing ({e}); skipping.")
                continue
            samples_t = torch.tensor(samples_np)
            n_events = samples_np.shape[0]
        else:
            sampler = build_pdmp_sampler(name, target, cfg)
            t0 = time.perf_counter()
            result = sampler.sample(N=N_SKELETON, diagnostics=True)
            elapsed = time.perf_counter() - t0
            print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s")
            samples_np = resample_pdmp(name, result, x_ref_np)
            samples_t = torch.tensor(samples_np)
            n_events = N_SKELETON

        eval_target = target
        if eval_target is None:
            eval_target = build_target(data, cfg)

        metrics = evaluate(samples_t, eval_target, data, cfg.noise_std)

        print(f"      rmse_orig={metrics['rmse_orig']:.4f}  "
              f"log_lik={metrics['log_lik']:.4f}  "
              f"pred_std={metrics['pred_std_mean']:.4f}  "
              f"ess_med={metrics['ess_median']:.1f}")

        path = save_run(out_dir, dataset_name, split_id, name,
                        samples_t, eval_target, cfg,
                        metrics, n_events, elapsed, data["y_std"])
        print(f"      saved -> {path}")

        rows.append({
            "dataset": dataset_name, "split_id": split_id, "sampler": name,
            **metrics, "n_skeleton": n_events, "elapsed_sec": elapsed,
        })

    map_rows = [r for r in rows if r["sampler"] == "map"]
    disk_rows = []
    for pt_path in sorted(sd.glob("*.pt")):
        payload = torch.load(pt_path, weights_only=False)
        disk_rows.append({
            "dataset":     payload.get("dataset", dataset_name),
            "split_id":    payload.get("split_id", split_id),
            "sampler":     payload.get("sampler", pt_path.stem),
            **payload.get("metrics", {}),
            "n_skeleton":  payload.get("n_skeleton", 0),
            "elapsed_sec": payload.get("elapsed_sec", 0.0),
        })
    final_rows = map_rows + disk_rows
    with open(sd / "metrics.json", "w") as f:
        json.dump(final_rows, f, indent=2)

    return final_rows


# ===========================================================================
# Aggregation — identical to uci_bnn.aggregate
# ===========================================================================

def aggregate(out_dir: Path, datasets: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for ds_dir in sorted(out_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        if datasets is not None and ds_dir.name not in datasets:
            continue
        for split in sorted(ds_dir.glob("split_*")):
            split_id = int(split.name.split("_")[-1])
            ds_name = ds_dir.name

            json_rows: list[dict] = []
            mj = split / "metrics.json"
            if mj.exists():
                with open(mj) as f:
                    json_rows = json.load(f)

            seen = {r["sampler"] for r in json_rows}
            for pt_path in sorted(split.glob("*.pt")):
                if pt_path.stem in seen:
                    continue
                payload = torch.load(pt_path, weights_only=False)
                json_rows.append({
                    "dataset":     payload.get("dataset", ds_name),
                    "split_id":    payload.get("split_id", split_id),
                    "sampler":     payload.get("sampler", pt_path.stem),
                    **payload.get("metrics", {}),
                    "n_skeleton":  payload.get("n_skeleton", 0),
                    "elapsed_sec": payload.get("elapsed_sec", 0.0),
                })

            rows.extend(json_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No metrics found to aggregate.")
        return df

    csv_path = out_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}  ({len(df)} rows)")

    metric_cols = ["rmse_orig", "rmse_std", "log_lik", "nll_orig",
                   "pred_std_mean", "ess_median", "elapsed_sec"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    summary = (
        df.groupby(["dataset", "sampler"])[metric_cols]
          .agg(["mean", "sem"])
          .round(4)
    )
    print("\n=== Summary (mean ± stderr over splits) ===")
    print(summary.to_string())
    return df


# ===========================================================================
# CLI — same shape as uci_bnn.main, just different default --out
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(UCI_DATASETS),
        choices=list(ALL_DATASETS),
    )
    parser.add_argument(
        "--samplers", nargs="+", default=None,
        choices=list(ALL_SAMPLER_NAMES),
    )
    parser.add_argument(
        "--splits", nargs="+", type=int, default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--include-nuts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--toy-dir", type=Path, default=TOY_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(args.out, args.datasets)
        return

    if args.samplers is None:
        samplers = list(PDMP_SAMPLER_NAMES)
        if args.include_nuts:
            samplers.append("nuts")
    else:
        samplers = list(args.samplers)
        if "nuts" in samplers and not args.include_nuts:
            print("Note: 'nuts' was explicitly requested; --include-nuts is "
                  "not required when listing samplers manually.")

    uci_to_run = [d for d in args.datasets if d in UCI_DATASETS]
    toy_to_run = [d for d in args.datasets if d in TOY_DATASETS]

    if uci_to_run:
        print("Loading raw UCI datasets...")
        raw = load_raw_datasets()
        input_dims_uci = {name: X.shape[1] for name, (X, _) in raw.items()
                          if name in uci_to_run}
        cfgs = configs_for(input_dims_uci)

        print(f"\nRunning splits {args.splits} on {uci_to_run} with "
              f"samplers {samplers} (v2 pipeline)")
        for ds_name in uci_to_run:
            X, y = raw[ds_name]
            for split_id in args.splits:
                seed = BASE_SEED + split_id
                data = make_split(ds_name, X, y, seed=seed)
                run_split(ds_name, split_id, data, cfgs[ds_name],
                          args.out, tuple(samplers), resume=args.resume)

    if toy_to_run:
        print("\nLoading toy 1D datasets...")
        try:
            toy_pairs = {name: load_toy(name, args.toy_dir) for name in toy_to_run}
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nGenerate the toy datasets first with:")
            print(f"    python -m sazz.scripts.generate_toys --names {' '.join(toy_to_run)}")
            return

        print(f"\nRunning {toy_to_run} (single split each) with "
              f"samplers {samplers} (v2 pipeline)")
        for ds_name in toy_to_run:
            data, cfg = toy_pairs[ds_name]
            run_split(ds_name, 0, data, cfg,
                      args.out, tuple(samplers), resume=args.resume)

    aggregate(args.out, args.datasets)


if __name__ == "__main__":
    main()