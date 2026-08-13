"""Logistic regression — Boomerang & Zig-Zag (sticky and non-sticky) vs NUTS.

Small synthetic benchmark (N=300, D=9) with a Gaussian prior. No closed-form
posterior; NUTS-Gaussian is used as the σ-ratio reference.

Usage:
    python -m sazz.scripts.logistic_regression
    python -m sazz.scripts.logistic_regression --N 500 --D 20 --n-signals 5 --seed 1
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from sazz.models.model import BayesianModel, TorchTarget
from sazz.models.priors import ModuleGaussianPrior
from sazz.models.likelihoods import ModuleBernoulliLikelihood
from sazz.models.glms import LogisticRegressionNet
from sazz.utils.bnn_utils import ParamSpec
from sazz.utils.warmup import find_reference_glm
from sazz.utils.glm_utils import (
    set_seed, run_pdmps, print_table, plot_coefs, save_run, _support_via_ci,
)

torch.set_default_dtype(torch.float64)


@dataclass
class Config:
    N: int = 300
    D: int = 8
    n_signals: int = 3
    signal_scale: float = 1.5
    intercept_true: float = 0.5

    prior_std: float = 1.0
    bias_prior_std: float = 10.0

    n_skel: int = 10_000
    n_resample: int = 5_000
    burnin_frac: float = 0.5
    refresh_rate: float = 1.0
    kappa_null: float = 0.4
    kappa_int: float = 1e6
    thinning: str = "pli"
    t_max_zz: float = 0.1
    gamma_zz: float = 0.01

    nuts_draws: int = 2_000
    nuts_tune: int = 1_000
    nuts_chains: int = 4

    seed: int = 0


def make_data(cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    X = rng.normal(size=(cfg.N, cfg.D))
    beta = np.zeros(cfg.D)
    if cfg.n_signals > 0:
        idx = np.sort(rng.choice(cfg.D, size=cfg.n_signals, replace=False))
        beta[idx] = cfg.signal_scale * rng.normal(size=cfg.n_signals)
    logits = X @ beta + cfg.intercept_true
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))
    X = (X - X.mean(0)) / X.std(0)
    true_coefs = np.concatenate([[cfg.intercept_true], beta])
    return X, y, true_coefs, true_coefs != 0


def make_target(X, y, cfg: Config) -> TorchTarget:
    X_t = torch.tensor(X)
    y_t = torch.tensor(y, dtype=torch.float64)
    net = LogisticRegressionNet(cfg.D)
    spec = ParamSpec.from_module(net)
    prec = torch.tensor(
        [1.0 / (cfg.bias_prior_std ** 2 if is_b else cfg.prior_std ** 2)
         for sz, is_b in zip(spec.numels, spec.is_bias) for _ in range(sz)]
    )
    model = BayesianModel(ModuleGaussianPrior(prec), ModuleBernoulliLikelihood(net, spec, X_t, y_t))
    x_ref, Sigma_inv = find_reference_glm(model.energy, spec.D, diagonal_only=False)
    return TorchTarget(f"logreg_D{cfg.D}", spec.D, model.grad_energy, x_ref, Sigma_inv,
                       meta={"model": model, "net": net, "spec": spec})


def run_nuts(X, y, cfg: Config) -> dict:
    import pymc as pm
    set_seed(cfg.seed)
    with pm.Model():
        intercept = pm.Normal("intercept", mu=0.0, sigma=cfg.bias_prior_std)
        betas = pm.Normal("betas", mu=0.0, sigma=cfg.prior_std, shape=cfg.D)
        pm.Bernoulli("y", logit_p=intercept + X @ betas, observed=y.astype(int))
        t0 = time.perf_counter()
        trace = pm.sample(draws=cfg.nuts_draws, tune=cfg.nuts_tune, chains=cfg.nuts_chains,
                          target_accept=0.9, progressbar=False, random_seed=cfg.seed)
        wall = time.perf_counter() - t0
    int3   = trace.posterior["intercept"].values[..., None]
    betas3 = trace.posterior["betas"].values
    samples = np.concatenate([int3, betas3], axis=-1).reshape(-1, cfg.D + 1)
    return dict(name="NUTS", samples=samples, wall=wall, sticky=False, color="C2", marker="D")


def run_nuts_horseshoe(X, y, cfg: Config) -> dict:
    import pymc as pm
    set_seed(cfg.seed)
    with pm.Model():
        intercept = pm.Normal("intercept", mu=0.0, sigma=cfg.bias_prior_std)
        tau   = pm.HalfCauchy("tau", beta=cfg.prior_std)
        lam   = pm.HalfCauchy("lam", beta=1.0, shape=cfg.D)
        W_raw = pm.Normal("W_raw", mu=0.0, sigma=1.0, shape=cfg.D)
        betas = pm.Deterministic("betas", tau * lam * W_raw)
        pm.Bernoulli("y", logit_p=intercept + X @ betas, observed=y.astype(int))
        t0 = time.perf_counter()
        trace = pm.sample(draws=cfg.nuts_draws, tune=cfg.nuts_tune, chains=cfg.nuts_chains,
                          target_accept=0.95, progressbar=False, random_seed=cfg.seed)
        wall = time.perf_counter() - t0
    int3   = trace.posterior["intercept"].values[..., None]
    betas3 = trace.posterior["betas"].values
    samples = np.concatenate([int3, betas3], axis=-1).reshape(-1, cfg.D + 1)
    return dict(name="NUTS-HS", samples=samples, wall=wall, sticky=False, color="C4", marker="P")


def compute_metrics(method, true_coefs, is_signal, X_te_aug, y_te, sd_ref):
    s = method["samples"]
    mu, sd = s.mean(0), s.std(0)
    probs  = 1.0 / (1.0 + np.exp(-(s @ X_te_aug.T)))
    p_mean = np.clip(probs.mean(0), 1e-12, 1 - 1e-12)
    nll    = -float((y_te * np.log(p_mean) + (1 - y_te) * np.log(1 - p_mean)).mean())
    p_zero = (np.abs(s) < 1e-8).mean(0) if method["sticky"] else None
    return {**method,
            "beta_rmse":   float(np.sqrt(((mu - true_coefs) ** 2).mean())),
            "test_nll":    nll,
            "f1_ci":       float(f1_score(is_signal, _support_via_ci(s), zero_division=0)),
            "sigma_ratio": float((sd[is_signal] / sd_ref[is_signal]).mean()) if sd_ref is not None and is_signal.any() else float("nan"),
            "p0_nulls":    float(p_zero[~is_signal].mean()) if (p_zero is not None and (~is_signal).any()) else float("nan")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N",          type=int,   default=300)
    p.add_argument("--D",          type=int,   default=8)
    p.add_argument("--n-signals",  type=int,   default=3)
    p.add_argument("--signal-scale", type=float, default=1.5)
    p.add_argument("--prior-std",  type=float, default=1.0)
    p.add_argument("--n-skel",     type=int,   default=10_000)
    p.add_argument("--n-resample", type=int,   default=5_000)
    p.add_argument("--thinning",   choices=["pli", "brent"], default="pli")
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--no-nuts",    action="store_true")
    p.add_argument("--nuts-hs",    action="store_true", help="Also run NUTS with horseshoe prior")
    p.add_argument("--no-plots",   action="store_true")
    p.add_argument("--save",       action="store_true")
    args = p.parse_args()

    cfg = Config(N=args.N, D=args.D, n_signals=args.n_signals,
                 signal_scale=args.signal_scale, prior_std=args.prior_std,
                 n_skel=args.n_skel, n_resample=args.n_resample,
                 thinning=args.thinning, seed=args.seed)
    set_seed(cfg.seed)
    print(f"Logistic regression  N={cfg.N} D={cfg.D} K={cfg.n_signals} seed={cfg.seed}")

    X, y, true_coefs, is_signal = make_data(cfg)
    print(f"class balance = {y.mean():.2f}")

    test_rng = np.random.default_rng(cfg.seed + 1)
    X_te = (lambda Z: (Z - Z.mean(0)) / Z.std(0))(test_rng.normal(size=(cfg.N, cfg.D)))
    y_te = test_rng.binomial(1, 1.0 / (1.0 + np.exp(-(X_te @ true_coefs[1:] + true_coefs[0]))))
    X_te_aug = np.column_stack([np.ones(cfg.N), X_te])

    rows = []
    if not args.no_nuts:
        print("Running NUTS...")
        rows.append(run_nuts(X, y, cfg))
    if args.nuts_hs:
        print("Running NUTS (horseshoe)...")
        rows.append(run_nuts_horseshoe(X, y, cfg))
    sd_ref = rows[0]["samples"].std(0) if rows else None

    target = make_target(X, y, cfg)
    print("Running PDMP samplers...")
    rows.extend(run_pdmps(target, cfg))

    rows = [compute_metrics(r, true_coefs, is_signal, X_te_aug, y_te, sd_ref) for r in rows]
    print_table(rows, metric_key="test_nll", metric_label="test NLL")

    out_dir = Path(f"results/logistic_regression/{args.thinning}/N{cfg.N}_D{cfg.D}_K{cfg.n_signals}_seed{cfg.seed}") if args.save else None
    if not args.no_plots:
        plot_coefs(rows, true_coefs, is_signal,
                   f"Logistic regression D={cfg.D}, K={cfg.n_signals}",
                   (out_dir / "coefs.png") if out_dir else None)
    if args.save:
        save_run(out_dir, cfg, rows,
                 ["name", "wall", "sticky", "beta_rmse", "test_nll", "f1_ci", "sigma_ratio", "p0_nulls"],
                 true_coefs, is_signal)


if __name__ == "__main__":
    main()
