"""Linear regression — Boomerang & Zig-Zag (sticky and non-sticky) vs NUTS.

A synthetic linear-regression benchmark with known ground-truth coefficients
and an analytical Gaussian-prior posterior. Runs four PDMP samplers,
Gaussian-prior NUTS, and optionally horseshoe NUTS. Reports parameter
recovery, predictive RMSE, posterior calibration, and (when sparse)
variable-selection F1.

The model always includes an intercept coordinate. Use `--intercept 0` for
sparse benchmarks where you want the "true intercept" to be zero — the
intercept slot still exists in the model and is sampled, just shrunk
toward zero by its prior.

The same script handles both regimes:
  - Small dense (D=9, 3 signals): the original notebook benchmark.
  - Sparse high-D (D=200, K=10): the closed-form / horseshoe comparison.

Switching between them is just CLI args.

Usage:
    # Small dense linear regression (notebook default)
    python -m sazz.scripts.linear_regression

    # Sparse regression (D=200, 10 true signals, true intercept = 0, save)
    python -m sazz.scripts.linear_regression --N 2000 --D 200 \
        --n-signals 10 --signal-scale 2.0 --intercept 0 --noise-std 1.0 \
        --prior-std 5.0 --lik-noise-std 1.0 --include-horseshoe --save

    # PDMP-only, no NUTS, no plots
    python -m sazz.scripts.linear_regression --no-nuts --no-plots

Storage (only when --save is passed):
    results/linear_regression/<name>/
      <sampler>.pt    # samples + config, one per sampler
      metrics.json    # all metric rows + true_coefs + config
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from sazz.samplers.AutomaticBoomerangSampler import AutomaticBoomerangSampler
from sazz.samplers.StickyAutomaticBoomerangSampler import StickyAutomaticBoomerangSampler
from sazz.samplers.AutomaticZigZagSampler_torch import AutomaticZigZagSampler
from sazz.samplers.StickyAutomaticZigZagSampler_torch import StickyAutomaticZigZagSampler
from sazz.models.make_models import make_linear_regression
from sazz.utils.sampling import (
    resample_boomerang_path, resample_boomerang_path_sticky,
    resample_zigzag_path, resample_zigzag_path_sticky,
)

torch.set_default_dtype(torch.float64)


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class Config:
    # Data
    N: int = 200
    D: int = 9                          # number of *covariates*; total dim is D + 1 (intercept)
    n_signals: int = 3
    signal_scale: float = 1.5
    intercept_true: float = 0.5         # can be 0 — intercept slot still exists in the model
    noise_std: float = 0.3              # DGP noise

    # Prior / likelihood (PDMPs and NUTS share these)
    prior_std: float = 1.0              # prior std on β coordinates
    intercept_prior_std: float = 10.0   # prior std on intercept (kept wide to let data speak)
    lik_noise_std: float = 0.5

    # PDMP budgets
    n_skel: int = 50_000
    n_resample: int = 4_000
    burnin_frac: float = 0.5
    refresh_rate: float = 1.0
    kappa_null: float = 1.0
    kappa_int: float = 1e6              # intercept never sticks
    thinning: str = "pli"
    t_max_zz: float = 0.1
    gamma_zz: float = 0.01

    # NUTS
    nuts_draws: int = 2_000
    nuts_tune: int = 1_000
    nuts_chains: int = 4
    nuts_target_accept: float = 0.9

    seed: int = 0


def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)


# ===========================================================================
# Data + analytic posterior
# ===========================================================================

def make_data(cfg: Config):
    """Generate (X_std, y, true_coefs, is_signal).

    `true_coefs` always has length D + 1: [intercept, β_1, ..., β_D].
    `is_signal[0]` is True iff `intercept_true != 0`.
    """
    rng = np.random.default_rng(cfg.seed)
    X = rng.normal(size=(cfg.N, cfg.D))
    beta = np.zeros(cfg.D)
    if cfg.n_signals > 0:
        idx = np.sort(rng.choice(cfg.D, size=cfg.n_signals, replace=False))
        beta[idx] = cfg.signal_scale * rng.normal(size=cfg.n_signals)
    y = X @ beta + cfg.intercept_true + cfg.noise_std * rng.normal(size=cfg.N)
    X = (X - X.mean(0)) / X.std(0)
    true_coefs = np.concatenate([[cfg.intercept_true], beta])
    return X, y, true_coefs, true_coefs != 0


def analytic_posterior(X_aug: np.ndarray, y: np.ndarray, cfg: Config):
    """Closed-form posterior on [intercept, β]. X_aug has the leading ones column.

    Diagonal prior precision: intercept gets 1/σ_int², slopes get 1/σ_p².
    """
    D_total = X_aug.shape[1]            # D + 1
    prec_prior = np.eye(D_total) / cfg.prior_std ** 2
    prec_prior[0, 0] = 1.0 / cfg.intercept_prior_std ** 2
    prec = X_aug.T @ X_aug / cfg.lik_noise_std ** 2 + prec_prior
    Sigma = np.linalg.inv(prec)
    mu = Sigma @ (X_aug.T @ y / cfg.lik_noise_std ** 2)
    return mu, Sigma


# ===========================================================================
# PDMP samplers — table-driven
# ===========================================================================

# (display name, family, sticky, color, marker)
PDMP_SPECS = [
    ("Boom",        "boomerang", False, "C0", "o"),
    ("Sticky-Boom", "boomerang", True,  "C1", "s"),
    ("ZZ",          "zigzag",    False, "C9", "v"),
    ("Sticky-ZZ",   "zigzag",    True,  "C3", "^"),
]


def _build(family, sticky, target, cfg, kappa):
    common = dict(grad_target=target.grad_target, D=target.D, thinning=cfg.thinning)
    if family == "boomerang":
        kw = dict(refresh_rate=cfg.refresh_rate)
        s = (StickyAutomaticBoomerangSampler(**common, **kw, kappa=kappa) if sticky
             else AutomaticBoomerangSampler(**common, **kw))
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
    else:
        kw = dict(t_max=cfg.t_max_zz, gamma=cfg.gamma_zz)
        s = (StickyAutomaticZigZagSampler(**common, **kw, kappa=kappa) if sticky
             else AutomaticZigZagSampler(**common, **kw))
    return s


def _resample(family, sticky, res, x_ref_np, cfg):
    pos = res["positions"].cpu().numpy()
    vel = res["velocities"].cpu().numpy()
    tim = res["times"].cpu().numpy()
    kw = dict(N_resample=cfg.n_resample, burnin_frac=cfg.burnin_frac)
    if family == "boomerang":
        fn = resample_boomerang_path_sticky if sticky else resample_boomerang_path
        return fn(pos, vel, tim, x_ref_np, **kw)
    fn = resample_zigzag_path_sticky if sticky else resample_zigzag_path
    return fn(pos, vel, tim, **kw)


def run_pdmps(target, cfg: Config) -> list[dict]:
    """Run all four PDMP samplers; return list of method dicts."""
    # The intercept is coordinate 0 in the PDMP target (D + 1 wide).
    kappa = torch.full((target.D,), cfg.kappa_null, dtype=torch.float64)
    kappa[0] = cfg.kappa_int

    x_ref_np = target.x_ref.cpu().numpy()
    out = []
    for name, family, sticky, color, marker in PDMP_SPECS:
        set_seed(cfg.seed)
        s = _build(family, sticky, target, cfg, kappa)
        x0 = target.x_ref.clone() + torch.randn(target.D)
        t0 = time.perf_counter()
        res = s.sample(N=cfg.n_skel, x0=x0, diagnostics=True)
        wall = time.perf_counter() - t0
        samples = _resample(family, sticky, res, x_ref_np, cfg)
        out.append(dict(name=name, samples=samples, wall=wall,
                        sticky=sticky, color=color, marker=marker))
        print(f"  {name:<13} wall={wall:6.2f}s  draws={samples.shape[0]}")
    return out


# ===========================================================================
# NUTS (Gaussian or horseshoe prior on β; intercept always Gaussian)
# ===========================================================================

def run_nuts(X, y, cfg: Config, prior: str, k_signals_guess: int = 0) -> dict:
    """Run NUTS on the same Gaussian-likelihood model as the PDMPs.

    Always samples an intercept; `prior` controls only the prior on β.
    Returns samples in [intercept, β_1, ..., β_D] order — the same convention
    as `make_data` and the PDMP samplers.
    """
    import pymc as pm
    set_seed(cfg.seed)
    target_accept = cfg.nuts_target_accept

    with pm.Model():
        intercept = pm.Normal("intercept", mu=0.0, sigma=cfg.intercept_prior_std)
        if prior == "gaussian":
            betas = pm.Normal("betas", mu=0.0, sigma=cfg.prior_std, shape=cfg.D)
        elif prior == "horseshoe":
            tau0 = max(k_signals_guess, 1) / max(cfg.D - k_signals_guess, 1) / np.sqrt(cfg.N)
            tau     = pm.HalfCauchy("tau",     beta=tau0)
            lambdas = pm.HalfCauchy("lambdas", beta=1.0, shape=cfg.D)
            z       = pm.Normal("z", mu=0.0, sigma=1.0, shape=cfg.D)
            betas   = pm.Deterministic("betas", z * lambdas * tau)
            target_accept = max(target_accept, 0.95)
        else:
            raise ValueError(prior)
        pm.Normal("y", mu=intercept + X @ betas, sigma=cfg.lik_noise_std, observed=y)

        t0 = time.perf_counter()
        trace = pm.sample(
            draws=cfg.nuts_draws, tune=cfg.nuts_tune, chains=cfg.nuts_chains,
            target_accept=target_accept, progressbar=False, random_seed=cfg.seed,
        )
        wall = time.perf_counter() - t0

    # Concatenate as [intercept, β]; reshape (chains, draws, D+1) -> (chains*draws, D+1)
    int3   = trace.posterior["intercept"].values[..., None]   # (C, S, 1)
    betas3 = trace.posterior["betas"].values                  # (C, S, D)
    arr    = np.concatenate([int3, betas3], axis=-1)
    samples = arr.reshape(-1, arr.shape[-1])

    name = "NUTS-Gaussian" if prior == "gaussian" else "NUTS-horseshoe"
    color, marker = ("C2", "D") if prior == "gaussian" else ("C4", "P")
    return dict(name=name, samples=samples, wall=wall, sticky=False,
                color=color, marker=marker)


# ===========================================================================
# Metrics
# ===========================================================================

def _support_via_ci(samples, alpha=0.05):
    lo = np.quantile(samples, alpha / 2,     axis=0)
    hi = np.quantile(samples, 1 - alpha / 2, axis=0)
    return (lo > 0) | (hi < 0)


def compute_metrics(method: dict, true_coefs, is_signal,
                    X_test_aug, y_test_clean, sd_ref) -> dict:
    """Append metric fields to a method dict."""
    from sklearn.metrics import f1_score
    s = method["samples"]
    mu, sd = s.mean(0), s.std(0)
    pred_mean = (s @ X_test_aug.T).mean(0)
    p_zero = (np.abs(s) < 1e-8).mean(0) if method["sticky"] else None
    out = dict(method)
    out["beta_rmse"]   = float(np.sqrt(((mu - true_coefs) ** 2).mean()))
    out["pred_rmse"]   = float(np.sqrt(((pred_mean - y_test_clean) ** 2).mean()))
    out["f1_ci"]       = float(f1_score(is_signal, _support_via_ci(s), zero_division=0))
    out["sigma_ratio"] = (float((sd[is_signal] / sd_ref[is_signal]).mean())
                          if sd_ref is not None and is_signal.any() else float("nan"))
    out["p0_nulls"]    = (float(p_zero[~is_signal].mean())
                          if (p_zero is not None and (~is_signal).any()) else float("nan"))
    return out


def print_table(rows: list[dict]):
    h = (f"{'sampler':<16} {'wall':>7} {'β-RMSE':>9} {'pred-RMSE':>10} "
         f"{'σ-ratio':>9} {'F1(95% CI)':>11} {'P(=0) nulls':>12}")
    print("\n" + h); print("-" * len(h))
    for r in rows:
        wall = f"{r['wall']:>7.2f}" if r.get("wall") is not None else f"{'—':>7}"
        sr = f"{r['sigma_ratio']:>9.3f}" if not np.isnan(r["sigma_ratio"]) else f"{'—':>9}"
        p0 = f"{r['p0_nulls']:>12.3f}"   if not np.isnan(r["p0_nulls"])    else f"{'—':>12}"
        print(f"{r['name']:<16} {wall} {r['beta_rmse']:>9.4f} "
              f"{r['pred_rmse']:>10.4f} {sr} {r['f1_ci']:>11.3f} {p0}")


# ===========================================================================
# Plots
# ===========================================================================

def plot_coefs(rows: list[dict], true_coefs, is_signal, title, out_path: Path | None):
    """Auto-pick layout: dense (single panel) or sparse (signals + nulls)."""
    import matplotlib.pyplot as plt
    sparse = (~is_signal).sum() > len(true_coefs) // 2

    if not sparse:
        fig, ax = plt.subplots(figsize=(min(13, 1.0 * len(true_coefs) + 4), 4.5))
        idx = np.arange(len(true_coefs))
        offsets = np.linspace(-0.30, 0.30, len(rows))
        for r, off in zip(rows, offsets):
            mu, sd = r["samples"].mean(0), r["samples"].std(0)
            ax.errorbar(idx + off, mu, yerr=2 * sd, fmt=r["marker"], color=r["color"],
                        label=r["name"], capsize=2, markersize=4, lw=1)
        ax.scatter(idx, true_coefs, marker="x", color="k", s=70, linewidths=2,
                   label="true", zorder=5)
        for i in np.where(is_signal)[0]:
            ax.axvspan(i - 0.45, i + 0.45, color="gold", alpha=0.12)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("coordinate"); ax.set_ylabel("coefficient")
        ax.set_title(title); ax.legend(fontsize=8, ncol=2, frameon=False)
    else:
        sig = np.where(is_signal)[0]; nul = np.where(~is_signal)[0]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                       gridspec_kw={"height_ratios": [1.2, 1]})
        offsets = np.linspace(-0.35, 0.35, len(rows))
        xs = np.arange(len(sig))
        for r, off in zip(rows, offsets):
            mu, sd = r["samples"].mean(0), r["samples"].std(0)
            ax1.errorbar(xs + off, mu[sig], yerr=2 * sd[sig], fmt=r["marker"],
                         color=r["color"], label=r["name"], capsize=2, markersize=4, lw=1)
            ax2.scatter(nul, mu[nul], color=r["color"], marker=r["marker"],
                        s=15, alpha=0.6, label=r["name"])
        ax1.scatter(xs, true_coefs[sig], marker="x", color="red", s=60,
                    linewidths=2, label="true", zorder=5)
        ax1.axhline(0, color="grey", lw=0.5)
        ax1.set_xticks(xs)
        ax1.set_xticklabels([f"β_{i}" for i in sig], rotation=45, ha="right")
        ax1.set_ylabel("coefficient"); ax1.set_title(f"Signals — {title}")
        ax1.legend(fontsize=8, ncol=3, frameon=False)
        ax2.axhline(0, color="red", lw=1)
        ax2.set_xlabel("coordinate"); ax2.set_ylabel("posterior mean")
        ax2.set_title(f"Nulls ({len(nul)} of {len(true_coefs)})")
        ax2.legend(fontsize=8, ncol=3, frameon=False)

    plt.tight_layout()
    if out_path: plt.savefig(out_path, dpi=120)
    plt.show()


# ===========================================================================
# Persistence
# ===========================================================================

def save_run(out_dir: Path, cfg: Config, rows: list[dict],
             true_coefs, is_signal):
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if "samples" not in r: continue
        torch.save({
            "name": r["name"], "samples": torch.tensor(r["samples"]),
            "wall": r.get("wall"), "config": asdict(cfg),
        }, out_dir / f"{r['name'].replace(' ', '_')}.pt")
    metric_rows = [{k: v for k, v in r.items() if k != "samples"} for r in rows]
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "config": asdict(cfg),
            "true_coefs": true_coefs.tolist(),
            "is_signal": is_signal.tolist(),
            "rows": metric_rows,
        }, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
    print(f"\nSaved -> {out_dir}/")


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--D", type=int, default=9)
    p.add_argument("--n-signals", type=int, default=3)
    p.add_argument("--signal-scale", type=float, default=1.5)
    p.add_argument("--intercept", type=float, default=0.5,
                   help="True intercept value (the slot always exists in the model).")
    p.add_argument("--noise-std", type=float, default=0.3)
    p.add_argument("--prior-std", type=float, default=1.0)
    p.add_argument("--intercept-prior-std", type=float, default=10.0)
    p.add_argument("--lik-noise-std", type=float, default=0.5)
    p.add_argument("--n-skel", type=int, default=50_000)
    p.add_argument("--n-resample", type=int, default=50_000)
    p.add_argument("--thinning", choices=["pli", "brent"], default="pli")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-horseshoe", action="store_true")
    p.add_argument("--no-nuts", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/linear_regression"))
    p.add_argument("--name", type=str, default=None)
    args = p.parse_args()

    cfg = Config(
        N=args.N, D=args.D, n_signals=args.n_signals, signal_scale=args.signal_scale,
        intercept_true=args.intercept, noise_std=args.noise_std,
        prior_std=args.prior_std, intercept_prior_std=args.intercept_prior_std,
        lik_noise_std=args.lik_noise_std,
        n_skel=args.n_skel, n_resample=args.n_resample,
        thinning=args.thinning, seed=args.seed,
    )
    set_seed(cfg.seed)
    print(f"Config: N={cfg.N} D={cfg.D} K={cfg.n_signals} thinning={cfg.thinning} seed={cfg.seed}")

    # --- Data + held-out test set (ones column always added) ---
    X, y, true_coefs, is_signal = make_data(cfg)
    print(f"signals = {int(is_signal.sum())}/{len(true_coefs)}  "
          f"intercept_true = {cfg.intercept_true}")

    test_rng = np.random.default_rng(cfg.seed + 1)
    X_te = test_rng.normal(size=(cfg.N, cfg.D))
    X_te = (X_te - X_te.mean(0)) / X_te.std(0)
    y_te_clean = X_te @ true_coefs[1:] + true_coefs[0]
    X_te_aug = np.column_stack([np.ones(cfg.N), X_te])

    # --- Run methods ---
    rows: list[dict] = []
    X_aug = np.column_stack([np.ones(cfg.N), X])
    mu_an, Sigma_an = analytic_posterior(X_aug, y, cfg)
    an_samples = np.random.default_rng(cfg.seed).multivariate_normal(
        mu_an, Sigma_an, size=cfg.nuts_draws * cfg.nuts_chains)
    rows.append(dict(name="Analytic", samples=an_samples, wall=None, sticky=False,
                     color="k", marker="D"))
    if not args.no_nuts:
        print("\nRunning Gaussian-prior NUTS...")
        rows.append(run_nuts(X, y, cfg, prior="gaussian"))
    if args.include_horseshoe:
        print("\nRunning horseshoe NUTS (slow)...")
        rows.append(run_nuts(X, y, cfg, prior="horseshoe", k_signals_guess=cfg.n_signals))

    target = make_linear_regression(
        torch.tensor(X), torch.tensor(y),
        prior_std=cfg.prior_std, intercept_prior_std=cfg.intercept_prior_std,
        noise_std=cfg.lik_noise_std, diagonal_only=True,
    )
    print("\nRunning PDMP samplers...")
    rows.extend(run_pdmps(target, cfg))

    # --- Metrics (Analytic is always available now → use it as σ-ratio reference) ---
    sd_ref = next(r["samples"].std(0) for r in rows if r["name"] == "Analytic")
    rows = [compute_metrics(r, true_coefs, is_signal, X_te_aug, y_te_clean, sd_ref)
            for r in rows]
    print_table(rows)

    # --- Plot + save ---
    out_dir = (args.out / (args.name or _auto_name(cfg))) if args.save else None
    if out_dir: out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        title = f"D={cfg.D}, K={cfg.n_signals}, {cfg.thinning}"
        plot_coefs(rows, true_coefs, is_signal, title,
                   (out_dir / "coefs.png") if out_dir else None)
    if args.save:
        save_run(out_dir, cfg, rows, true_coefs, is_signal)


def _auto_name(cfg: Config) -> str:
    return f"N{cfg.N}_D{cfg.D}_K{cfg.n_signals}_seed{cfg.seed}"


if __name__ == "__main__":
    main()
