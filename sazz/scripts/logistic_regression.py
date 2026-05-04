"""Logistic regression — Boomerang & Zig-Zag (sticky and non-sticky) vs NUTS.

Synthetic binary-classification benchmark with known ground-truth coefficients.
The model always includes an intercept coordinate (slot 0); `--intercept 0`
keeps the slot but generates data with zero true intercept. No closed-form
posterior exists for logistic regression, so NUTS-Gaussian is the σ-ratio
reference.

Usage:
    python -m sazz.scripts.logistic_regression
    python -m sazz.scripts.logistic_regression --N 2000 --D 200 \
        --n-signals 10 --signal-scale 2.0 \
        --prior-dist "Gauss" --thinning pli \
        --prior-std 5.0 --include-horseshoe --save --no-plots

    python -m sazz.scripts.logistic_regression --N 2000 --D 200 \
        --n-signals 8 --signal-scale 2.0 \
        --prior-dist "StudT" --thinning brent \
        --n_skel 10_000 --n_resample 5000 \
        --prior-std 5.0 --include-horseshoe --save --no-plots
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
from sazz.models.make_models import make_logistic_regression
from sazz.utils.sampling import (
    resample_boomerang_path, resample_boomerang_path_sticky,
    resample_zigzag_path, resample_zigzag_path_sticky,
)

torch.set_default_dtype(torch.float64)


@dataclass
class Config:
    # Data — D is the number of *covariates*; sampler dim is D + 1 (slot 0 = intercept).
    N: int = 300
    D: int = 9
    n_signals: int = 3
    signal_scale: float = 1.5
    intercept_true: float = 0.5

    # Prior (shared by NUTS and PDMP)
    prior_dist: str = "Gauss"
    prior_std: float = 1.0
    intercept_prior_std: float = 10.0

    # PDMP budgets
    n_skel: int = 100_000
    n_resample: int = 50_000
    burnin_frac: float = 0.5
    refresh_rate: float = 1.0
    kappa_null: float = 0.4             # ~ 1/sqrt(2*pi); intercept gets kappa_int
    kappa_int: float = 1e6
    thinning: str = "pli"
    t_max_zz: float = 0.1
    gamma_zz: float = 0.01

    # NUTS
    nuts_draws: int = 2_000
    nuts_tune: int = 1_000
    nuts_chains: int = 2
    nuts_target_accept: float = 0.9

    seed: int = 0


def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)


# ===========================================================================
# Data
# ===========================================================================

def make_data(cfg: Config):
    """Returns (X_std, y, true_coefs, is_signal). true_coefs has length D + 1."""
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


# ===========================================================================
# PDMP samplers
# ===========================================================================

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
    """target.D == cfg.D + 1; coordinate 0 is the intercept (kappa_int → never freezes)."""
    kappa = torch.full((target.D,), cfg.kappa_null, dtype=torch.float64)
    kappa[0] = cfg.kappa_int
    x_ref_np = target.x_ref.cpu().numpy()

    out = []
    for name, family, sticky, color, marker in PDMP_SPECS:
        set_seed(cfg.seed)
        s = _build(family, sticky, target, cfg, kappa)
        x0 = target.x_ref.clone()
        t0 = time.perf_counter()
        res = s.sample(N=cfg.n_skel, x0=x0, diagnostics=False)
        samples = _resample(family, sticky, res, x_ref_np, cfg)
        wall = time.perf_counter() - t0     # incl. resampling, matches NUTS pipeline timing
        out.append(dict(name=name, samples=samples, wall=wall,
                        sticky=sticky, color=color, marker=marker))
        print(f"  {name:<13} wall={wall:6.2f}s  draws={samples.shape[0]}")
    return out


# ===========================================================================
# NUTS — same Bernoulli likelihood + Gaussian/horseshoe prior on β
# ===========================================================================

def run_nuts(X, y, cfg: Config, prior: str, k_signals_guess: int = 0) -> dict:
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
        pm.Bernoulli("y", logit_p=intercept + X @ betas, observed=y.astype(int))

        t0 = time.perf_counter()
        trace = pm.sample(
            draws=cfg.nuts_draws, tune=cfg.nuts_tune, chains=cfg.nuts_chains,
            target_accept=target_accept, progressbar=False, random_seed=cfg.seed,
        )
        wall = time.perf_counter() - t0

    int3   = trace.posterior["intercept"].values[..., None]
    betas3 = trace.posterior["betas"].values
    samples = np.concatenate([int3, betas3], axis=-1).reshape(-1, cfg.D + 1)

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
                    X_test_aug, y_test, sd_ref) -> dict:
    """Test NLL is the negative log predictive density on a fresh test set,
    marginalised over the full posterior (probabilities averaged across draws)."""
    from sklearn.metrics import f1_score
    s = method["samples"]
    mu, sd = s.mean(0), s.std(0)
    probs  = 1.0 / (1.0 + np.exp(-(s @ X_test_aug.T)))
    p_mean = np.clip(probs.mean(0), 1e-12, 1 - 1e-12)
    nll    = -float((y_test * np.log(p_mean) + (1 - y_test) * np.log(1 - p_mean)).mean())
    p_zero = (np.abs(s) < 1e-8).mean(0) if method["sticky"] else None

    out = dict(method)
    out["beta_rmse"]   = float(np.sqrt(((mu - true_coefs) ** 2).mean()))
    out["test_nll"]    = nll
    out["f1_ci"]       = float(f1_score(is_signal, _support_via_ci(s), zero_division=0))
    out["sigma_ratio"] = (float((sd[is_signal] / sd_ref[is_signal]).mean())
                          if sd_ref is not None and is_signal.any() else float("nan"))
    out["p0_nulls"]    = (float(p_zero[~is_signal].mean())
                          if (p_zero is not None and (~is_signal).any()) else float("nan"))
    return out


def print_table(rows: list[dict]):
    h = (f"{'sampler':<16} {'wall':>7} {'β-RMSE':>9} {'test NLL':>10} "
         f"{'σ-ratio':>9} {'F1(95% CI)':>11} {'P(=0) nulls':>12}")
    print("\n" + h); print("-" * len(h))
    for r in rows:
        wall = f"{r['wall']:>7.2f}" if r.get("wall") is not None else f"{'—':>7}"
        sr = f"{r['sigma_ratio']:>9.3f}" if not np.isnan(r["sigma_ratio"]) else f"{'—':>9}"
        p0 = f"{r['p0_nulls']:>12.3f}"   if not np.isnan(r["p0_nulls"])    else f"{'—':>12}"
        print(f"{r['name']:<16} {wall} {r['beta_rmse']:>9.4f} "
              f"{r['test_nll']:>10.4f} {sr} {r['f1_ci']:>11.3f} {p0}")


# ===========================================================================
# Plot
# ===========================================================================

def plot_coefs(rows, true_coefs, is_signal, title, out_path: Path | None):
    import matplotlib.pyplot as plt
    sparse = (~is_signal).sum() > len(true_coefs) // 2
    sig = np.where(is_signal)[0]; nul = np.where(~is_signal)[0]
    offsets = np.linspace(-0.30, 0.30, len(rows))

    if not sparse:
        fig, ax = plt.subplots(figsize=(min(13, 1.0 * len(true_coefs) + 4), 4.5))
        idx = np.arange(len(true_coefs))
        for r, off in zip(rows, offsets):
            mu, sd = r["samples"].mean(0), r["samples"].std(0)
            ax.errorbar(idx + off, mu, yerr=2 * sd, fmt=r["marker"], color=r["color"],
                        label=r["name"], capsize=2, markersize=4, lw=1)
        ax.scatter(idx, true_coefs, marker="x", color="k", s=70, linewidths=2,
                   label="true", zorder=5)
        for i in sig: ax.axvspan(i - 0.45, i + 0.45, color="gold", alpha=0.12)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set(xlabel="coordinate", ylabel="coefficient", title=title)
        ax.legend(fontsize=8, ncol=2, frameon=False)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                       gridspec_kw={"height_ratios": [1.2, 1]})
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
        ax1.set_xticks(xs); ax1.set_xticklabels([f"β_{i}" for i in sig], rotation=45, ha="right")
        ax1.set(ylabel="coefficient", title=f"Signals — {title}")
        ax1.legend(fontsize=8, ncol=3, frameon=False)
        ax2.axhline(0, color="red", lw=1)
        ax2.set(xlabel="coordinate", ylabel="posterior mean",
                title=f"Nulls ({len(nul)} of {len(true_coefs)})")
        ax2.legend(fontsize=8, ncol=3, frameon=False)

    plt.tight_layout()
    if out_path: plt.savefig(out_path, dpi=120)
    plt.show()


# ===========================================================================
# Persistence
# ===========================================================================

_METRIC_KEYS = ["name", "wall", "sticky", "beta_rmse", "test_nll",
                "f1_ci", "sigma_ratio", "p0_nulls"]


def _to_jsonable(v):
    """Recursively convert torch/numpy scalars and arrays into JSON-safe primitives."""
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return None if np.isnan(v) else v
    if isinstance(v, np.integer):                  return int(v)
    if isinstance(v, np.floating):
        f = float(v); return None if np.isnan(f) else f
    if isinstance(v, np.ndarray):                  return v.tolist()
    if isinstance(v, torch.Tensor):                return v.detach().cpu().tolist()
    if isinstance(v, (list, tuple)):               return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):                        return {str(k): _to_jsonable(x) for k, x in v.items()}
    return None


def save_run(out_dir: Path, cfg: Config, rows, true_coefs, is_signal):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = _to_jsonable(asdict(cfg))
    for r in rows:
        if "samples" not in r: continue
        torch.save({
            "name": r["name"], "samples": torch.tensor(r["samples"]),
            "wall": r.get("wall"), "config": cfg_dict,
        }, out_dir / f"{r['name'].replace(' ', '_')}.pt")
    metric_rows = [{k: _to_jsonable(r.get(k)) for k in _METRIC_KEYS} for r in rows]
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "config":     cfg_dict,
            "true_coefs": true_coefs.tolist(),
            "is_signal":  [bool(b) for b in is_signal.tolist()],
            "rows":       metric_rows,
        }, f, indent=2)
    print(f"\nSaved -> {out_dir}/")


# ===========================================================================
# Main
# ===========================================================================

def main():
    defaults = Config()
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--N",                   type=int,   default=defaults.N)
    p.add_argument("--D",                   type=int,   default=defaults.D)
    p.add_argument("--n-signals",           type=int,   default=defaults.n_signals)
    p.add_argument("--signal-scale",        type=float, default=defaults.signal_scale)
    p.add_argument("--intercept",           type=float, default=defaults.intercept_true,
                   help="True intercept value (the slot always exists in the model).")
    p.add_argument("--prior-dist", choices=["Gauss", "StudT"],
                    default=defaults.prior_dist,
                    help="Prior on slope coefficients.")
    p.add_argument("--prior-std",           type=float, default=defaults.prior_std)
    p.add_argument("--intercept-prior-std", type=float, default=defaults.intercept_prior_std)
    p.add_argument("--n-skel",              type=int,   default=defaults.n_skel)
    p.add_argument("--n-resample",          type=int,   default=defaults.n_resample)
    p.add_argument("--thinning",            choices=["pli", "brent"], default=defaults.thinning)
    p.add_argument("--seed",                type=int,   default=defaults.seed)
    p.add_argument("--include-horseshoe", action="store_true")
    p.add_argument("--no-nuts",           action="store_true")
    p.add_argument("--no-plots",          action="store_true")
    p.add_argument("--save",              action="store_true")
    args = p.parse_args()

    cfg = Config(
        N=args.N, D=args.D, n_signals=args.n_signals, signal_scale=args.signal_scale,
        intercept_true=args.intercept, 
        prior_dist=args.prior_dist,
        prior_std=args.prior_std, intercept_prior_std=args.intercept_prior_std,
        n_skel=args.n_skel, n_resample=args.n_resample,
        thinning=args.thinning, seed=args.seed,
    )
    set_seed(cfg.seed)
    print(f"Config: N={cfg.N} D={cfg.D} K={cfg.n_signals} prior={cfg.prior_dist} "
        f"thinning={cfg.thinning} seed={cfg.seed}")

    # --- Data + held-out test set (test X has D covariates; intercept added below) ---
    X, y, true_coefs, is_signal = make_data(cfg)
    print(f"signals = {int(is_signal.sum())}/{len(true_coefs)}  "
          f"intercept_true = {cfg.intercept_true}  class balance = {y.mean():.2f}")

    test_rng = np.random.default_rng(cfg.seed + 1)
    X_te = test_rng.normal(size=(cfg.N, cfg.D))
    X_te = (X_te - X_te.mean(0)) / X_te.std(0)
    y_te = test_rng.binomial(1, 1.0 / (1.0 + np.exp(-(X_te @ true_coefs[1:] + true_coefs[0]))))
    X_te_aug = np.column_stack([np.ones(cfg.N), X_te])

    # --- Run methods ---
    rows: list[dict] = []
    if not args.no_nuts:
        print("\nRunning Gaussian-prior NUTS...")
        rows.append(run_nuts(X, y, cfg, prior="gaussian"))
    if args.include_horseshoe:
        print("\nRunning horseshoe NUTS (slow)...")
        rows.append(run_nuts(X, y, cfg, prior="horseshoe", k_signals_guess=cfg.n_signals))

    target = make_logistic_regression(
        torch.tensor(X), torch.tensor(y, dtype=torch.long),
        prior_dist=cfg.prior_dist,
        prior_std=cfg.prior_std, intercept_prior_std=cfg.intercept_prior_std,
        diagonal_only=False,
    )
    print("\nRunning PDMP samplers...")
    rows.extend(run_pdmps(target, cfg))

    # --- Metrics (NUTS-Gaussian is the σ-ratio reference) ---
    sd_ref = next((r["samples"].std(0) for r in rows
                   if r["name"] == "NUTS-Gaussian"), None)
    rows = [compute_metrics(r, true_coefs, is_signal, X_te_aug, y_te, sd_ref)
            for r in rows]
    print_table(rows)

    # --- Plot + save ---
    out_dir = Path("results/logistic_regression") / _auto_name(cfg) if args.save else None
    if out_dir: out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        title = f"D={cfg.D}, K={cfg.n_signals}, {cfg.prior_dist} prior, {cfg.thinning}"
        plot_coefs(rows, true_coefs, is_signal, title,
                   (out_dir / "coefs.png") if out_dir else None)
    if args.save:
        save_run(out_dir, cfg, rows, true_coefs, is_signal)


def _auto_name(cfg: Config) -> str:
    return f"N{cfg.N}_D{cfg.D}_K{cfg.n_signals}_{cfg.prior_dist}_{cfg.thinning}_seed{cfg.seed}"


if __name__ == "__main__":
    main()

