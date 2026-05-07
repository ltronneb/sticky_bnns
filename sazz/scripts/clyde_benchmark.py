"""Clyde, Ghosh & Littman (2010) benchmark — sticky PDMPs vs exact g-prior.

Reproduces Table 2 of the paper: RMSE of marginal inclusion probabilities and
BMA fitted values over 100 simulated datasets (n=100, p=15, g=n).

The slab is a Gaussian N(0, tau^2 * I) with tau^2 = g/n * sigma^2, which is
exactly ModuleGaussianPrior with prec = n/(g*sigma^2). The spike-and-slab
structure is handled by kappa and the sticky sampler — the prior only encodes
the slab density.

Only the sticky variants are run; the exact g-prior posterior (enumeration
over 2^p models) is the reference.

Usage:
    python -m sazz.scripts.clyde_benchmark
    python -m sazz.scripts.clyde_benchmark --n-sim 20 --n-skel 100000 --seed 1
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sazz.models.model import BayesianModel, TorchTarget
from sazz.models.priors import ModuleGaussianPrior
from sazz.models.likelihoods import ModuleGaussianLikelihood
from sazz.models.glms import LinearRegressionNet
from sazz.utils.bnn_utils import ParamSpec, make_kappa_from_inclusion
from sazz.utils.warmup import find_reference_glm
from sazz.samplers.StickyAutomaticBoomerangSampler import StickyAutomaticBoomerangSampler
from sazz.samplers.StickyAutomaticZigZagSampler import StickyAutomaticZigZagSampler
from sazz.utils.sampling import (
    resample_boomerang_path_sticky,
    resample_zigzag_path_sticky,
)

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data-generating process (matches the paper exactly)
    n: int   = 100
    p: int   = 15
    sigma: float = 1.0          # observation noise std
    g: int   = 100              # Zellner g = n

    # True coefficients (1-indexed in paper, 0-indexed here)
    # β = (−0.48, 8.72, −1.76, −1.87, 0, 0, 0, 0, 4.00, 0, 0, 0, 0, 0, 0)
    # Column 9 (index 8) is correlated ~0.99 with column 2 (index 1).
    alpha_true: float = 2.0
    rho: float = 0.99           # correlation between columns 2 and 9

    # Prior
    bias_prior_std: float = 2.0   # wide Gaussian on intercept
    w_prior: float = 0.2           # prior inclusion probability

    # Sampler
    n_skel: int    = 50_000
    n_resample: int = 10_000
    burnin_frac: float = 0.5
    refresh_rate: float = 1.0
    t_max_zz: float = 0.1
    gamma_zz: float = 0.01
    thinning: str  = "pli"

    # Simulation study
    n_sim: int = 100
    seed: int  = 42


BETA_TRUE = np.array([
    -0.48, 8.72, -1.76, -1.87, 0.0,
     0.0,  0.0,  0.0,   4.00, 0.0,
     0.0,  0.0,  0.0,   0.0,  0.0,
])

# Paper reference numbers for Table 2
PAPER_REFERENCE = {
    "BAS-eplogp":  {"pi_rmse_mean": 1.10, "mu_rmse": 6.75},
    "BAS-uniform": {"pi_rmse_mean": 1.12, "mu_rmse": 6.01},
    "MC³":         {"pi_rmse_mean": 8.50, "mu_rmse": 28.56},
    "RS-Thin":     {"pi_rmse_mean": 1.20, "mu_rmse": 5.93},
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(rng: np.random.Generator, cfg: Config):
    X = rng.normal(size=(cfg.n, cfg.p))
    # Column 9 (index 8) correlated ~rho with column 2 (index 1)
    X[:, 8] = cfg.rho * X[:, 1] + np.sqrt(1 - cfg.rho ** 2) * rng.normal(size=cfg.n)
    X = X - X.mean(0)
    y = cfg.alpha_true + X @ BETA_TRUE + rng.normal(scale=cfg.sigma, size=cfg.n)
    return X, y


# ---------------------------------------------------------------------------
# Exact g-prior posterior (enumerate 2^p models)
# ---------------------------------------------------------------------------

def exact_g_prior_posterior(X: np.ndarray, y: np.ndarray,
                             g: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (pi, mu_bma): marginal inclusion probs [p] and BMA fitted values [n]."""
    n, p = X.shape
    y_c = y - y.mean()
    X_c = X - X.mean(0)
    SST = y_c @ y_c

    n_models = 2 ** p
    log_mls = np.empty(n_models)
    gammas  = np.zeros((n_models, p), dtype=bool)

    for k in range(n_models):
        gamma = np.array([(k >> j) & 1 for j in range(p)], dtype=bool)
        gammas[k] = gamma
        idx = np.where(gamma)[0]
        p_g = len(idx)
        if p_g == 0:
            R2 = 0.0
        else:
            Xg = X_c[:, idx]
            beta_ols = np.linalg.lstsq(Xg, y_c, rcond=None)[0]
            R2 = float(np.clip((Xg @ beta_ols) @ y_c / SST, 0.0, 1.0 - 1e-15))
        log_mls[k] = ((n - p_g - 1) / 2.0) * np.log1p(g) \
                   - ((n - 1)     / 2.0) * np.log1p(g * (1 - R2))

    log_mls -= log_mls.max()
    probs = np.exp(log_mls)
    probs /= probs.sum()

    pi = (probs[:, None] * gammas).sum(0)

    # BMA fitted values (centred + intercept = ybar)
    mu = np.full(n, y.mean())
    for k in range(n_models):
        if probs[k] < 1e-15:
            continue
        idx = np.where(gammas[k])[0]
        if len(idx) == 0:
            continue
        Xg = X_c[:, idx]
        beta_ols = np.linalg.lstsq(Xg, y_c, rcond=None)[0]
        mu += probs[k] * (g / (1 + g)) * (Xg @ beta_ols)

    return pi, mu


# ---------------------------------------------------------------------------
# Target builder
# ---------------------------------------------------------------------------

def make_target(X: np.ndarray, y: np.ndarray, cfg: Config) -> TorchTarget:
    """Build TorchTarget with the exact Zellner g-prior.

    The slab covariance is g * sigma^2 * (X'X)^{-1}, so the precision matrix
    for the weight block is (X'X) / (g * sigma^2). This is passed as a dense
    [D, D] matrix to ModuleGaussianPrior.

    The intercept gets an independent wide Gaussian (1/bias_prior_std^2) on
    the diagonal — it is integrated out analytically in the exact posterior,
    but we keep it as a sampled coordinate so the sampler dimension matches.

    ParamSpec for nn.Linear orders parameters as [weight (p,), bias (1,)].
    The full precision matrix is therefore block-diagonal:
        prec = diag( (X'X)/(g*sigma^2),  1/bias_prior_std^2 )
    """
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    net  = LinearRegressionNet(cfg.p)
    spec = ParamSpec.from_module(net)
    D    = spec.D  # = p + 1 (weights then bias)

    # Exact g-prior precision for the weight block
    X_c   = X - X.mean(0)
    XtX   = torch.tensor(X_c.T @ X_c)                    # [p, p]
    prec_w = XtX / (cfg.g * cfg.sigma ** 2)              # [p, p]

    # Full precision matrix: weights block + intercept scalar
    prec = torch.zeros(D, D, dtype=torch.float64)
    prec[:cfg.p, :cfg.p] = prec_w
    prec[cfg.p, cfg.p]   = 1.0 / cfg.bias_prior_std ** 2

    prior      = ModuleGaussianPrior(prec)
    likelihood = ModuleGaussianLikelihood(net, spec, X_t, y_t, cfg.sigma)
    model      = BayesianModel(prior, likelihood)

    x_ref, Sigma_inv = find_reference_glm(model.energy, spec.D, diagonal_only=False, lr=5e-3, n_steps=10_000)
    return TorchTarget(
        f"clyde_p{cfg.p}", spec.D, model.grad_energy, x_ref, Sigma_inv,
        meta={"model": model, "spec": spec},
    )


# ---------------------------------------------------------------------------
# kappa from spike-and-slab / g-prior calibration
# ---------------------------------------------------------------------------

def make_kappa(spec: ParamSpec, cfg: Config) -> torch.Tensor:
    """kappa = (w/(1-w)) * 1/(tau * sqrt(2pi)) for weights, large for intercept."""
    tau = np.sqrt(cfg.g / cfg.n * cfg.sigma ** 2)
    kappa_weight = (cfg.w_prior / (1 - cfg.w_prior)) / (tau * np.sqrt(2 * np.pi))
    kappa = torch.full((spec.D,), kappa_weight, dtype=torch.float64)
    # intercept is the bias — large kappa so it's effectively never frozen
    kappa[spec.D - 1] = 1e6   # bias is last before rotation
    # After [weight..., bias] -> [bias, weight...] rotation in run_one:
    # we build kappa pre-rotation so rotate here too
    kappa = torch.cat([kappa[-1:], kappa[:-1]])
    return kappa


# ---------------------------------------------------------------------------
# Run one sampler on one dataset
# ---------------------------------------------------------------------------

def run_one(sampler_name: str, target: TorchTarget, cfg: Config,
            seed: int) -> np.ndarray:
    """Run one sticky sampler; return samples [S, D] in [bias, weight...] order."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    spec  = target.meta["spec"]
    kappa = make_kappa(spec, cfg)

    # kappa is already rotated; x_ref/Sigma_inv are in [weight..., bias] order
    # from find_reference_glm. Rotate x_ref for x0.
    x_ref_rot = torch.cat([target.x_ref[-1:], target.x_ref[:-1]])

    common = dict(grad_target=target.grad_target, D=target.D, thinning=cfg.thinning)

    if sampler_name == "sticky_boomerang":
        s = StickyAutomaticBoomerangSampler(
            **common, refresh_rate=cfg.refresh_rate, kappa=kappa, cold_start_threshold=1e-4
        )
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
        res = s.sample(N=cfg.n_skel, x0=target.x_ref.clone(), diagnostics=False)
        x_ref_np = target.x_ref.cpu().numpy()
        samples = resample_boomerang_path_sticky(
            res["positions"].cpu().numpy(),
            res["velocities"].cpu().numpy(),
            res["times"].cpu().numpy(),
            x_ref_np,
            N_resample=cfg.n_resample,
            burnin_frac=cfg.burnin_frac,
        )
    elif sampler_name == "sticky_zigzag":
        s = StickyAutomaticZigZagSampler(
            **common, t_max=cfg.t_max_zz, gamma=cfg.gamma_zz, kappa=kappa, cold_start_threshold=1e-4
        )
        res = s.sample(N=cfg.n_skel, x0=x_ref_rot.clone(), diagnostics=False)
        samples = resample_zigzag_path_sticky(
            res["positions"].cpu().numpy(),
            res["velocities"].cpu().numpy(),
            res["times"].cpu().numpy(),
            N_resample=cfg.n_resample,
            burnin_frac=cfg.burnin_frac,
        )
    else:
        raise ValueError(sampler_name)

    # Rotate from [weight..., bias] -> [bias, weight...] to match NUTS convention
    return np.concatenate([samples[:, -1:], samples[:, :-1]], axis=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def inclusion_probs_from_samples(samples: np.ndarray,
                                  threshold: float = 1e-8) -> np.ndarray:
    """Fraction of samples where |beta_j| > threshold. Excludes intercept (index 0)."""
    return (np.abs(samples[:, 1:]) > threshold).mean(0)


def bma_fitted(samples: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """BMA posterior mean of fitted values: ybar + X_c @ E[beta | y]."""
    X_c = X - X.mean(0)
    # samples[:, 0] is the intercept, samples[:, 1:] are the weights
    beta_mean = samples[:, 1:].mean(0)
    return y.mean() + X_c @ beta_mean


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

SAMPLER_NAMES = ("sticky_boomerang", "sticky_zigzag")
SAMPLER_LABELS = {
    "sticky_boomerang": "Sticky Boomerang",
    "sticky_zigzag":    "Sticky ZigZag",
}


def run_simulation(cfg: Config, save_dir: Path | None = None):
    master_rng = np.random.default_rng(cfg.seed)

    # Storage: [n_sim, p] squared errors for pi, [n_sim] MSE for mu
    pi_se  = {nm: np.zeros((cfg.n_sim, cfg.p)) for nm in SAMPLER_NAMES}
    mu_mse = {nm: np.zeros(cfg.n_sim)          for nm in SAMPLER_NAMES}
    pi_true_all = np.zeros((cfg.n_sim, cfg.p))
    wall_all    = {nm: np.zeros(cfg.n_sim)     for nm in SAMPLER_NAMES}

    for s in range(cfg.n_sim):
        seed_s = int(master_rng.integers(0, 10**6))
        rng_s  = np.random.default_rng(seed_s)

        X, y = generate_data(rng_s, cfg)

        print(f"\n[{s+1:3d}/{cfg.n_sim}] seed={seed_s}  computing exact posterior...")
        t0 = time.perf_counter()
        pi_true, mu_true = exact_g_prior_posterior(X, y, cfg.g)
        pi_true_all[s] = pi_true
        print(f"  exact done in {time.perf_counter()-t0:.1f}s")

        target = make_target(X, y, cfg)

        for nm in SAMPLER_NAMES:
            print(f"  [{SAMPLER_LABELS[nm]}]", end="", flush=True)
            t0 = time.perf_counter()
            samples = run_one(nm, target, cfg, seed=seed_s)
            wall = time.perf_counter() - t0
            wall_all[nm][s] = wall

            pi_hat = inclusion_probs_from_samples(samples)
            mu_hat = bma_fitted(samples, X, y)

            pi_se[nm][s]  = (pi_hat - pi_true) ** 2
            mu_mse[nm][s] = float(np.mean((mu_hat - mu_true) ** 2))
            print(f"  wall={wall:.1f}s  "
                  f"RMSE(pi)×10²={np.sqrt(pi_se[nm][s].mean())*100:.2f}  "
                  f"RMSE(mu)×10³={np.sqrt(mu_mse[nm][s])*1000:.2f}")

    # Aggregate
    print_table(pi_se, mu_mse, wall_all, pi_true_all, cfg)

    if save_dir is not None:
        save_results(save_dir, pi_se, mu_mse, wall_all, pi_true_all, cfg)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(pi_se, mu_mse, wall_all, pi_true_all, cfg: Config):
    mean_pi_true = pi_true_all.mean(0)
    order = np.argsort(mean_pi_true)

    print(f"\n{'='*70}")
    print(f"Table 2 reproduction — RMSE over {cfg.n_sim} simulations")
    print(f"(RMSE × 10² for π_j, RMSE × 10³ for μ)\n")

    header = f"{'Variable':>10}  {'True π':>7}  {'β_true':>7}"
    for nm in SAMPLER_NAMES:
        header += f"  {SAMPLER_LABELS[nm]:>18}"
    print(header)
    print("-" * len(header))

    for j in order:
        row = f"  γ_{j+1:>2d}      {mean_pi_true[j]:>7.3f}  {BETA_TRUE[j]:>7.2f}"
        for nm in SAMPLER_NAMES:
            rmse_pi = float(np.sqrt(pi_se[nm][:, j].mean())) * 100
            row += f"  {rmse_pi:>18.2f}"
        print(row)

    print("-" * len(header))
    row = f"  {'μ':>10}  {'—':>7}  {'—':>7}"
    for nm in SAMPLER_NAMES:
        rmse_mu = float(np.sqrt(mu_mse[nm].mean())) * 1000
        row += f"  {rmse_mu:>18.2f}"
    print(row)

    row = f"  {'wall (s)':>10}  {'—':>7}  {'—':>7}"
    for nm in SAMPLER_NAMES:
        row += f"  {wall_all[nm].mean():>18.1f}"
    print(row)

    print(f"\nPaper reference (Table 2, approximate averages):")
    for label, vals in PAPER_REFERENCE.items():
        print(f"  {label:<15}  RMSE(π)×10² ≈ {vals['pi_rmse_mean']:.2f}  "
              f"RMSE(μ)×10³ ≈ {vals['mu_rmse']:.2f}")


def save_results(save_dir: Path, pi_se, mu_mse, wall_all, pi_true_all, cfg: Config):
    import json
    from dataclasses import asdict
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "pi_true_all.npy", pi_true_all)
    for nm in SAMPLER_NAMES:
        np.save(save_dir / f"pi_se_{nm}.npy", pi_se[nm])
        np.save(save_dir / f"mu_mse_{nm}.npy", mu_mse[nm])
        np.save(save_dir / f"wall_{nm}.npy", wall_all[nm])
    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"\nSaved -> {save_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-sim",      type=int,   default=100)
    p.add_argument("--n-skel",     type=int,   default=50_000)
    p.add_argument("--n-resample", type=int,   default=10_000)
    p.add_argument("--thinning",   choices=["pli", "brent"], default="pli")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--save",       action="store_true")
    args = p.parse_args()

    cfg = Config(
        n_sim=args.n_sim,
        n_skel=args.n_skel,
        n_resample=args.n_resample,
        thinning=args.thinning,
        seed=args.seed,
    )

    print(f"Clyde benchmark  n={cfg.n} p={cfg.p} g={cfg.g} "
          f"n_sim={cfg.n_sim} n_skel={cfg.n_skel}")
    print(f"tau = {np.sqrt(cfg.g/cfg.n*cfg.sigma**2):.4f}  "
          f"w = {cfg.w_prior}  "
          f"kappa ≈ {cfg.w_prior/(1-cfg.w_prior)/(np.sqrt(cfg.g/cfg.n*cfg.sigma**2)*np.sqrt(2*np.pi)):.4f}")

    save_dir = Path(f"results/clyde_benchmark/nsim{cfg.n_sim}_skel{cfg.n_skel}_seed{cfg.seed}") \
               if args.save else None

    run_simulation(cfg, save_dir)


if __name__ == "__main__":
    main()
