"""Shared helpers for GLM benchmark scripts."""

from __future__ import annotations

import time
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch

from sazz.samplers.AutomaticBoomerangSampler import AutomaticBoomerangSampler
from sazz.samplers.StickyAutomaticBoomerangSampler import StickyAutomaticBoomerangSampler
from sazz.samplers.AutomaticZigZagSampler import AutomaticZigZagSampler
from sazz.samplers.StickyAutomaticZigZagSampler import StickyAutomaticZigZagSampler
from sazz.utils.sampling import (
    resample_boomerang_path, resample_boomerang_path_sticky,
    resample_zigzag_path, resample_zigzag_path_sticky,
)

PDMP_SPECS = [
    ("Boom",        "boomerang", False, "C0", "o"),
    ("Sticky-Boom", "boomerang", True,  "C1", "s"),
    ("ZZ",          "zigzag",    False, "C9", "v"),
    ("Sticky-ZZ",   "zigzag",    True,  "C3", "^"),
]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_sampler(family, sticky, target, cfg, kappa):
    common = dict(grad_target=target.grad_target, D=target.D, thinning=cfg.thinning)
    if family == "boomerang":
        s = (StickyAutomaticBoomerangSampler(**common, refresh_rate=cfg.refresh_rate, kappa=kappa)
             if sticky else AutomaticBoomerangSampler(**common, refresh_rate=cfg.refresh_rate))
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
    else:
        s = (StickyAutomaticZigZagSampler(**common, t_max=cfg.t_max_zz, gamma=cfg.gamma_zz, kappa=kappa)
             if sticky else AutomaticZigZagSampler(**common, t_max=cfg.t_max_zz, gamma=cfg.gamma_zz))
    return s


def resample(family, sticky, res, x_ref_np, cfg):
    pos = res["positions"].cpu().numpy()
    vel = res["velocities"].cpu().numpy()
    tim = res["times"].cpu().numpy()
    kw = dict(N_resample=cfg.n_resample, burnin_frac=cfg.burnin_frac)
    if family == "boomerang":
        fn = resample_boomerang_path_sticky if sticky else resample_boomerang_path
        return fn(pos, vel, tim, x_ref_np, **kw)
    fn = resample_zigzag_path_sticky if sticky else resample_zigzag_path
    return fn(pos, vel, tim, **kw)


def run_pdmps(target, cfg, kappa=None) -> list[dict]:
    if kappa is None:
        kappa = torch.ones(target.D, dtype=torch.float64) * cfg.kappa_null
        kappa[0] = cfg.kappa_int
    x_ref_np = target.x_ref.cpu().numpy()

    out = []
    for name, family, sticky, color, marker in PDMP_SPECS:
        set_seed(cfg.seed)
        s = build_sampler(family, sticky, target, cfg, kappa)
        t0 = time.perf_counter()
        res = s.sample(N=cfg.n_skel, x0=target.x_ref.clone(), diagnostics=False)
        samples = resample(family, sticky, res, x_ref_np, cfg)
        samples = np.concatenate([samples[:, -1:], samples[:, :-1]], axis=1)
        wall = time.perf_counter() - t0
        out.append(dict(name=name, samples=samples, wall=wall,
                        sticky=sticky, color=color, marker=marker))
        print(f"  {name:<13} wall={wall:6.2f}s  draws={samples.shape[0]}")
    return out


def _support_via_ci(samples, alpha=0.05):
    lo = np.quantile(samples, alpha / 2,     axis=0)
    hi = np.quantile(samples, 1 - alpha / 2, axis=0)
    return (lo > 0) | (hi < 0)


def print_table(rows: list[dict], metric_key: str = "pred_rmse", metric_label: str = "pred-RMSE"):
    h = (f"{'sampler':<16} {'wall':>7} {'β-RMSE':>9} {metric_label:>10} "
         f"{'σ-ratio':>9} {'F1(95% CI)':>11} {'P(=0) nulls':>12}")
    print("\n" + h)
    print("-" * len(h))
    for r in rows:
        wall = f"{r['wall']:>7.2f}" if r.get("wall") is not None else f"{'—':>7}"
        sr   = f"{r['sigma_ratio']:>9.3f}" if not np.isnan(r["sigma_ratio"]) else f"{'—':>9}"
        p0   = f"{r['p0_nulls']:>12.3f}"   if not np.isnan(r["p0_nulls"])    else f"{'—':>12}"
        mv   = f"{r[metric_key]:>10.4f}"
        print(f"{r['name']:<16} {wall} {r['beta_rmse']:>9.4f} {mv} {sr} {r['f1_ci']:>11.3f} {p0}")


def plot_coefs(rows, true_coefs, is_signal, title, out_path=None):
    import matplotlib.pyplot as plt
    sparse = (~is_signal).sum() > len(true_coefs) // 2
    sig = np.where(is_signal)[0]
    nul = np.where(~is_signal)[0]
    offsets = np.linspace(-0.30, 0.30, len(rows))

    # coordinate labels with math subscripts: β_0, β_1, ...
    coord_labels = [rf"$\beta_{{{i}}}$" for i in range(len(true_coefs))]

    if not sparse:
        fig, ax = plt.subplots(figsize=(min(13, 1.0 * len(true_coefs) + 4), 4.5))
        idx = np.arange(len(true_coefs))
        for r, off in zip(rows, offsets):
            mu, sd = r["samples"].mean(0), r["samples"].std(0)
            ax.errorbar(idx + off, mu, yerr=2 * sd, fmt=r["marker"], color=r["color"],
                        label=r["name"], capsize=2, markersize=4, lw=1)
        ax.scatter(idx, true_coefs, marker="x", color="k", s=70, linewidths=2, label="true", zorder=5)
        for i in sig:
            ax.axvspan(i - 0.45, i + 0.45, color="gold", alpha=0.12)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(idx)
        ax.set_xticklabels(coord_labels)
        ax.set(ylabel="coefficient", title=title)
        ax.legend(fontsize=8, ncol=2, frameon=False)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                        gridspec_kw={"height_ratios": [1.2, 1]})
        xs = np.arange(len(sig))
        for r, off in zip(rows, offsets):
            mu, sd = r["samples"].mean(0), r["samples"].std(0)
            ax1.errorbar(xs + off, mu[sig], yerr=2 * sd[sig], fmt=r["marker"],
                         color=r["color"], label=r["name"], capsize=2, markersize=4, lw=1)
            ax2.scatter(np.arange(len(nul)), mu[nul], color=r["color"], marker=r["marker"],
                        s=15, alpha=0.6, label=r["name"])
        ax1.scatter(xs, true_coefs[sig], marker="x", color="red", s=60,
                    linewidths=2, label="true", zorder=5)
        ax1.axhline(0, color="grey", lw=0.5)
        ax1.set_xticks(xs)
        ax1.set_xticklabels([coord_labels[i] for i in sig], rotation=45, ha="right")
        ax1.set(ylabel="coefficient", title=f"Signals — {title}")
        ax1.legend(fontsize=8, ncol=3, frameon=False)
        ax2.axhline(0, color="red", lw=1)
        ax2.set_xticks(np.arange(len(nul)))
        ax2.set_xticklabels([coord_labels[i] for i in nul], rotation=45, ha="right", fontsize=7)
        ax2.set(ylabel="posterior mean",
                title=f"Nulls ({len(nul)} of {len(true_coefs)})")
        ax2.legend(fontsize=8, ncol=3, frameon=False)

    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=120)
    plt.show()


def _to_jsonable(v):
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return None if np.isnan(v) else v
    if isinstance(v, np.integer):   return int(v)
    if isinstance(v, np.floating):
        f = float(v); return None if np.isnan(f) else f
    if isinstance(v, np.ndarray):   return v.tolist()
    if isinstance(v, torch.Tensor): return v.detach().cpu().tolist()
    if isinstance(v, (list, tuple)):return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):         return {str(k): _to_jsonable(x) for k, x in v.items()}
    return None


def save_run(out_dir: Path, cfg, rows, metric_keys, true_coefs, is_signal):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = _to_jsonable(asdict(cfg))
    for r in rows:
        if "samples" not in r:
            continue
        torch.save({
            "name": r["name"], "samples": torch.tensor(r["samples"]),
            "wall": r.get("wall"), "config": cfg_dict,
        }, out_dir / f"{r['name'].replace(' ', '_')}.pt")
    metric_rows = [{k: _to_jsonable(r.get(k)) for k in metric_keys} for r in rows]
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "config":     cfg_dict,
            "true_coefs": true_coefs.tolist(),
            "is_signal":  [bool(b) for b in is_signal.tolist()],
            "rows":       metric_rows,
        }, f, indent=2)
    print(f"\nSaved -> {out_dir}/")
