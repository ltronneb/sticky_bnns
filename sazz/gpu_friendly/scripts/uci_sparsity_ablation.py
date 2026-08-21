"""
Prune -> refit -> sample pipeline for the UCI sticky grid samplers, mirroring
fast_mnist_cnn.py / refit_pruned_lenet_reference.py's approach for the MNIST
CNN, applied to the small UCI regression BNNs.

Motivation: uci_bnn_grid.py never prunes x_ref -- the sticky samplers start
fully thawed (GRID_STICKY_COLD_START_THRESHOLD=None) and sparsity is earned
live via kappa (from prior_inclusion_weight). Investigating this surfaced
that the plain-Gaussian MAP for e.g. Boston/small is already ~67% sparse
(most freezable coordinates sit at a genuine gradient-zero of bm.energy --
confirmed directly, not merely "small"), while the sticky samplers' own
final sparsity in existing runs is only ~20-35%. kappa is uniform per layer
(build_kappa_from_inclusion depends only on prior_std_weight/
prior_inclusion_weight/fan_in, never on Sigma_inv or which specific
coordinates the MAP found dead) and Sigma_inv is floored at prior_precision
(diag_prec = prior_prec + fisher_diag, fisher_diag >= 0), so neither channel
currently tells the sticky dynamics "the MAP already identified these
specific coordinates as dead." This script tests whether directly informing
the samplers of that MAP-level sparsity (via cold_start_threshold =
cold_start_mask, seeded from an accuracy-gated prune+refit of the MAP)
produces meaningfully higher final sampler sparsity than the current
unpruned-MAP baseline -- if so, that is a real speed/quality improvement
worth re-running existing UCI results with.

Pipeline, run fresh per (dataset, split) -- a new split means a new MAP,
not a cached one:
  1. fit x_ref via find_reference_bnn (same as uci_bnn_grid.py -- one MAP,
     one diagonal Laplace Sigma_inv, both computed once per split),
  2. prune_to_tolerance: magnitude-threshold x_ref relative to each
     coordinate's own prior std, restricted to can_freeze (biases/noise
     never prune) -- same shape as fast_mnist_cnn.py::prune_x_ref, sweeping
     candidate thresholds and picking the LARGEST sparsity whose RMSE stays
     within --prune-rmse-tolerance of the unpruned baseline (default 0.01,
     matching PRUNE_ACC_DROP_TOLERANCE=0.01 there),
  3. refit_active_coords: masked-gradient Adam repairs stationarity in the
     surviving (active) subspace only -- copied verbatim from
     refit_pruned_lenet_reference.py, already target-agnostic,
  4. save the pruned+refit reference point (+ cold_start_mask, Sigma_inv,
     RMSE at MAP/pruned/refit stages) to <out-maps>/ for reuse,
  5. run grid_sticky_zigzag / grid_sticky_boomerang from that single
     reference point, cold_start_threshold=cold_start_mask (freezes exactly
     the pruned coordinates, not a uniform absolute threshold -- see
     GridStickyBoomerangSampler's docstring).

Boomerang's x_ref is baked into its dynamics on every step (trajectory/
grad_U_excess use self.x_ref, not just at init), so the SAME pruned+refit
vector must reach preprocess(x_ref=...), sample()'s implicit start point,
and the sticky resample call -- exactly the coupling fast_mnist_cnn.py's
module docstring warns about. This script threads x_refit through all of
those from a single local variable, never re-deriving it.

uci_bnn_grid.py / uci_bnn_grid_skeleton.py are untouched -- this script
duplicates the ~15-line sticky sampler builder bodies locally (instead of
importing uci_bnn_grid.py's, which hard-code
cold_start_threshold=GRID_STICKY_COLD_START_THRESHOLD=None) so it can pass
a MAP-derived mask without modifying that module.

Usage:
    python -m sazz.gpu_friendly.scripts.uci_sparsity_ablation \\
        --datasets boston --splits 0 --hidden-variant small

    python -m sazz.gpu_friendly.scripts.uci_sparsity_ablation \\
        --datasets boston --splits 0 --hidden-variant deep_narrow

    # cheap smoke test
    python -m sazz.gpu_friendly.scripts.uci_sparsity_ablation \\
        --datasets boston --splits 0 --n-skeleton 500 --refit-n-steps 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.neural_networks import FFN
from sazz.gpu_friendly.models.priors import (
    build_fan_in_prior_precision, build_kappa_from_inclusion, build_can_freeze_mask,
)
from sazz.gpu_friendly.samplers.grid_sticky_zigzag import GridStickyZigZagSampler
from sazz.gpu_friendly.samplers.grid_sticky_boomerang import GridStickyBoomerangSampler
from sazz.gpu_friendly.utils.resample import (
    resample_zigzag_path_sticky_torch, resample_boomerang_path_sticky_torch,
)
from sazz.gpu_friendly.scripts.uci_bnn_grid import (
    DEVICE, DTYPE, N_SKELETON, BURNIN_FRAC, BASE_SEED,
    HIDDEN_VARIANTS, DEFAULT_HIDDEN_VARIANT, UCI_DATASETS,
    SIGMA_INV_SCALE, REFRESH_RATE, GAMMA,
    GRID_N_SEGMENTS, GRID_T_MAX_INIT_BOOM, GRID_T_MAX_INIT_ZIGZAG,
    GRID_ALPHA_PLUS, GRID_ALPHA_MINUS,
    GRID_STICKY_ZIGZAG_SPACING, GRID_STICKY_BOOM_SPACING,
    BNNConfig, configs_for, load_raw_datasets, make_split, build_target,
)

N_RESAMPLE = 50_000
N_SAVE = 5_000

PRUNE_N_THRESHOLDS = 60
PRUNE_RMSE_TOLERANCE = 0.01
REFIT_N_STEPS = 200
REFIT_LR = 1e-3

SAMPLER_NAMES = ("grid_sticky_zigzag", "grid_sticky_boomerang")

OUT_MAPS_DIR = Path("results/maps/uci_sparsity_ablation")
OUT_DIR = Path("results/grid/uci_sparsity_ablation")


# ===========================================================================
# bm construction WITHOUT the expensive find_reference_bnn MAP fit -- used
# only on the checkpoint-hit path in run_split, where x_ref/Sigma_inv are
# loaded from a saved pruned+refit checkpoint instead of being refit. Same
# module/prior-precision/BayesianModule.build sequence as
# uci_bnn_grid.py::build_target's first half (that function fits bm
# construction and find_reference_bnn together with no way to opt out of
# the fit, so this is duplicated here rather than importing it).
# ===========================================================================

def build_bm_only(data: dict[str, Any], cfg: BNNConfig, dtype=DTYPE, device=DEVICE) -> BayesianModule:
    X = data["X_train"].to(dtype=dtype, device=device)
    y = data["y_train"].to(dtype=dtype, device=device)
    module = FFN(cfg.layer_sizes, cfg.activation)
    prec = build_fan_in_prior_precision(
        module, cfg.prior_std_weight, cfg.prior_std_bias, cfg.fan_in_scaling, dtype=dtype, device=device,
    )
    return BayesianModule.build(
        module, likelihood="gaussian", X=X, y=y,
        prior_precision=prec, prior_sigma_scale=cfg.prior_sigma_scale, dtype=dtype, device=device,
    )


# ===========================================================================
# Regression eval (RMSE, not accuracy -- cnn_reference.py::eval_accuracy is
# classification-only)
# ===========================================================================

@torch.no_grad()
def eval_rmse(bm: BayesianModule, beta: Tensor, X: Tensor, y: Tensor, y_std: float = 1.0) -> float:
    """
    RMSE in original y-units. bm.param_dict_fn only unflattens the network
    parameters (D_network long) -- when noise is learned (always true in
    this script, same as every sampler in uci_bnn_grid.py), beta carries a
    trailing log_sigma coordinate that must be sliced off first, exactly
    as grid_uci_investigate.ipynb's own MAP-prediction cell does
    (`weights_ref = x_ref[:-1]` before `bm.param_dict_fn(weights_ref)`).
    """
    weights = beta[:-1] if bm.learns_noise else beta
    pred = torch.func.functional_call(bm.module, bm.param_dict_fn(weights), (X,)).squeeze(-1)
    return (((pred - y) ** 2).mean().sqrt() * y_std).item()


# ===========================================================================
# Prune -- same shape/criterion as fast_mnist_cnn.py::prune_x_ref: sweep
# log-spaced threshold multipliers (per-coordinate, relative to that
# coordinate's own prior std, restricted to can_freeze), pick the LARGEST
# sparsity whose RMSE stays within rmse_tolerance of the unpruned baseline.
# ===========================================================================

def prune_to_tolerance(bm: BayesianModule, x_ref: Tensor, X_eval: Tensor, y_eval: Tensor,
                        y_std: float, can_freeze: Tensor, rmse_tolerance: float = PRUNE_RMSE_TOLERANCE,
                        n_thresholds: int = PRUNE_N_THRESHOLDS) -> tuple[Tensor, Tensor, list[dict], float]:
    prior_std = bm.prior_precision.clamp(min=1e-12).rsqrt()
    baseline_rmse = eval_rmse(bm, x_ref, X_eval, y_eval, y_std)

    sweep_multipliers = np.logspace(-3, 0, n_thresholds)
    sweep_log: list[dict] = []
    best_frac, best_mask, best_rmse, best_mult = 0.0, torch.zeros_like(can_freeze), baseline_rmse, 0.0

    for m in sweep_multipliers:
        mask = (x_ref.abs() < m * prior_std) & can_freeze
        x_pruned = torch.where(mask, torch.zeros_like(x_ref), x_ref)
        achieved_frac = float(mask.float().mean())
        rmse = eval_rmse(bm, x_pruned, X_eval, y_eval, y_std)
        sweep_log.append({"multiplier": float(m), "achieved_frac": achieved_frac, "rmse": rmse})

        if rmse <= baseline_rmse + rmse_tolerance and achieved_frac >= best_frac:
            best_frac, best_mask, best_rmse, best_mult = achieved_frac, mask, rmse, float(m)

    x_pruned = torch.where(best_mask, torch.zeros_like(x_ref), x_ref)
    n_freezable = int(can_freeze.sum())
    n_pruned = int(best_mask.sum())
    print(f"  prune_to_tolerance: threshold={best_mult:.4f}*std  "
          f"pruned {n_pruned}/{n_freezable} freezable coords ({best_frac:.3f} of freezable, "
          f"{n_pruned / bm.D:.3f} of all D)  params kept={bm.D - n_pruned}  "
          f"RMSE: baseline={baseline_rmse:.4f} pruned={best_rmse:.4f} "
          f"(delta={best_rmse - baseline_rmse:.4f}, tolerance={rmse_tolerance:.4f})")
    return x_pruned, best_mask, sweep_log, baseline_rmse


def refit_active_coords(bm: BayesianModule, x_pruned: Tensor, frozen_mask: Tensor,
                         n_steps: int = REFIT_N_STEPS, lr: float = REFIT_LR) -> Tensor:
    """
    Copied verbatim (structure) from
    refit_pruned_lenet_reference.py::refit_active_coords -- masked-gradient
    Adam on bm.energy, frozen coordinates held at exactly 0 throughout, no
    CNN-specific code (operates purely on bm.energy/bm.D), so this works
    unchanged for the Gaussian-likelihood UCI BNNs.
    """
    active_mask = ~frozen_mask
    beta = x_pruned.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([beta], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.1)

    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = bm.energy(beta)
        loss.backward()
        with torch.no_grad():
            beta.grad.mul_(active_mask.to(beta.grad.dtype))
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        beta[frozen_mask] = 0.0  # cheap insurance against float drift; not load-bearing
    return beta.detach()


# ===========================================================================
# Single pruned+refit reference-point construction
# ===========================================================================

def build_pruned_checkpoint(
    bm: BayesianModule, x_ref: Tensor, Sigma_inv: Tensor, can_freeze: Tensor,
    data: dict[str, Any], cfg: BNNConfig, dataset_name: str, split_id: int,
    rmse_tolerance: float, refit_n_steps: int, refit_lr: float, out_maps_dir: Path,
) -> dict:
    X_test, y_test, y_std = data["X_test"], data["y_test"], data["y_std"]
    grad_target = torch.func.grad(bm.energy)

    grad_norm_map = grad_target(x_ref).abs().max().item()

    x_pruned, mask, sweep_log, rmse_map = prune_to_tolerance(
        bm, x_ref, X_test, y_test, y_std, can_freeze, rmse_tolerance,
    )
    rmse_pruned = eval_rmse(bm, x_pruned, X_test, y_test, y_std)
    grad_norm_pruned = grad_target(x_pruned).abs().max().item()
    print(f"  ||grad_target||_inf: MAP={grad_norm_map:.4e}  pruned (before refit)={grad_norm_pruned:.4e}")

    x_refit = refit_active_coords(bm, x_pruned, mask, refit_n_steps, refit_lr)
    rmse_refit = eval_rmse(bm, x_refit, X_test, y_test, y_std)
    grad_norm_refit = grad_target(x_refit).abs().max().item()
    achieved_frac = float(mask.float().mean())
    print(f"  RMSE after refit={rmse_refit:.4f}  ||grad_target||_inf after refit={grad_norm_refit:.4e}")

    ckpt = {
        "dataset": dataset_name, "split_id": split_id,
        "x_ref": x_refit.cpu(), "cold_start_mask": mask.cpu(),
        "Sigma_inv": Sigma_inv.cpu(),
        "achieved_sparsity": achieved_frac, "rmse_tolerance": rmse_tolerance,
        "rmse_map": rmse_map, "rmse_pruned": rmse_pruned, "rmse_refit": rmse_refit,
        "grad_norm_inf_map": grad_norm_map,
        "grad_norm_inf_pruned": grad_norm_pruned,
        "grad_norm_inf_refit": grad_norm_refit,
        "sweep_log": sweep_log,
        "layer_sizes": cfg.layer_sizes, "activation": cfg.activation,
        "prior_sigma_scale": cfg.prior_sigma_scale, "y_std": y_std,
    }
    out_path = out_maps_dir / f"{dataset_name}_split{split_id:02d}_pruned_refit.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"  saved -> {out_path}")

    return {
        "achieved_frac": achieved_frac, "x_refit": x_refit, "mask": mask,
        "rmse_map": rmse_map, "rmse_pruned": rmse_pruned, "rmse_refit": rmse_refit,
        "y_std": y_std, "Sigma_inv": Sigma_inv,
    }


def checkpoint_path_for(out_maps_dir: Path, dataset_name: str, split_id: int) -> Path:
    return out_maps_dir / f"{dataset_name}_split{split_id:02d}_pruned_refit.pt"


def load_pruned_checkpoint(out_path: Path, expected_D: int, device=DEVICE) -> Optional[dict]:
    """
    Loads a previously saved pruned+refit checkpoint if it matches the
    current target's D (guards against a stale checkpoint from a different
    architecture/hidden-variant silently being reused -- D differs
    substantially between e.g. "small" [50] and "deep_narrow" [50,50,50]).
    Returns None (falls through to a fresh MAP fit + prune + refit) if the
    file doesn't exist or D doesn't match.
    """
    if not out_path.exists():
        return None
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    x_ref = ckpt["x_ref"].to(dtype=DTYPE, device=device)
    if x_ref.shape[0] != expected_D:
        print(f"  found checkpoint at {out_path} but D mismatch "
              f"(ckpt D={x_ref.shape[0]} vs expected D={expected_D}) -- refitting")
        return None
    print(f"  loaded existing pruned+refit checkpoint -> {out_path} "
          f"(achieved_sparsity={ckpt['achieved_sparsity']:.3f}, "
          f"rmse_map={ckpt['rmse_map']:.4f}, rmse_refit={ckpt['rmse_refit']:.4f})")
    return {
        "achieved_frac": ckpt["achieved_sparsity"],
        "x_refit": x_ref,
        "mask": ckpt["cold_start_mask"].to(dtype=torch.bool, device=device),
        "rmse_map": ckpt["rmse_map"], "rmse_pruned": ckpt["rmse_pruned"], "rmse_refit": ckpt["rmse_refit"],
        "y_std": ckpt["y_std"], "Sigma_inv": ckpt["Sigma_inv"].to(dtype=DTYPE, device=device),
    }


# ===========================================================================
# Sticky sampler builders -- duplicated locally (not imported from
# uci_bnn_grid.py) so cold_start_threshold can be the MAP-derived mask;
# uci_bnn_grid.py's own builders hard-code
# cold_start_threshold=GRID_STICKY_COLD_START_THRESHOLD (module global,
# always None there) and are left untouched.
# ===========================================================================

def _build_sticky_kappa_can_freeze(bm: BayesianModule, cfg: BNNConfig) -> tuple[Tensor, Tensor]:
    kappa_net = build_kappa_from_inclusion(
        bm.module, cfg.prior_std_weight, cfg.prior_inclusion_weight,
        cfg.fan_in_scaling, dtype=DTYPE, device=bm.device,
    )
    can_freeze_net = build_can_freeze_mask(bm.module, device=bm.device)

    if bm.learns_noise:
        kappa = torch.cat([kappa_net, torch.zeros(1, dtype=DTYPE, device=bm.device)])
        can_freeze = torch.cat([can_freeze_net, torch.zeros(1, dtype=torch.bool, device=bm.device)])
    else:
        kappa = kappa_net
        can_freeze = can_freeze_net
    return kappa, can_freeze


def build_sticky_zigzag_sampler(bm: BayesianModule, cfg: BNNConfig, cold_start_mask: Tensor) -> GridStickyZigZagSampler:
    kappa, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
    return GridStickyZigZagSampler(
        grad_target=torch.func.grad(bm.energy), D=bm.D,
        kappa=kappa, can_freeze=can_freeze, cold_start_threshold=cold_start_mask,
        gamma=GAMMA, grid_t_max_init=GRID_T_MAX_INIT_ZIGZAG, n_segments=GRID_N_SEGMENTS,
        grid_spacing=GRID_STICKY_ZIGZAG_SPACING, alpha_plus=GRID_ALPHA_PLUS, alpha_minus=GRID_ALPHA_MINUS,
        dtype=DTYPE, device=bm.device,
    )


def build_sticky_boomerang_sampler(bm: BayesianModule, cfg: BNNConfig, x_ref: Tensor,
                                    Sigma_inv: Tensor, cold_start_mask: Tensor) -> GridStickyBoomerangSampler:
    kappa, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
    sampler = GridStickyBoomerangSampler(
        grad_target=torch.func.grad(bm.energy), D=bm.D,
        kappa=kappa, can_freeze=can_freeze, cold_start_threshold=cold_start_mask,
        grid_spacing=GRID_STICKY_BOOM_SPACING, refresh_rate=REFRESH_RATE,
        grid_t_max_init=GRID_T_MAX_INIT_BOOM, n_segments=GRID_N_SEGMENTS,
        alpha_plus=GRID_ALPHA_PLUS, alpha_minus=GRID_ALPHA_MINUS,
        dtype=DTYPE, device=bm.device,
    )
    sampler.preprocess(x_ref=x_ref, Sigma_inv=Sigma_inv * SIGMA_INV_SCALE)
    return sampler


# ===========================================================================
# Sampler runners -- same skeleton/save shape as uci_bnn_grid_skeleton.py's
# save_skeleton, so grid_uci_investigate.ipynb's existing sparsity-
# trajectory cells (section 7d) can load these with minimal changes.
# x_refit reaches the sampler builder (-> preprocess -> x_ref for
# Boomerang), sample()'s x0 (ZigZag only -- Boomerang has no x0, it starts
# from self.x_ref + refreshed velocity), AND the resample call below --
# all sites, matching fast_mnist_cnn.py's own three-site requirement.
# ===========================================================================

def save_skeleton(out_path: Path, *, ref: dict, sampler: str, result: dict,
                   samples: Tensor, cfg: BNNConfig, elapsed_sec: float, n_skeleton: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sampler": sampler,
        "samples": samples.cpu(),
        "x_ref": ref["x_refit"].cpu(),
        "cold_start_mask": ref["mask"].cpu(),
        "achieved_sparsity": ref["achieved_frac"],
        "rmse_map": ref["rmse_map"], "rmse_pruned": ref["rmse_pruned"], "rmse_refit": ref["rmse_refit"],
        "layer_sizes": cfg.layer_sizes, "activation": cfg.activation,
        "learned_noise": True, "prior_sigma_scale": cfg.prior_sigma_scale,
        "y_std": ref["y_std"],
        "n_events": n_skeleton, "elapsed_sec": elapsed_sec,
        "bound_violations": result["bound_violations"],
        "gradient_evals": result["gradient_evals"],
        "grid_t_max_log": result["grid_t_max_log"],
        "diagnostics": result.get("diagnostics"),
        "frozen_mask_final": result["frozen_mask_final"].cpu(),
    }, out_path)


def run_grid_sticky_zigzag(ref: dict, cfg: BNNConfig, sd: Path, bm: BayesianModule, n_skeleton: int) -> None:
    out_path = sd / "grid_sticky_zigzag_skeleton.pt"
    x_refit, mask = ref["x_refit"], ref["mask"]

    sampler = build_sticky_zigzag_sampler(bm, cfg, mask)

    n_freezable = int(sampler.can_freeze.sum())
    n_frozen_init = int(mask.sum())
    print(f"      cold start: {n_frozen_init}/{n_freezable} freezable coords "
          f"frozen ({100 * n_frozen_init / max(n_freezable, 1):.1f}%)")

    t0 = time.perf_counter()
    result = sampler.sample(N=n_skeleton, x0=x_refit, diagnostics=True)
    elapsed = time.perf_counter() - t0

    samples = resample_zigzag_path_sticky_torch(
        result["positions"], result["velocities"], result["times"],
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    final_sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {n_skeleton} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, final sparsity {final_sparsity:.2f})")

    save_skeleton(
        out_path, ref=ref, sampler="grid_sticky_zigzag", result=result,
        samples=samples[:N_SAVE] if samples.shape[0] > N_SAVE else samples,
        cfg=cfg, elapsed_sec=elapsed, n_skeleton=n_skeleton,
    )
    print(f"      saved -> {out_path}")


def run_grid_sticky_boomerang(ref: dict, cfg: BNNConfig, sd: Path, bm: BayesianModule,
                               Sigma_inv: Tensor, n_skeleton: int) -> None:
    out_path = sd / "grid_sticky_boomerang_skeleton.pt"
    x_refit, mask = ref["x_refit"], ref["mask"]

    sampler = build_sticky_boomerang_sampler(bm, cfg, x_refit, Sigma_inv, mask)

    n_freezable = int(sampler.can_freeze.sum())
    n_frozen_init = int(mask.sum())
    print(f"      cold start: {n_frozen_init}/{n_freezable} freezable coords "
          f"frozen ({100 * n_frozen_init / max(n_freezable, 1):.1f}%)")

    t0 = time.perf_counter()
    result = sampler.sample(N=n_skeleton, diagnostics=True)
    elapsed = time.perf_counter() - t0

    samples = resample_boomerang_path_sticky_torch(
        result["positions"], result["velocities"], result["times"], x_refit,
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    final_sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {n_skeleton} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, final sparsity {final_sparsity:.2f})")

    save_skeleton(
        out_path, ref=ref, sampler="grid_sticky_boomerang", result=result,
        samples=samples[:N_SAVE] if samples.shape[0] > N_SAVE else samples,
        cfg=cfg, elapsed_sec=elapsed, n_skeleton=n_skeleton,
    )
    print(f"      saved -> {out_path}")


# ===========================================================================
# Per-(dataset, split) driver -- one MAP fit per split (fresh every time,
# never cached across splits), same as uci_bnn_grid.py's run_split.
# ===========================================================================

def run_split(dataset_name: str, split_id: int, data: dict[str, Any], cfg: BNNConfig,
              out_maps_dir: Path, out_dir: Path, rmse_tolerance: float,
              refit_n_steps: int, refit_lr: float, n_skeleton: int,
              resume: bool, refit_maps: bool) -> None:
    print(f"\n--- {dataset_name.upper()} split {split_id:02d} | "
          f"layers={cfg.layer_sizes} | act={cfg.activation} | "
          f"noise=learned (HalfNormal scale={cfg.prior_sigma_scale:.4f}) | "
          f"seed={BASE_SEED + split_id} ---")

    seed = BASE_SEED + split_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    expected_D = sum(a * b + b for a, b in zip(cfg.layer_sizes[:-1], cfg.layer_sizes[1:])) + 1
    ckpt_path = checkpoint_path_for(out_maps_dir, dataset_name, split_id)

    ref = None if refit_maps else load_pruned_checkpoint(ckpt_path, expected_D)
    if ref is not None:
        # Reusing a saved MAP -- still need bm (module/energy/param_dict_fn)
        # for the samplers, but skip the expensive find_reference_bnn fit
        # and the prune/refit stage entirely.
        bm = build_bm_only(data, cfg)
        print(f"  D = {bm.D}")
    else:
        print("\n[prune + refit]")
        bm, x_ref, Sigma_inv = build_target(data, cfg)
        print(f"  D = {bm.D}")
        _, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
        ref = build_pruned_checkpoint(
            bm, x_ref, Sigma_inv, can_freeze, data, cfg, dataset_name, split_id,
            rmse_tolerance, refit_n_steps, refit_lr, out_maps_dir,
        )

    sd = out_dir / dataset_name / f"split_{split_id:02d}"
    sd.mkdir(parents=True, exist_ok=True)
    zz_path = sd / "grid_sticky_zigzag_skeleton.pt"
    boom_path = sd / "grid_sticky_boomerang_skeleton.pt"

    print("\n[sample]")
    if not (resume and zz_path.exists()):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print("  [grid_sticky_zigzag]")
        run_grid_sticky_zigzag(ref, cfg, sd, bm, n_skeleton)
    else:
        print(f"  [grid_sticky_zigzag] skipping — exists at {zz_path}")

    if not (resume and boom_path.exists()):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print("  [grid_sticky_boomerang]")
        run_grid_sticky_boomerang(ref, cfg, sd, bm, ref["Sigma_inv"], n_skeleton)
    else:
        print(f"  [grid_sticky_boomerang] skipping — exists at {boom_path}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=["boston"], choices=list(UCI_DATASETS))
    parser.add_argument("--splits", nargs="+", type=int, default=[0])
    parser.add_argument("--hidden-variant", choices=list(HIDDEN_VARIANTS), default=DEFAULT_HIDDEN_VARIANT)
    parser.add_argument("--prune-rmse-tolerance", type=float, default=PRUNE_RMSE_TOLERANCE,
                         help="Max acceptable (rmse_pruned - rmse_map), in standardized y-units "
                              "(matches PRUNE_ACC_DROP_TOLERANCE=0.01's role in fast_mnist_cnn.py).")
    parser.add_argument("--refit-n-steps", type=int, default=REFIT_N_STEPS)
    parser.add_argument("--refit-lr", type=float, default=REFIT_LR)
    parser.add_argument("--n-skeleton", type=int, default=N_SKELETON)
    parser.add_argument("--prior-inclusion-weight", type=float, default=0.1,
                         help="Sticky-only spike-and-slab prior inclusion probability w, fed to "
                              "build_kappa_from_inclusion (kappa = (w/(1-w))/(sigma_w*sqrt(2pi)), "
                              "kappa is the THAW rate multiplier -- rate_i = kappa_i*|v_i|, so "
                              "expected time-to-thaw = 1/rate_i). Higher w -> higher kappa -> "
                              "FASTER thawing (less sticky); lower w -> longer expected time frozen. "
                              "Default here is 0.1 (NOT configs_for's usual 0.7 default) -- "
                              "smoke-tested on boston/small: w=0.7 drains the cold-started ~72% "
                              "sparsity down to 36-47% within 500 events (kappa too large, "
                              "thaw-dominated); w=0.1 holds ~72% (zigzag, roughly balanced "
                              "freeze/thaw) or grows to ~77% (boomerang, freeze-dominated). "
                              "w=0.7 would defeat the point of cold-starting from a pruned MAP.")
    parser.add_argument("--resume", action="store_true",
                         help="Skip a sampler run if its output .pt already exists.")
    parser.add_argument("--refit-maps", action="store_true",
                         help="Force a fresh MAP fit + prune + refit even if a saved checkpoint "
                              "exists at <out-maps>/<dataset>_split<k>_pruned_refit.pt. Default "
                              "(flag omitted) is to reuse an existing checkpoint when present -- "
                              "the whole point of saving it -- so re-running this script (e.g. "
                              "after tuning --prior-inclusion-weight) does not re-fit the MAP.")
    parser.add_argument("--out-maps", type=Path, default=OUT_MAPS_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    hidden = HIDDEN_VARIANTS[args.hidden_variant]
    out_maps_dir = args.out_maps if args.hidden_variant == DEFAULT_HIDDEN_VARIANT else args.out_maps / args.hidden_variant
    out_dir = args.out if args.hidden_variant == DEFAULT_HIDDEN_VARIANT else args.out / args.hidden_variant
    out_maps_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw UCI datasets...")
    raw = load_raw_datasets(tuple(args.datasets))
    missing = [d for d in args.datasets if d not in raw]
    if missing:
        print(f"  (skipping {missing} -- data file(s) not found)")
    datasets_to_run = [d for d in args.datasets if d in raw]

    cfgs = configs_for({n: X.shape[1] for n, (X, _) in raw.items()}, hidden,
                        prior_inclusion_weight=args.prior_inclusion_weight)

    print(f"\nRunning {datasets_to_run} | splits: {args.splits} | "
          f"hidden_variant={args.hidden_variant} ({hidden}) | "
          f"prune_rmse_tolerance={args.prune_rmse_tolerance} | prior_inclusion_weight={args.prior_inclusion_weight} | "
          f"N_SKELETON={args.n_skeleton}")

    for ds in datasets_to_run:
        X, y = raw[ds]
        cfg = cfgs[ds]
        for split_id in args.splits:
            data = make_split(X, y, seed=BASE_SEED + split_id, dtype=DTYPE, device=DEVICE)
            run_split(
                ds, split_id, data, cfg, out_maps_dir, out_dir,
                args.prune_rmse_tolerance, args.refit_n_steps, args.refit_lr,
                args.n_skeleton, args.resume, args.refit_maps,
            )


if __name__ == "__main__":
    main()
