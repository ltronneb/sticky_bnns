"""

Script for running a sweep across different PIW values
Uses the same configs as uci_bnn_grid.py and runs only the sticky variants.

Usage:
    python -m sazz.gpu_friendly.scripts.uci_sparsity_ablation \\
        --datasets boston --splits 0 --hidden-variant small \\
        --prior-inclusion-weight 0.1 0.3 0.5 0.7

    # cheap smoke test
    python -m sazz.gpu_friendly.scripts.uci_sparsity_ablation \\
        --datasets boston --splits 0 --n-skeleton 500
"""

from __future__ import annotations

import argparse
import time
import math
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
    SIGMA_INV_SCALE, GAMMA, REFRESH_RATE,
    GRID_N_SEGMENTS, GRID_T_MAX_INIT_ZIGZAG, GRID_T_MAX_INIT_BOOM,
    GRID_ALPHA_PLUS, GRID_ALPHA_MINUS,
    GRID_STICKY_ZIGZAG_SPACING, GRID_STICKY_BOOM_SPACING,
    BNNConfig, configs_for, load_raw_datasets, make_split, build_target,
    thin_to,
)

# GRID_T_MAX_INIT_BOOM = 0.002
# GRID_STICKY_BOOM_SPACING = 0.0002 #math.pi / 8
# REFRESH_RATE = 1.0

N_RESAMPLE = 50_000
N_SAVE = 5_000

SAMPLER_NAMES = ("grid_sticky_zigzag", "grid_sticky_boomerang")

OUT_MAPS_DIR = Path("results/maps/uci_sparsity_ablation")
OUT_DIR = Path("results/grid/uci_sparsity_ablation")


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

@torch.no_grad()
def eval_rmse(bm: BayesianModule, beta: Tensor, X: Tensor, y: Tensor, y_std: float = 1.0) -> float:
    """
    RMSE in original y-units.
    """
    weights = beta[:-1] if bm.learns_noise else beta
    pred = torch.func.functional_call(bm.module, bm.param_dict_fn(weights), (X,)).squeeze(-1)
    return (((pred - y) ** 2).mean().sqrt() * y_std).item()


def build_map_checkpoint(
    bm: BayesianModule, x_ref: Tensor, Sigma_inv: Tensor,
    data: dict[str, Any], cfg: BNNConfig, dataset_name: str, split_id: int, out_maps_dir: Path,
) -> dict:
    X_test, y_test, y_std = data["X_test"], data["y_test"], data["y_std"]
    grad_target = torch.func.grad(bm.energy)

    rmse_map = eval_rmse(bm, x_ref, X_test, y_test, y_std)
    grad_norm_map = grad_target(x_ref).abs().max().item()
    print(f"  RMSE MAP={rmse_map:.4f}  ||grad_target||_inf MAP={grad_norm_map:.4e}")

    ckpt = {
        "dataset": dataset_name, "split_id": split_id,
        "x_ref": x_ref.cpu(),
        "Sigma_inv": Sigma_inv.cpu(),
        "rmse_map": rmse_map,
        "grad_norm_inf_map": grad_norm_map,
        "layer_sizes": cfg.layer_sizes, "activation": cfg.activation,
        "prior_sigma_scale": cfg.prior_sigma_scale, "y_std": y_std,
    }
    out_path = out_maps_dir / f"{dataset_name}_split{split_id:02d}_map.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"  saved -> {out_path}")

    return {"x_ref": x_ref, "rmse_map": rmse_map, "y_std": y_std, "Sigma_inv": Sigma_inv}


def checkpoint_path_for(out_maps_dir: Path, dataset_name: str, split_id: int) -> Path:
    return out_maps_dir / f"{dataset_name}_split{split_id:02d}_map.pt"


def load_map_checkpoint(out_path: Path, expected_D: int, device=DEVICE) -> Optional[dict]:
    """
    Loads a previously saved MAP checkpoint if it exists
    """
    if not out_path.exists():
        return None
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    x_ref = ckpt["x_ref"].to(dtype=DTYPE, device=device)
    if x_ref.shape[0] != expected_D:
        print(f"  found checkpoint at {out_path} but D mismatch "
              f"(ckpt D={x_ref.shape[0]} vs expected D={expected_D}) -- refitting")
        return None
    print(f"  loaded existing MAP checkpoint -> {out_path} (rmse_map={ckpt['rmse_map']:.4f})")
    return {
        "x_ref": x_ref,
        "rmse_map": ckpt["rmse_map"], "y_std": ckpt["y_std"],
        "Sigma_inv": ckpt["Sigma_inv"].to(dtype=DTYPE, device=device),
    }


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


def build_sticky_zigzag_sampler(bm: BayesianModule, cfg: BNNConfig) -> GridStickyZigZagSampler:
    kappa, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
    return GridStickyZigZagSampler(
        grad_target=torch.func.grad(bm.energy), D=bm.D,
        kappa=kappa, can_freeze=can_freeze, cold_start_threshold=None,
        gamma=GAMMA, grid_t_max_init=GRID_T_MAX_INIT_ZIGZAG, n_segments=GRID_N_SEGMENTS,
        grid_spacing=GRID_STICKY_ZIGZAG_SPACING, alpha_plus=GRID_ALPHA_PLUS, alpha_minus=GRID_ALPHA_MINUS,
        dtype=DTYPE, device=bm.device,
    )


def build_sticky_boomerang_sampler(bm: BayesianModule, cfg: BNNConfig, x_ref: Tensor,
                                    Sigma_inv: Tensor) -> GridStickyBoomerangSampler:
    kappa, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
    sampler = GridStickyBoomerangSampler(
        grad_target=torch.func.grad(bm.energy), D=bm.D,
        kappa=kappa, can_freeze=can_freeze, cold_start_threshold=None,
        grid_spacing=GRID_STICKY_BOOM_SPACING, refresh_rate=REFRESH_RATE,
        grid_t_max_init=GRID_T_MAX_INIT_BOOM, n_segments=GRID_N_SEGMENTS,
        alpha_plus=GRID_ALPHA_PLUS, alpha_minus=GRID_ALPHA_MINUS,
        dtype=DTYPE, device=bm.device,
    )
    sampler.preprocess(x_ref=x_ref, Sigma_inv=Sigma_inv * SIGMA_INV_SCALE)
    return sampler


def save_skeleton(out_path: Path, *, ref: dict, sampler: str, result: dict,
                   samples: Tensor, cfg: BNNConfig, elapsed_sec: float, n_skeleton: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sampler": sampler,
        "samples": samples.cpu(),
        "x_ref": ref["x_ref"].cpu(),
        "cold_start_mask": None,
        "rmse_map": ref["rmse_map"],
        "layer_sizes": cfg.layer_sizes, "activation": cfg.activation,
        "learned_noise": True, "prior_sigma_scale": cfg.prior_sigma_scale,
        "prior_inclusion_weight": cfg.prior_inclusion_weight,
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
    x_ref = ref["x_ref"]

    sampler = build_sticky_zigzag_sampler(bm, cfg)

    t0 = time.perf_counter()
    result = sampler.sample(N=n_skeleton, x0=x_ref, diagnostics=True)
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
        samples=thin_to(samples, N_SAVE),
        cfg=cfg, elapsed_sec=elapsed, n_skeleton=n_skeleton,
    )
    print(f"      saved -> {out_path}")


def run_grid_sticky_boomerang(ref: dict, cfg: BNNConfig, sd: Path, bm: BayesianModule,
                               Sigma_inv: Tensor, n_skeleton: int) -> None:
    out_path = sd / "grid_sticky_boomerang_skeleton.pt"
    x_ref = ref["x_ref"]

    sampler = build_sticky_boomerang_sampler(bm, cfg, x_ref, Sigma_inv)

    t0 = time.perf_counter()
    result = sampler.sample(N=n_skeleton, diagnostics=True)
    elapsed = time.perf_counter() - t0

    samples = resample_boomerang_path_sticky_torch(
        result["positions"], result["velocities"], result["times"], x_ref,
        N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
    )

    final_sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {n_skeleton} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, final sparsity {final_sparsity:.2f})")

    save_skeleton(
        out_path, ref=ref, sampler="grid_sticky_boomerang", result=result,
        samples=thin_to(samples, N_SAVE),
        cfg=cfg, elapsed_sec=elapsed, n_skeleton=n_skeleton,
    )
    print(f"      saved -> {out_path}")


def run_split(dataset_name: str, split_id: int, data: dict[str, Any], cfg: BNNConfig,
              out_maps_dir: Path, out_dir: Path, n_skeleton: int,
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

    ref = None if refit_maps else load_map_checkpoint(ckpt_path, expected_D)
    if ref is not None:
        bm = build_bm_only(data, cfg)
        print(f"  D = {bm.D}")
    else:
        print("\n[fit MAP]")
        bm, x_ref, Sigma_inv = build_target(data, cfg)
        print(f"  D = {bm.D}")
        ref = build_map_checkpoint(bm, x_ref, Sigma_inv, data, cfg, dataset_name, split_id, out_maps_dir)

    sd = out_dir / dataset_name / f"split_{split_id:02d}" / f"piw_{cfg.prior_inclusion_weight:g}"
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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=["boston"], choices=list(UCI_DATASETS))
    parser.add_argument("--splits", nargs="+", type=int, default=[0])
    parser.add_argument("--hidden-variant", choices=list(HIDDEN_VARIANTS), default=DEFAULT_HIDDEN_VARIANT)
    parser.add_argument("--n-skeleton", type=int, default=N_SKELETON)
    parser.add_argument("--prior-inclusion-weight", nargs="+", type=float, default=[0.1],
                         help="Sticky-only spike-and-slab prior inclusion probability w"
                              "kappa = (w/(1-w))/(sigma_w*sqrt(2pi)).")
    parser.add_argument("--resume", action="store_true",
                         help="Skip a sampler run if its output .pt already exists.")
    parser.add_argument("--refit-maps", action="store_true",
                         help="Force a fresh MAP fit")
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

    print(f"\nRunning {datasets_to_run} | splits: {args.splits} | "
          f"hidden_variant={args.hidden_variant} ({hidden}) | "
          f"prior_inclusion_weight sweep={args.prior_inclusion_weight} | "
          f"N_SKELETON={args.n_skeleton}")

    # The MAP checkpoint doesn't depend on prior_inclusion_weight
    for piw in args.prior_inclusion_weight:
        cfgs = configs_for({n: X.shape[1] for n, (X, _) in raw.items()}, hidden,
                            prior_inclusion_weight=piw)
        print(f"\n=== prior_inclusion_weight={piw:g} ===")
        for ds in datasets_to_run:
            X, y = raw[ds]
            cfg = cfgs[ds]
            for split_id in args.splits:
                data = make_split(X, y, seed=BASE_SEED + split_id, dtype=DTYPE, device=DEVICE)
                run_split(
                    ds, split_id, data, cfg, out_maps_dir, out_dir,
                    args.n_skeleton, args.resume, args.refit_maps,
                )


if __name__ == "__main__":
    main()
