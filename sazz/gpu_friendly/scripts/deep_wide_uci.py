"""
Wide-architecture UCI regression BNN driver -- the memory-bounded analog of
uci_bnn_grid.py for the "deep_wide" ([256, 128, 64]) hidden variant, whose
D ~ 45,000 makes uci_bnn_grid.py's single in-memory [N_SKELETON, D]
skeleton allocation OOM a 22 GiB GPU (150,000 * 45,000 * 4 bytes ~ 27 GiB
in fp32, before the model itself).

This script keeps everything about uci_bnn_grid.py that defines the target
-- identical data loading, split construction, BNNConfig, build_target
(MAP + Laplace via find_reference_bnn), and the NUTS / NUTS-HS runners are
imported verbatim from it -- and swaps ONLY the four grid PDMP runners for
versions built on the Fast* sticky samplers (FastGridStickyZigZagSampler /
FastGridStickyBoomerangSampler), which stream the skeleton to disk in
chunks of SKELETON_CHUNK_SIZE rows (peak GPU memory O(chunk_size * D)
instead of O(N * D)) and resample via resample_*_sticky_chunked_torch.
This is the same chunk-to-disk mechanism fast_cifar_resnet.py uses at
CNN/ResNet scale.

WHAT IS AND IS NOT COMPARABLE TO uci_bnn_grid.py:
  * The gradient is FULL BATCH here -- grad_target = torch.func.grad(bm.energy)
    over the entire training set, exactly as in uci_bnn_grid.py. No
    minibatching. So NUTS vs sticky-PDMP for deep_wide stays apples-to-
    apples: every sampler sees the identical posterior.
  * Chunking changes only WHERE skeleton rows are stored (disk vs GPU RAM),
    not the dynamics. A full-batch, chunk_size=None run of a Fast* sticky
    sampler is intended to be behaviorally identical to the corresponding
    Grid* sticky sampler in uci_bnn_grid.py (see fast_grid_sticky_zigzag.py's
    docstring: "bounds peak memory to O(chunk_size * D)" is the stated
    difference). Confirm this if the deep_wide sticky numbers need to sit
    in the same table as the small / deep_narrow sticky numbers.
  * ONLY the two sticky PDMP families are available here. Plain
    grid_zigzag / grid_boomerang have no Fast* chunked equivalent, so they
    are simply not offered -- deep_wide gets grid_sticky_zigzag,
    grid_sticky_boomerang, nuts, nuts_horseshoe.
  * The Fast* samplers' sample() has NO grad_budget parameter (unlike the
    Grid* samplers). PDMP runs here are event-count driven via --n-skeleton
    only. --grad-budget still applies to NUTS / NUTS-HS (their runners are
    the uci_bnn_grid.py ones unchanged).

Grid / rate constants are inherited from uci_bnn_grid.py (UCI-scale: tanh,
no BatchNorm), NOT from fast_cifar_resnet.py (CNN-scale). deep_wide's D is
~10x the "small" variant's, so grid_t_max_init / grid_spacing may still
need a mild retune -- use diagnose_zigzag_rate.py against a real reference
checkpoint before a non-smoke run. alpha_violation (a Fast*-only knob with
no Grid* equivalent) defaults to fast_cifar_resnet.py's 1.1.

Usage:
    python -m sazz.gpu_friendly.scripts.deep_wide_uci \\
        --datasets boston --splits 0 \\
        --samplers grid_sticky_zigzag grid_sticky_boomerang \\
        --n-skeleton 150_000 --skeleton-chunk-size 5_000 \\
        --out results/paper/deep_wide

    python -m sazz.gpu_friendly.scripts.deep_wide_uci \\
        --datasets boston energy naval --samplers nuts nuts_horseshoe \\
        --grad-budget 1_000_000 --out results/paper/deep_wide
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from sazz.gpu_friendly.models.priors import (
    build_kappa_from_inclusion, build_can_freeze_mask,
)
from sazz.gpu_friendly.utils.resample import (
    resample_zigzag_path_sticky_torch, resample_boomerang_path_sticky_torch,
    resample_zigzag_path_sticky_chunked_torch, resample_boomerang_path_sticky_chunked_torch,
)
from sazz.gpu_friendly.samplers.fast_grid_sticky_zigzag import FastGridStickyZigZagSampler
from sazz.gpu_friendly.samplers.fast_grid_sticky_boomerang import FastGridStickyBoomerangSampler

# Everything that defines the target / NUTS is imported verbatim from
# uci_bnn_grid.py -- this script only replaces the four grid PDMP runners.
from sazz.gpu_friendly.scripts.uci_bnn_grid import (
    DEVICE, DTYPE,
    N_SKELETON, N_RESAMPLE, N_SAVE, BURNIN_FRAC, BASE_SEED,
    GRID_N_SEGMENTS, GRID_ALPHA_PLUS, GRID_ALPHA_MINUS,
    GRID_STICKY_COLD_START_THRESHOLD,
    HIDDEN_VARIANTS, UCI_DATASETS,
    BNNConfig, configs_for,
    load_raw_datasets, make_split, build_target,
    thin_to, split_dir, save_run,
    run_nuts_dataset, run_nuts_horseshoe_dataset,
)

# Fast*-only knob (no Grid* equivalent); fast_cifar_resnet.py's value.
GRID_ALPHA_VIOLATION = 1.1
SIGMA_INV_SCALE = 0.1

GAMMA = 1e-6
GRID_T_MAX_INIT_ZIGZAG = 0.001
GRID_SPACING_ZIGZAG = 1e-5

REFRESH_RATE = 5e2
GRID_T_MAX_INIT_BOOM = 0.01 #math.pi / 64  
GRID_SPACING_BOOM = 1e-3 #math.pi / 64 

GRID_STICKY_BOOM_SPACING = GRID_SPACING_BOOM
GRID_STICKY_ZIGZAG_SPACING = GRID_SPACING_ZIGZAG

# Fast*-only vmap grid-eval batch size (DISTINCT from the skeleton
# chunk_size below -- same name, different kwarg on the sampler ctor).
# Small is fine; keeps the per-grid-call vmap footprint tiny.
GRID_VMAP_CHUNK_SIZE = 4

# --- Skeleton-to-disk chunking (the whole point of this script). None =>
# behave exactly like uci_bnn_grid.py's Grid* sticky runners (one [N, D]
# allocation) -- only useful here for the smaller UCI datasets / small
# hidden variants where deep_wide's OOM doesn't bite. Set a positive int
# (e.g. 5_000) for deep_wide. Wired from --skeleton-chunk-size. ---
SKELETON_CHUNK_SIZE: Optional[int] = None
SKELETON_CHUNK_DIR: Optional[Path] = None

# This script exists FOR the wide variant; default accordingly (still
# overridable via --hidden-variant for A/B checks against uci_bnn_grid.py).
DEFAULT_HIDDEN_VARIANT = "deep_wide"

OUT_DIR = Path("results/grid/uci_bnn_deep_wide")

SAMPLER_NAMES = ("grid_sticky_zigzag", "grid_sticky_boomerang", "nuts", "nuts_horseshoe")
PDMP_SAMPLERS = ("grid_sticky_zigzag", "grid_sticky_boomerang")


# ===========================================================================
# Sticky kappa / can_freeze -- the FFN builders from uci_bnn_grid.py's
# build_sticky_*_sampler (NOT fast_cifar_resnet.py's _resnet BatchNorm-aware
# variants -- deep_wide is a plain FFN). Factored out here because both the
# zigzag and boomerang runners need the identical construction.
# ===========================================================================

def _build_sticky_kappa_can_freeze(bm, cfg: BNNConfig):
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


def build_sticky_zigzag_sampler(bm, cfg: BNNConfig) -> FastGridStickyZigZagSampler:
    kappa, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
    return FastGridStickyZigZagSampler(
        grad_target=torch.func.grad(bm.energy),  # FULL BATCH -- no minibatching
        D=bm.D,
        kappa=kappa,
        can_freeze=can_freeze,
        cold_start_threshold=GRID_STICKY_COLD_START_THRESHOLD,
        gamma=GAMMA,
        grid_t_max_init=GRID_T_MAX_INIT_ZIGZAG,
        n_segments=GRID_N_SEGMENTS,
        grid_spacing=GRID_STICKY_ZIGZAG_SPACING,
        alpha_plus=GRID_ALPHA_PLUS,
        alpha_minus=GRID_ALPHA_MINUS,
        alpha_violation=GRID_ALPHA_VIOLATION,
        chunk_size=GRID_VMAP_CHUNK_SIZE,
        dtype=DTYPE,
        device=bm.device,
        resample_grad_batch=None,  # full batch: nothing to resample
    )


def build_sticky_boomerang_sampler(bm, cfg: BNNConfig,
                                    x_ref: torch.Tensor, Sigma_inv: torch.Tensor
                                    ) -> FastGridStickyBoomerangSampler:
    kappa, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
    sampler = FastGridStickyBoomerangSampler(
        grad_target=torch.func.grad(bm.energy),  # FULL BATCH -- no minibatching
        D=bm.D,
        kappa=kappa,
        can_freeze=can_freeze,
        cold_start_threshold=GRID_STICKY_COLD_START_THRESHOLD,
        grid_spacing=GRID_STICKY_BOOM_SPACING,
        refresh_rate=REFRESH_RATE,
        grid_t_max_init=GRID_T_MAX_INIT_BOOM,
        n_segments=GRID_N_SEGMENTS,
        alpha_plus=GRID_ALPHA_PLUS,
        alpha_minus=GRID_ALPHA_MINUS,
        alpha_violation=GRID_ALPHA_VIOLATION,
        chunk_size=GRID_VMAP_CHUNK_SIZE,
        dtype=DTYPE,
        device=bm.device,
        resample_grad_batch=None,
    )
    # Same Sigma_inv rescale as uci_bnn_grid.py's build_boomerang_sampler.
    sampler.preprocess(x_ref=x_ref, Sigma_inv=Sigma_inv * SIGMA_INV_SCALE)
    return sampler


# ===========================================================================
# Per-(dataset, split) PDMP runners -- Fast* + chunked resample. Mirrors
# fast_cifar_resnet.py's run_grid_sticky_* structure (chunking_active branch,
# sample_kwargs, chunked-vs-not resample), adapted to uci_bnn_grid.py's
# save_run signature (which takes dataset/split_id/y_std, unlike the CNN one).
# ===========================================================================

def _chunk_dir_for(dataset_name: str, split_id: int, sampler: str) -> Path:
    return SKELETON_CHUNK_DIR / dataset_name / f"split_{split_id:02d}" / sampler


def run_grid_sticky_zigzag(dataset_name: str, split_id: int, data: dict[str, Any],
                            cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    sampler = build_sticky_zigzag_sampler(bm, cfg)

    chunking_active = SKELETON_CHUNK_SIZE is not None
    sample_kwargs: dict[str, Any] = {}
    if chunking_active:
        sample_kwargs["chunk_size"] = SKELETON_CHUNK_SIZE
        sample_kwargs["chunk_dir"] = _chunk_dir_for(dataset_name, split_id, "grid_sticky_zigzag")

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, x0=x_ref, diagnostics=True, **sample_kwargs)
    elapsed = time.perf_counter() - t0

    if chunking_active:
        samples = resample_zigzag_path_sticky_chunked_torch(
            result["chunk_files"], N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
            manifest_path=result["manifest_path"], dtype=DTYPE, device=DEVICE,
        )
    else:
        samples = resample_zigzag_path_sticky_torch(
            result["positions"], result["velocities"], result["times"],
            N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
        )

    sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, "
          f"final sparsity {sparsity:.2f})")

    out_path = sd / "grid_sticky_zigzag.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_sticky_zigzag",
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=N_SKELETON,
        bound_violations=result["bound_violations"],
        gradient_evals=result["gradient_evals"],
        grid_t_max_log=result["grid_t_max_log"],
        grad_budget=None,  # Fast* sample() has no grad_budget; PDMP here is event-count driven
    )
    print(f"      saved {samples.shape[0]} samples (thinned to {N_SAVE}) -> {out_path}")


def run_grid_sticky_boomerang(dataset_name: str, split_id: int, data: dict[str, Any],
                               cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    sampler = build_sticky_boomerang_sampler(bm, cfg, x_ref, Sigma_inv)

    chunking_active = SKELETON_CHUNK_SIZE is not None
    sample_kwargs: dict[str, Any] = {}
    if chunking_active:
        sample_kwargs["chunk_size"] = SKELETON_CHUNK_SIZE
        sample_kwargs["chunk_dir"] = _chunk_dir_for(dataset_name, split_id, "grid_sticky_boomerang")

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, x0=x_ref, diagnostics=True, **sample_kwargs)
    elapsed = time.perf_counter() - t0

    if chunking_active:
        samples = resample_boomerang_path_sticky_chunked_torch(
            result["chunk_files"], x_ref, N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
            manifest_path=result["manifest_path"], dtype=DTYPE, device=DEVICE,
        )
    else:
        samples = resample_boomerang_path_sticky_torch(
            result["positions"], result["velocities"], result["times"], x_ref,
            N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
        )

    sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, "
          f"final sparsity {sparsity:.2f})")

    out_path = sd / "grid_sticky_boomerang.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_sticky_boomerang",
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=N_SKELETON,
        bound_violations=result["bound_violations"],
        gradient_evals=result["gradient_evals"],
        grid_t_max_log=result["grid_t_max_log"],
        grad_budget=None,
    )
    print(f"      saved {samples.shape[0]} samples (thinned to {N_SAVE}) -> {out_path}")


SAMPLER_RUNNERS = {
    "grid_sticky_zigzag": run_grid_sticky_zigzag,
    "grid_sticky_boomerang": run_grid_sticky_boomerang,
    # NUTS runners imported verbatim from uci_bnn_grid.py -- full-batch,
    # grad_budget-aware (via that module's GRAD_BUDGET global, which main()
    # below sets).
    "nuts": run_nuts_dataset,
    "nuts_horseshoe": run_nuts_horseshoe_dataset,
}


def run_split(dataset_name: str, split_id: int, data: dict[str, Any],
              cfg: BNNConfig, out_dir: Path, samplers: list[str], resume: bool) -> None:
    print(f"\n--- {dataset_name.upper()} split {split_id:02d} | "
          f"layers={cfg.layer_sizes} | act={cfg.activation} | "
          f"noise=learned (HalfNormal scale={cfg.prior_sigma_scale:.4f}) | "
          f"seed={BASE_SEED + split_id} ---")

    sd = split_dir(out_dir, dataset_name, split_id)

    pending = [s for s in samplers if not (resume and (sd / f"{s}.pt").exists())]
    for s in samplers:
        if s not in pending:
            print(f"  [{s}] skipping — exists at {sd / f'{s}.pt'}")
    if not pending:
        return

    seed = BASE_SEED + split_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, x_ref, Sigma_inv = build_target(data, cfg)
    print(f"  D = {bm.D}")
    if SKELETON_CHUNK_SIZE is not None:
        approx_gib = SKELETON_CHUNK_SIZE * bm.D * (4 if DTYPE == torch.float32 else 8) / 1024**3
        print(f"  skeleton chunking: {SKELETON_CHUNK_SIZE} rows/chunk "
              f"(~{approx_gib:.2f} GiB/buffer x3 buffers) -> {SKELETON_CHUNK_DIR}")
    else:
        print("  skeleton chunking: OFF (single [N, D] allocation -- will OOM for deep_wide)")

    for sampler_name in pending:
        print(f"  [{sampler_name}]")
        torch.manual_seed(seed)
        np.random.seed(seed)
        SAMPLER_RUNNERS[sampler_name](dataset_name, split_id, data, cfg, sd, bm, x_ref, Sigma_inv)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    global N_SKELETON, N_RESAMPLE, SKELETON_CHUNK_SIZE, SKELETON_CHUNK_DIR

    # uci_bnn_grid.py's NUTS runners read GRAD_BUDGET as a module global on
    # that module -- set it there, not here, so --grad-budget reaches them.
    import sazz.gpu_friendly.scripts.uci_bnn_grid as uci_grid

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=["boston"],
                         choices=list(UCI_DATASETS))
    parser.add_argument("--samplers", nargs="+", default=list(SAMPLER_NAMES),
                         choices=list(SAMPLER_NAMES),
                         help="Only the two sticky PDMP families are available (plain "
                              "grid_zigzag/grid_boomerang have no Fast* chunked equivalent); "
                              "nuts/nuts_horseshoe delegate to uci_bnn_grid.py's runners.")
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--hidden-variant", choices=list(HIDDEN_VARIANTS),
                         default=DEFAULT_HIDDEN_VARIANT,
                         help="Defaults to deep_wide (this script's reason to exist). "
                              "Overridable for A/B behavioral checks against uci_bnn_grid.py "
                              "at smaller sizes. Results are written under <out>/<variant>/... "
                              "for every variant except 'small' (flat <out>/...), matching "
                              "uci_bnn_grid.py's on-disk convention.")
    parser.add_argument("--n-skeleton", type=int, default=N_SKELETON,
                         help="Total skeleton events per PDMP run. The Fast* samplers have "
                              "no grad_budget, so this is the ONLY stopping control for the "
                              "sticky PDMP samplers here.")
    parser.add_argument("--n-resample", type=int, default=N_RESAMPLE)
    parser.add_argument("--skeleton-chunk-size", type=int, default=None,
                         help="Rows per on-disk skeleton chunk. Peak GPU memory for the "
                              "skeleton buffers is O(chunk_size * D) instead of O(n-skeleton * D). "
                              "REQUIRED (set a positive int, e.g. 5000) for deep_wide -- omitting "
                              "it reproduces uci_bnn_grid.py's single [N, D] allocation, which OOMs.")
    parser.add_argument("--skeleton-chunk-dir", type=Path, default=None,
                         help="Where per-(dataset, split, sampler) chunk_*.pt / manifest.pt / "
                              "diag_*.pt files go. Default: <out>/chunks. These are large "
                              "(the full skeleton); point at scratch, not the results tree, "
                              "if disk is tight. Safe to delete after the .pt sample files "
                              "are written.")
    parser.add_argument("--grad-budget", type=int, default=None,
                         help="Applies to NUTS / NUTS-HS ONLY (delegated to uci_bnn_grid.py's "
                              "budget-truncation logic). The Fast* sticky PDMP samplers ignore "
                              "it -- they stop at --n-skeleton events. Since .pt filenames "
                              "don't encode run mode, budget runs should use a distinct --out.")
    parser.add_argument("--prior-inclusion-weight", type=float, default=0.05,
                         help="Sticky-only spike-and-slab inclusion probability "
                              "(BNNConfig.prior_inclusion_weight -> build_kappa_from_inclusion).")
    args = parser.parse_args()

    if args.hidden_variant == "small":
        print("WARNING: --hidden-variant small on deep_wide_uci.py -- this script's grid "
              "constants and OOM-avoidance exist for the wide variant. For 'small' just use "
              "uci_bnn_grid.py directly unless you're doing a deliberate Fast*-vs-Grid* A/B.")
    if any(s in PDMP_SAMPLERS for s in args.samplers) and args.skeleton_chunk_size is None:
        print("WARNING: sticky PDMP samplers requested with --skeleton-chunk-size unset -- "
              "skeleton will be allocated as one [n-skeleton, D] tensor and WILL OOM for "
              "deep_wide. Pass e.g. --skeleton-chunk-size 5000.")
    if args.grad_budget is not None and any(s in PDMP_SAMPLERS for s in args.samplers):
        print(f"NOTE: --grad-budget {args.grad_budget} is ignored by the sticky PDMP samplers "
              f"(Fast* sample() has no grad_budget); it applies to nuts/nuts_horseshoe only. "
              f"The sticky runs will stop at --n-skeleton={args.n_skeleton} events.")

    N_SKELETON = args.n_skeleton
    N_RESAMPLE = args.n_resample
    SKELETON_CHUNK_SIZE = args.skeleton_chunk_size
    SKELETON_CHUNK_DIR = (
        args.skeleton_chunk_dir if args.skeleton_chunk_dir is not None
        else args.out / "chunks"
    )

    # Propagate to the imported NUTS runners' module.
    uci_grid.N_SKELETON = N_SKELETON
    uci_grid.N_RESAMPLE = N_RESAMPLE
    uci_grid.GRAD_BUDGET = args.grad_budget

    hidden = HIDDEN_VARIANTS[args.hidden_variant]
    out_dir = args.out if args.hidden_variant == "small" else args.out / args.hidden_variant
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw UCI datasets...")
    raw = load_raw_datasets(tuple(args.datasets))
    missing = [d for d in args.datasets if d not in raw]
    if missing:
        print(f"  (skipping {missing} -- data file(s) not found)")
    datasets_to_run = [d for d in args.datasets if d in raw]

    cfgs = configs_for({n: X.shape[1] for n, (X, _) in raw.items()}, hidden,
                        prior_inclusion_weight=args.prior_inclusion_weight)

    print(f"\nRunning {datasets_to_run} | samplers: {args.samplers} | splits: {args.splits} | "
          f"hidden_variant={args.hidden_variant} ({hidden}) | "
          f"N_SKELETON={N_SKELETON} N_RESAMPLE={N_RESAMPLE} "
          f"skeleton_chunk_size={SKELETON_CHUNK_SIZE} "
          f"prior_inclusion_weight={args.prior_inclusion_weight} "
          f"Sigma inverse scale={SIGMA_INV_SCALE}")

    for ds in datasets_to_run:
        X, y = raw[ds]
        cfg = cfgs[ds]
        for split_id in args.splits:
            data = make_split(X, y, seed=BASE_SEED + split_id, dtype=DTYPE, device=DEVICE)
            run_split(ds, split_id, data, cfg, out_dir, list(args.samplers), resume=args.resume)


if __name__ == "__main__":
    main()
