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
STAGED versions built on the _Cheap sticky samplers
(FastGridStickyZigZagSampler_Cheap / FastGridStickyBoomerangSampler_Cheap).
The skeleton is generated in STAGE_SIZE-sized stages chained via the
samplers' resume_state contract; immediately after each stage:
  1. that stage's chunk_*.pt files are resampled into N_RESAMPLE/n_stages
     draws (equal draws per stage, NOT time-proportional across stages),
  2. the entire stage_XXXX/ dir (chunk_*.pt + diag_*.pt + manifest.pt) is
     deleted,
  3. the sampler's resume_state is threaded into the next stage.
All stages' draws are concatenated into the same "samples" field a
non-staged run would produce. Peak disk is ONE stage
(O(stage_size * D)), not the whole skeleton -- this is the same
resample-and-discard loop fast_cheap_cifar_resnet.py uses at ResNet scale,
back-ported here so a full --n-skeleton 1_000_000 run at D~45,000 doesn't
need ~360 GB of transient chunk files on disk at once.

WHAT IS AND IS NOT COMPARABLE TO uci_bnn_grid.py:
  * The gradient is FULL BATCH here -- resample_grad_batch=None on the
    _Cheap samplers, so grad_target = torch.func.grad(bm.energy) over the
    entire training set, exactly as in uci_bnn_grid.py. No minibatching.
    The _Cheap classes are strict supersets of the plain Fast* ones: same
    dynamics, plus an OPTIONAL minibatch hook (unused here) and the
    resume_state contract (used here for staging). So NUTS vs sticky-PDMP
    for deep_wide stays apples-to-apples: every sampler sees the identical
    posterior.
  * Staging changes WHERE skeleton rows live (one stage on disk at a time)
    and makes the resample per-stage-uniform rather than globally uniform
    over simulation time (equal draws per stage; a stage covering more
    sim-time is under-weighted relative to a global draw). This is the
    same tradeoff fast_cheap_cifar_resnet.py accepts. Confirm it doesn't
    move the deep_wide sticky numbers before putting them in the same
    table as the small / deep_narrow sticky numbers (which ARE globally
    uniform). Set --stage-size >= --n-skeleton to force a single stage
    (recovers exact global-uniform resampling, at full disk cost).
  * ONLY the two sticky PDMP families are available here. Plain
    grid_zigzag / grid_boomerang have no _Cheap/chunked equivalent, so
    they are simply not offered -- deep_wide gets grid_sticky_zigzag,
    grid_sticky_boomerang, nuts, nuts_horseshoe.
  * The _Cheap samplers' sample() has NO grad_budget parameter (unlike the
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
        --n-skeleton 1_000_000 --stage-size 5_000 \\
        --out results/paper/deep_wide

    python -m sazz.gpu_friendly.scripts.deep_wide_uci \\
        --datasets boston energy naval --samplers nuts nuts_horseshoe \\
        --grad-budget 1_000_000 --out results/paper/deep_wide
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from sazz.gpu_friendly.models.priors import (
    build_kappa_from_inclusion, build_can_freeze_mask,
)
from sazz.gpu_friendly.utils.resample import (
    resample_zigzag_path_sticky_chunked_torch, resample_boomerang_path_sticky_chunked_torch,
)
from sazz.gpu_friendly.samplers.fast_grid_sticky_zigzag_cheap import FastGridStickyZigZagSampler_Cheap
from sazz.gpu_friendly.samplers.fast_grid_sticky_boomerang_cheap import FastGridStickyBoomerangSampler_Cheap

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
SIGMA_INV_SCALE = 10.0  # WEIGHT block of Sigma_inv only

SIGMA_LOGSIGMA_PREC_SCALE = 1.0

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

# --- Staged skeleton generation (the whole point of this script). The
# skeleton is produced in STAGE_SIZE-event stages chained via resume_state;
# each stage is resampled then its chunk dir is deleted, so peak disk is
# O(STAGE_SIZE * D), not O(N_SKELETON * D). REQUIRED (a positive int) for
# any sticky PDMP run -- wired from --stage-size (alias: --skeleton-chunk-
# size). STAGE_SIZE >= N_SKELETON => a single stage (exact global-uniform
# resample, full disk cost). ---
STAGE_SIZE: Optional[int] = None
STAGE_DIR: Optional[Path] = None

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

def _scaled_sigma_inv(Sigma_inv: torch.Tensor, bm) -> torch.Tensor:
    """SIGMA_INV_SCALE on the weight block; SIGMA_LOGSIGMA_PREC_SCALE on the
    trailing log_sigma coordinate (present only when bm.learns_noise). Mirrors
    uci_bnn_grid.py's _scaled_sigma_inv -- see the constants' comments there."""
    scale = torch.full_like(Sigma_inv, SIGMA_INV_SCALE)
    if bm.learns_noise:
        scale[-1] = SIGMA_LOGSIGMA_PREC_SCALE
    return Sigma_inv * scale


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


def build_sticky_zigzag_sampler(bm, cfg: BNNConfig) -> FastGridStickyZigZagSampler_Cheap:
    kappa, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
    return FastGridStickyZigZagSampler_Cheap(
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
        resample_grad_batch=None,  # full batch: nothing to resample between stages
    )


def build_sticky_boomerang_sampler(bm, cfg: BNNConfig,
                                    x_ref: torch.Tensor, Sigma_inv: torch.Tensor
                                    ) -> FastGridStickyBoomerangSampler_Cheap:
    kappa, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
    sampler = FastGridStickyBoomerangSampler_Cheap(
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
    # Same Sigma_inv rescale as uci_bnn_grid.py's build_boomerang_sampler
    # (weight block: SIGMA_INV_SCALE; log_sigma: SIGMA_LOGSIGMA_PREC_SCALE).
    sampler.preprocess(x_ref=x_ref, Sigma_inv=_scaled_sigma_inv(Sigma_inv, bm))
    return sampler


# ===========================================================================
# Per-(dataset, split) PDMP runners -- STAGED: _Cheap sticky sampler chained
# across ceil(N_SKELETON / STAGE_SIZE) stages via resume_state, each stage
# resampled into N_RESAMPLE/n_stages draws and then its chunk dir deleted so
# peak disk stays O(STAGE_SIZE * D). Ported from fast_cheap_cifar_resnet.py's
# run_grid_sticky_* loop, adapted to uci_bnn_grid.py's save_run signature
# (dataset/split_id/y_std; no test_accuracy) and full-batch gradients.
# ===========================================================================

def _stage_dir_base_for(dataset_name: str, split_id: int, sampler: str) -> Path:
    return STAGE_DIR / dataset_name / f"split_{split_id:02d}" / sampler


def _run_staged_sticky(
    sampler,
    *,
    sampler_name: str,
    dataset_name: str,
    split_id: int,
    data: dict[str, Any],
    cfg: BNNConfig,
    sd: Path,
    x_ref: torch.Tensor,
    resample_stage_fn,
) -> None:
    """Shared staged-sampling driver for both sticky families.

    resample_stage_fn(chunk_files, manifest_path, n_draws, burnin_frac) ->
    Tensor[n_draws, D] is the one piece that differs between ZigZag and
    Boomerang (the latter also needs x_ref); the caller binds it.
    """
    assert STAGE_SIZE is not None, "deep_wide_uci.py sticky runs require --stage-size."

    stage_base = _stage_dir_base_for(dataset_name, split_id, sampler_name)
    n_stages = math.ceil(N_SKELETON / STAGE_SIZE)
    draws_per_stage = max(N_RESAMPLE // n_stages, 1)

    all_draws: list[torch.Tensor] = []
    total_grad_evals = 0
    total_bound_violations = 0
    all_tmax_log: list[float] = []
    frozen_mask_final = None
    resume_state = None

    t0 = time.perf_counter()
    for stage in range(n_stages):
        # sample()'s N counts row 0 (seed / resumed position) PLUS N-1 new
        # events, so stage_N = stage_new_events + 1 -- same convention as
        # fast_cheap_cifar_resnet.py's staged runners.
        stage_new_events = min(STAGE_SIZE, N_SKELETON - stage * STAGE_SIZE)
        stage_N = stage_new_events + 1
        stage_dir = stage_base / f"stage_{stage:04d}"
        print(f"      [stage {stage + 1}/{n_stages}] sampling {stage_new_events} skeleton points "
              f"({'cold start' if stage == 0 else 'resumed'}) -> {stage_dir}")

        result = sampler.sample(
            N=stage_N,
            x0=(x_ref if stage == 0 else None),
            resume_state=resume_state,
            diagnostics=True,
            chunk_size=STAGE_SIZE,
            chunk_dir=stage_dir,
        )

        # burn-in only on stage 0 (its first BURNIN_FRAC of events); later
        # stages are already past burn-in, so keep all of their events.
        stage_draws = resample_stage_fn(
            result["chunk_files"], result["manifest_path"],
            draws_per_stage, BURNIN_FRAC if stage == 0 else 0.0,
        )
        # .cpu() each stage -- avoids holding n_stages worth of draws on the
        # GPU plus the final torch.cat's transient full-size allocation.
        all_draws.append(stage_draws.cpu())

        total_grad_evals += result["gradient_evals"]
        total_bound_violations += result["bound_violations"]
        all_tmax_log.extend(result["grid_t_max_log"])
        frozen_mask_final = result["frozen_mask_final"]
        resume_state = result["resume_state"]

        # Delete the whole stage dir -- chunk_*.pt (the [rows, D] skeleton),
        # diag_*.pt and manifest.pt. Nothing downstream needs it once the
        # stage's draws are in all_draws; this is what bounds peak disk.
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        print(f"      [stage {stage + 1}/{n_stages}] freed {stage_dir}")

    elapsed = time.perf_counter() - t0
    samples = torch.cat(all_draws, dim=0)

    final_sparsity = float(frozen_mask_final.float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s across {n_stages} stages "
          f"({total_bound_violations} bound violations, final sparsity {final_sparsity:.2f})")

    out_path = sd / f"{sampler_name}.pt"
    save_run(
        out_path, dataset=dataset_name, split_id=split_id, sampler=sampler_name,
        samples=samples, x_ref=x_ref, cfg=cfg, y_std=data["y_std"],
        elapsed_sec=elapsed, n_events=N_SKELETON,
        bound_violations=total_bound_violations,
        gradient_evals=total_grad_evals,
        grid_t_max_log=all_tmax_log,
        grad_budget=None,  # _Cheap sample() has no grad_budget; PDMP here is event-count driven
    )
    print(f"      saved {samples.shape[0]} samples (thinned to {N_SAVE}) -> {out_path}")


def run_grid_sticky_zigzag(dataset_name: str, split_id: int, data: dict[str, Any],
                            cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    sampler = build_sticky_zigzag_sampler(bm, cfg)

    def resample_stage(chunk_files, manifest_path, n_draws, burnin_frac):
        return resample_zigzag_path_sticky_chunked_torch(
            chunk_files, N_resample=n_draws, burnin_frac=burnin_frac,
            manifest_path=manifest_path, dtype=DTYPE, device=DEVICE,
        )

    _run_staged_sticky(
        sampler, sampler_name="grid_sticky_zigzag",
        dataset_name=dataset_name, split_id=split_id, data=data, cfg=cfg, sd=sd,
        x_ref=x_ref, resample_stage_fn=resample_stage,
    )


def run_grid_sticky_boomerang(dataset_name: str, split_id: int, data: dict[str, Any],
                               cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    sampler = build_sticky_boomerang_sampler(bm, cfg, x_ref, Sigma_inv)

    def resample_stage(chunk_files, manifest_path, n_draws, burnin_frac):
        return resample_boomerang_path_sticky_chunked_torch(
            chunk_files, x_ref, N_resample=n_draws, burnin_frac=burnin_frac,
            manifest_path=manifest_path, dtype=DTYPE, device=DEVICE,
        )

    _run_staged_sticky(
        sampler, sampler_name="grid_sticky_boomerang",
        dataset_name=dataset_name, split_id=split_id, data=data, cfg=cfg, sd=sd,
        x_ref=x_ref, resample_stage_fn=resample_stage,
    )


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
    if STAGE_SIZE is not None:
        approx_gib = STAGE_SIZE * bm.D * (4 if DTYPE == torch.float32 else 8) / 1024**3
        n_stages = math.ceil(N_SKELETON / STAGE_SIZE)
        print(f"  staged skeleton: {STAGE_SIZE} events/stage x {n_stages} stages "
              f"(~{approx_gib:.2f} GiB/buffer x3 buffers, peak disk ~1 stage) -> {STAGE_DIR}")
    else:
        print("  STAGE_SIZE unset -- sticky PDMP runners will assert. Pass --stage-size.")

    for sampler_name in pending:
        print(f"  [{sampler_name}]")
        torch.manual_seed(seed)
        np.random.seed(seed)
        SAMPLER_RUNNERS[sampler_name](dataset_name, split_id, data, cfg, sd, bm, x_ref, Sigma_inv)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    global N_SKELETON, N_RESAMPLE, STAGE_SIZE, STAGE_DIR

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
                         help="Total skeleton events per PDMP run (summed across stages). "
                              "The _Cheap samplers have no grad_budget, so this is the ONLY "
                              "stopping control for the sticky PDMP samplers here.")
    parser.add_argument("--n-resample", type=int, default=N_RESAMPLE,
                         help="Total resample draws, split equally across stages "
                              "(N_RESAMPLE // n_stages per stage). Not exactly reproducible "
                              "if you change --stage-size, since draws-per-stage changes.")
    parser.add_argument("--stage-size", "--skeleton-chunk-size", type=int, default=None,
                         dest="stage_size",
                         help="Skeleton events per stage. The skeleton is generated in "
                              "ceil(n-skeleton / stage-size) stages chained via resume_state; "
                              "each stage is resampled then its chunk dir is deleted, so peak "
                              "disk is O(stage-size * D), not O(n-skeleton * D). REQUIRED "
                              "(a positive int, e.g. 5000) for any sticky PDMP run. "
                              "stage-size >= n-skeleton => a single stage (exact global-uniform "
                              "resample, full disk cost). --skeleton-chunk-size is a "
                              "backward-compatible alias.")
    parser.add_argument("--stage-dir", "--skeleton-chunk-dir", type=Path, default=None,
                         dest="stage_dir",
                         help="Where per-(dataset, split, sampler)/stage_XXXX/ dirs (chunk_*.pt "
                              "+ diag_*.pt + manifest.pt) go transiently. Default: <out>/chunks. "
                              "Only one stage exists at a time, so this needs O(stage-size * D * 3) "
                              "free, not the full skeleton -- but pointing it at node scratch "
                              "keeps even that off your quota. --skeleton-chunk-dir is an alias.")
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
    if any(s in PDMP_SAMPLERS for s in args.samplers) and args.stage_size is None:
        parser.error("sticky PDMP samplers require --stage-size (e.g. 5000). The skeleton is "
                     "generated in ceil(n-skeleton / stage-size) stages; each is resampled then "
                     "deleted, bounding peak disk to O(stage-size * D).")
    if args.stage_size is not None and args.stage_size <= 0:
        parser.error("--stage-size must be a positive integer.")
    if args.grad_budget is not None and any(s in PDMP_SAMPLERS for s in args.samplers):
        print(f"NOTE: --grad-budget {args.grad_budget} is ignored by the sticky PDMP samplers "
              f"(_Cheap sample() has no grad_budget); it applies to nuts/nuts_horseshoe only. "
              f"The sticky runs will stop at --n-skeleton={args.n_skeleton} events.")

    N_SKELETON = args.n_skeleton
    N_RESAMPLE = args.n_resample
    STAGE_SIZE = args.stage_size
    STAGE_DIR = (
        args.stage_dir if args.stage_dir is not None
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

    # configs_for populates cfg.sigma_inv_scale from uci_bnn_grid's
    # SIGMA_INV_SCALE_TABLE, but this script overrides it below with its own
    # module-level SIGMA_INV_SCALE (deep_wide's Fast* grid constants and
    # reference scale are tuned here, not in that table).
    cfgs = configs_for({n: X.shape[1] for n, (X, _) in raw.items()}, hidden,
                        hidden_variant=args.hidden_variant,
                        prior_inclusion_weight=args.prior_inclusion_weight)
    for cfg in cfgs.values():
        cfg.sigma_inv_scale = SIGMA_INV_SCALE

    n_stages_str = (
        str(math.ceil(N_SKELETON / STAGE_SIZE)) if STAGE_SIZE is not None else "n/a"
    )
    print(f"\nRunning {datasets_to_run} | samplers: {args.samplers} | splits: {args.splits} | "
          f"hidden_variant={args.hidden_variant} ({hidden}) | "
          f"N_SKELETON={N_SKELETON} N_RESAMPLE={N_RESAMPLE} "
          f"stage_size={STAGE_SIZE} n_stages={n_stages_str} stage_dir={STAGE_DIR} "
          f"prior_inclusion_weight={args.prior_inclusion_weight} "
          f"Sigma inverse scale={SIGMA_INV_SCALE} "
          f"log_sigma prec scale={SIGMA_LOGSIGMA_PREC_SCALE}")

    for ds in datasets_to_run:
        X, y = raw[ds]
        cfg = cfgs[ds]
        for split_id in args.splits:
            data = make_split(X, y, seed=BASE_SEED + split_id, dtype=DTYPE, device=DEVICE)
            run_split(ds, split_id, data, cfg, out_dir, list(args.samplers), resume=args.resume)


if __name__ == "__main__":
    main()
