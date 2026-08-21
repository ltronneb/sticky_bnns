"""
Skeleton-only variant of uci_bnn_grid.py, for local inspection of how the
grid PDMP samplers' rate/grid_t_max behaves over a run -- mirrors that
script's data/config/model/sampler wiring exactly, but skips the
resample_*_path_torch call and saves the raw skeleton (positions,
velocities, times, grid_t_max_log, ...) instead of resampled draws. Only
the four grid PDMP samplers are included -- NUTS/NUTS-HS have no skeleton/
rate concept, see uci_bnn_grid.py for those.

100K skeleton points is a lot to keep around just to eyeball rate
behavior, so only the last N_SKELETON_SAVE (default 20_000) events are
saved -- still enough to see steady-state grid_t_max/rate behavior, at a
fraction of the disk footprint. Samples can always be drawn later from the
full skeleton by rerunning uci_bnn_grid.py; this script is diagnostic-only.

Usage:
    python -m sazz.gpu_friendly.scripts.uci_bnn_grid_skeleton
    python -m sazz.gpu_friendly.scripts.uci_bnn_grid_skeleton --datasets boston --splits 0
    python -m sazz.gpu_friendly.scripts.uci_bnn_grid_skeleton --samplers grid_boomerang grid_sticky_boomerang
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sazz.gpu_friendly.scripts.uci_bnn_grid import (
    DEVICE, DTYPE,
    N_SKELETON, BASE_SEED,
    HIDDEN_VARIANTS, DEFAULT_HIDDEN_VARIANT,
    UCI_DATASETS,
    BNNConfig, configs_for,
    load_raw_datasets, make_split, build_target,
    build_zigzag_sampler, build_sticky_zigzag_sampler,
    build_boomerang_sampler, build_sticky_boomerang_sampler,
    split_dir,
)

N_SKELETON_SAVE = 20_000

OUT_DIR = Path("results/grid/uci_bnn_skeleton")

SAMPLER_NAMES = ("grid_zigzag", "grid_sticky_zigzag", "grid_boomerang", "grid_sticky_boomerang")


# ===========================================================================
# Persistence -- saves the tail of the skeleton (last N_SKELETON_SAVE
# events) instead of resampled draws. No thin_to/resample_*_path_torch
# call anywhere in this script.
# ===========================================================================

def save_skeleton(out_path: Path, *, dataset: str, split_id: int, sampler: str,
                   result: dict, cfg: BNNConfig, y_std: float, elapsed_sec: float,
                   n_skeleton: int, n_save: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tail = slice(-n_save, None)

    # diag_log ("diagnostics") and grid_t_max_log are both appended once per
    # *sampler-loop iteration*, which is a finer-grained unit than
    # positions/velocities/times: no-event/freeze/thaw iterations don't
    # always append a new row to positions (see e.g.
    # grid_sticky_boomerang.sample -- freeze/thaw rows advance time_passed
    # without incrementing n), and refresh emits an extra synthetic
    # diagnostics row not tied to any single skeleton index. So both logs'
    # length need not equal N_SKELETON, they are NOT index-aligned with
    # positions/velocities/times, and their tail is sliced independently
    # (by row count) from the positions tail -- do not zip them together.
    diag_log = result.get("diagnostics")
    diagnostics_tail = diag_log[-n_save:] if diag_log else diag_log
    t_max_log = result.get("grid_t_max_log")
    t_max_tail = t_max_log[-n_save:] if t_max_log else t_max_log

    payload = {
        "dataset":           dataset,
        "split_id":          split_id,
        "sampler":           sampler,
        "positions":         result["positions"][tail].cpu(),
        "velocities":        result["velocities"][tail].cpu(),
        "times":             result["times"][tail].cpu(),
        "diagnostics":       diagnostics_tail,
        "n_skeleton_total":  n_skeleton,
        "n_skeleton_saved":  min(n_save, n_skeleton),
        "layer_sizes":       cfg.layer_sizes,
        "activation":        cfg.activation,
        "learned_noise":     True,
        "prior_sigma_scale": cfg.prior_sigma_scale,
        "y_std":             y_std,
        "elapsed_sec":       elapsed_sec,
        "bound_violations":  result["bound_violations"],
        "gradient_evals":    result["gradient_evals"],
        "grid_t_max_log":    t_max_tail,
    }
    if "frozen_mask_final" in result:
        payload["frozen_mask_final"] = result["frozen_mask_final"].cpu()
    torch.save(payload, out_path)


# ===========================================================================
# Per-(dataset, split) runners -- same sampler construction as
# uci_bnn_grid.py, no resampling step.
# ===========================================================================

def run_grid_zigzag(dataset_name: str, split_id: int, data: dict[str, Any],
                     cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    sampler = build_zigzag_sampler(bm)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, x0=x_ref, diagnostics=True)
    elapsed = time.perf_counter() - t0

    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations)")

    out_path = sd / "grid_zigzag_skeleton.pt"
    save_skeleton(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_zigzag",
        result=result, cfg=cfg, y_std=data["y_std"], elapsed_sec=elapsed,
        n_skeleton=N_SKELETON, n_save=N_SKELETON_SAVE,
    )
    print(f"      saved last {min(N_SKELETON_SAVE, N_SKELETON)} skeleton events -> {out_path}")


def run_grid_sticky_zigzag(dataset_name: str, split_id: int, data: dict[str, Any],
                            cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    sampler = build_sticky_zigzag_sampler(bm, cfg)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, x0=x_ref, diagnostics=True)
    elapsed = time.perf_counter() - t0

    sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, "
          f"final sparsity {sparsity:.2f})")

    out_path = sd / "grid_sticky_zigzag_skeleton.pt"
    save_skeleton(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_sticky_zigzag",
        result=result, cfg=cfg, y_std=data["y_std"], elapsed_sec=elapsed,
        n_skeleton=N_SKELETON, n_save=N_SKELETON_SAVE,
    )
    print(f"      saved last {min(N_SKELETON_SAVE, N_SKELETON)} skeleton events -> {out_path}")


def run_grid_boomerang(dataset_name: str, split_id: int, data: dict[str, Any],
                        cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    sampler = build_boomerang_sampler(bm, x_ref, Sigma_inv)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, diagnostics=True)
    elapsed = time.perf_counter() - t0

    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations)")

    out_path = sd / "grid_boomerang_skeleton.pt"
    save_skeleton(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_boomerang",
        result=result, cfg=cfg, y_std=data["y_std"], elapsed_sec=elapsed,
        n_skeleton=N_SKELETON, n_save=N_SKELETON_SAVE,
    )
    print(f"      saved last {min(N_SKELETON_SAVE, N_SKELETON)} skeleton events -> {out_path}")


def run_grid_sticky_boomerang(dataset_name: str, split_id: int, data: dict[str, Any],
                               cfg: BNNConfig, sd: Path, bm, x_ref, Sigma_inv) -> None:
    sampler = build_sticky_boomerang_sampler(bm, cfg, x_ref, Sigma_inv)

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, diagnostics=True)
    elapsed = time.perf_counter() - t0

    sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, "
          f"final sparsity {sparsity:.2f})")

    out_path = sd / "grid_sticky_boomerang_skeleton.pt"
    save_skeleton(
        out_path, dataset=dataset_name, split_id=split_id, sampler="grid_sticky_boomerang",
        result=result, cfg=cfg, y_std=data["y_std"], elapsed_sec=elapsed,
        n_skeleton=N_SKELETON, n_save=N_SKELETON_SAVE,
    )
    print(f"      saved last {min(N_SKELETON_SAVE, N_SKELETON)} skeleton events -> {out_path}")


SAMPLER_RUNNERS = {
    "grid_zigzag": run_grid_zigzag,
    "grid_sticky_zigzag": run_grid_sticky_zigzag,
    "grid_boomerang": run_grid_boomerang,
    "grid_sticky_boomerang": run_grid_sticky_boomerang,
}


def run_split(dataset_name: str, split_id: int, data: dict[str, Any],
              cfg: BNNConfig, out_dir: Path, samplers: list[str], resume: bool) -> None:
    print(f"\n--- {dataset_name.upper()} split {split_id:02d} | "
          f"layers={cfg.layer_sizes} | act={cfg.activation} | "
          f"noise=learned (HalfNormal scale={cfg.prior_sigma_scale:.4f}) | "
          f"seed={BASE_SEED + split_id} ---")

    sd = split_dir(out_dir, dataset_name, split_id)

    pending = [
        s for s in samplers
        if not (resume and (sd / f"{s}_skeleton.pt").exists())
    ]
    for s in samplers:
        if s not in pending:
            print(f"  [{s}] skipping — exists at {sd / f'{s}_skeleton.pt'}")

    if not pending:
        return

    seed = BASE_SEED + split_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, x_ref, Sigma_inv = build_target(data, cfg)
    print(f"  D = {bm.D}")

    for sampler_name in pending:
        print(f"  [{sampler_name}]")
        torch.manual_seed(seed)
        np.random.seed(seed)
        SAMPLER_RUNNERS[sampler_name](dataset_name, split_id, data, cfg, sd, bm, x_ref, Sigma_inv)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    global N_SKELETON, N_SKELETON_SAVE

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", default=["boston"],
                         choices=list(UCI_DATASETS))
    parser.add_argument("--samplers", nargs="+", default=list(SAMPLER_NAMES),
                         choices=list(SAMPLER_NAMES))
    parser.add_argument("--splits", nargs="+", type=int, default=[0])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--hidden-variant", choices=list(HIDDEN_VARIANTS),
                         default=DEFAULT_HIDDEN_VARIANT)
    parser.add_argument("--n-skeleton", type=int, default=N_SKELETON,
                         help="Total skeleton events to simulate (default matches "
                              "uci_bnn_grid.py's N_SKELETON).")
    parser.add_argument("--n-skeleton-save", type=int, default=N_SKELETON_SAVE,
                         help="How many of the *last* skeleton events to save to disk "
                              "(default 20_000).")
    parser.add_argument("--prior-inclusion-weight", type=float, default=0.7,
                         help="Sticky-only spike-and-slab prior inclusion probability. "
                              "Only affects grid_sticky_zigzag/grid_sticky_boomerang.")
    args = parser.parse_args()

    N_SKELETON = args.n_skeleton
    N_SKELETON_SAVE = args.n_skeleton_save

    hidden = HIDDEN_VARIANTS[args.hidden_variant]
    out_dir = args.out if args.hidden_variant == DEFAULT_HIDDEN_VARIANT else args.out / args.hidden_variant
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
          f"N_SKELETON={N_SKELETON} N_SKELETON_SAVE={N_SKELETON_SAVE} "
          f"prior_inclusion_weight={args.prior_inclusion_weight}")

    for ds in datasets_to_run:
        X, y = raw[ds]
        cfg = cfgs[ds]
        for split_id in args.splits:
            data = make_split(X, y, seed=BASE_SEED + split_id, dtype=DTYPE, device=DEVICE)
            run_split(ds, split_id, data, cfg, out_dir, list(args.samplers), resume=args.resume)


if __name__ == "__main__":
    main()
