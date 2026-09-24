"""
Multi-chain convergence driver for the deep_wide UCI BNNs -- deep_wide_uci.py
run from several DIFFERENT MAPs so the resulting chains can be compared
against each other (R-hat / ESS / pairwise marginal agreement) instead of
being assessed from a single trajectory.

WHY THIS SCRIPT EXISTS. deep_wide_uci.py fits exactly one reference
(MAP + diagonal Laplace) per (dataset, split) and runs every sampler from
it. A single chain from a single start point cannot distinguish "the
sampler has converged" from "the sampler is stuck near where it started" --
at D ~ 45,000 with a non-convex posterior that is a real risk. This script
fits N INDEPENDENT references and runs one chain from each, so the chains
are overdispersed with respect to the target by construction and
between-chain disagreement is diagnostic.

EVERYTHING ABOUT THE TARGET AND THE SAMPLING IS deep_wide_uci.py's.
Data loading, split construction, BNNConfig, the staged _Cheap sticky
runners, time-weighted pooling, grid constants and save_run are imported
verbatim from it (which imports most of them from uci_bnn_grid.py in turn).
This script adds exactly three things:
  1. `fit-maps`: fit N references from N different random inits and save
     them as inspectable map_XX.pt files.
  2. `sample`:   load one saved reference per chain and run the staged
     samplers from it, writing to a per-chain output dir.
  3. the bookkeeping to keep those chains from colliding on disk.

HOW THE MAPS DIFFER. find_reference_bnn starts its Adam run from
`torch.randn(bm.D)` (warmup.py) and takes no seed argument, so the init is
governed by ambient torch RNG state. Chain i therefore seeds
torch.manual_seed(MAP_SEED_BASE + i) immediately before the fit, giving a
different random init -- and, at this dimensionality on a non-convex loss,
a genuinely different mode. Every MAP still optimizes the SAME objective
(the full training split), so all N are valid MAPs of the SAME posterior:
they are legitimate overdispersed starts, not MAPs of N perturbed targets.
(Subsampling the training data per MAP would separate them further but each
would then be the MAP of a slightly different posterior, which is a weaker
basis for a convergence claim. Not done here.)

SIGMA_INV AND WHAT "THE SAME SAMPLER" MEANS. For Boomerang the reference
measure (x_ref, Sigma_inv) defines the DYNAMICS, not merely the start
point, so per-chain Sigma_inv would make the N chains N different Markov
kernels -- and R-hat across different kernels does not test what it is
usually taken to test. Both options are therefore preserved:
  * fit-maps saves each chain's own Sigma_inv, AND
  * `sample --shared-sigma-inv` (DEFAULT) makes every chain use map_00's
    Sigma_inv while keeping its own x_ref, so all chains share one kernel
    and differ only in where they start -- the cleaner convergence test.
  * `sample --per-chain-sigma-inv` uses each chain's own, i.e. N fully
    independent runs of the whole pipeline.
ZigZag has no reference measure and is unaffected by this flag; it only
ever uses x_ref as its x0.

Usage (two steps -- fit once, then sample as many chains as you like):

    # 1. fit 4 references (reports pairwise separation, warns if too close)
    python -m sazz.gpu_friendly.scripts.uci_deep_wide_convergence fit-maps \\
        --datasets boston --splits 0 --n-maps 4

    # 2a. one chain at a time (resumable; run these sequentially or on
    #     separate nodes)
    python -m sazz.gpu_friendly.scripts.uci_deep_wide_convergence sample \\
        --datasets boston --splits 0 --chains 0 \\
        --samplers grid_sticky_boomerang \\
        --n-skeleton 1_000_000 --stage-size 5_000

    # 2b. or all four in one invocation
    python -m sazz.gpu_friendly.scripts.uci_deep_wide_convergence sample \\
        --datasets boston --splits 0 --chains 0 1 2 3 \\
        --samplers grid_sticky_zigzag grid_sticky_boomerang \\
        --n-skeleton 1_000_000 --stage-size 5_000

On-disk layout (under --out, default results/paper/deep_wide_convergence):

    <out>/deep_wide/boston/split_00/maps/map_00.pt        <- fit-maps
    <out>/deep_wide/boston/split_00/maps/map_01.pt
    ...
    <out>/deep_wide/boston/split_00/chain_0/grid_sticky_boomerang.pt   <- sample
    <out>/deep_wide/boston/split_00/chain_1/grid_sticky_boomerang.pt
    ...

Each chain_N/<sampler>.pt is a normal save_run payload (so existing
analysis code reads it unchanged) with chain_id / map_seed / x_ref
provenance patched in.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

import sazz.gpu_friendly.scripts.deep_wide_uci as dw
import sazz.gpu_friendly.scripts.uci_bnn_grid as uci_grid

from sazz.gpu_friendly.scripts.uci_bnn_grid import (
    DEVICE, DTYPE, BASE_SEED, N_FISHER, REFERENCE,
    HIDDEN_VARIANTS, UCI_DATASETS,
    BNNConfig, configs_for,
    load_raw_datasets, make_split,
)
from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.neural_networks import FFN
from sazz.gpu_friendly.models.priors import build_fan_in_prior_precision
from sazz.gpu_friendly.utils.warmup import find_reference_bnn, find_reference_bnn_ggn


# ===========================================================================
# Config
# ===========================================================================

# Chain i's MAP is fitted under torch.manual_seed(MAP_SEED_BASE + 1000*split
# + i). Deliberately disjoint from BASE_SEED (42) + split_id, which
# deep_wide_uci.py already uses for its single-reference runs, so chain 0
# here is NOT accidentally the same fit as a plain deep_wide_uci.py run --
# every chain in this script is a fresh draw.
MAP_SEED_BASE = 90_000

# Pairwise MAP separation below this (relative: ||a-b|| / max(||a||,||b||))
# triggers a warning from fit-maps. Two references this close are not
# meaningfully overdispersed starts and the convergence test built on them
# would be weak. Not an error -- for a near-convex target it could be
# legitimate, and that is the user's call to make.
MAP_MIN_REL_SEPARATION = 0.05

DEFAULT_N_MAPS = 4

OUT_DIR = Path("results/paper/deep_wide_convergence")

# This script covers only the sticky PDMP families. NUTS/NUTS-HS are
# excluded deliberately: they do their own multi-chain warmup internally
# (NUTS_CHAINS in uci_bnn_grid.py) and do not consume x_ref at all, so
# "run NUTS from MAP_i" is not a meaningful instruction.
SAMPLER_NAMES = dw.PDMP_SAMPLERS


# ===========================================================================
# Paths
# ===========================================================================

def variant_root(out_dir: Path, hidden_variant: str) -> Path:
    """<out>/<variant>/ for every variant except 'small' (flat <out>/),
    matching uci_bnn_grid.py / deep_wide_uci.py's on-disk convention."""
    return out_dir if hidden_variant == "small" else out_dir / hidden_variant


def split_root(out_dir: Path, hidden_variant: str, dataset: str, split_id: int) -> Path:
    return variant_root(out_dir, hidden_variant) / dataset / f"split_{split_id:02d}"


def maps_dir(out_dir: Path, hidden_variant: str, dataset: str, split_id: int) -> Path:
    return split_root(out_dir, hidden_variant, dataset, split_id) / "maps"


def map_path(out_dir: Path, hidden_variant: str, dataset: str, split_id: int,
             map_id: int) -> Path:
    return maps_dir(out_dir, hidden_variant, dataset, split_id) / f"map_{map_id:02d}.pt"


def chain_dir(out_dir: Path, hidden_variant: str, dataset: str, split_id: int,
              chain_id: int) -> Path:
    return split_root(out_dir, hidden_variant, dataset, split_id) / f"chain_{chain_id}"


# ===========================================================================
# Target construction -- build_target split so the MAP can be re-seeded and
# re-fitted without rebuilding the (deterministic) BayesianModule.
# ===========================================================================

def build_bm(data: dict[str, Any], cfg: BNNConfig, dtype=DTYPE, device=DEVICE) -> BayesianModule:
    """The BayesianModule half of uci_bnn_grid.py::build_target, verbatim.

    Deterministic given (data, cfg): FFN(...) constructs its own parameters
    but bm only ever uses their SHAPES (via bm.param_dict_fn / bm.D), never
    their values -- every sampler and the MAP finder supply the flat
    parameter vector themselves -- so no seeding is needed here.
    """
    X = data["X_train"].to(dtype=dtype, device=device)
    y = data["y_train"].to(dtype=dtype, device=device)

    module = FFN(cfg.layer_sizes, cfg.activation)
    prec = build_fan_in_prior_precision(
        module, cfg.prior_std_weight, cfg.prior_std_bias,
        cfg.fan_in_scaling, dtype=dtype, device=device,
    )
    # noise_std omitted -> learned, HalfNormal(prior_sigma_scale) prior.
    return BayesianModule.build(
        module, likelihood="gaussian", X=X, y=y,
        prior_precision=prec, prior_sigma_scale=cfg.prior_sigma_scale,
        dtype=dtype, device=device,
    )


def fit_one_reference(bm: BayesianModule, cfg: BNNConfig, seed: int,
                      dtype=DTYPE, device=DEVICE) -> tuple[torch.Tensor, torch.Tensor]:
    """One (x_ref, Sigma_inv) from a seed-determined random init.

    find_reference_bnn takes no seed and starts from torch.randn(bm.D), so
    seeding the ambient RNG here is what makes chain i's MAP differ from
    chain j's. Same finder, same n_steps/lr/n_fisher as build_target.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2**31 - 1))
    finder = {
        "empirical_fisher": find_reference_bnn,
        "ggn": find_reference_bnn_ggn,
    }[REFERENCE]
    return finder(
        bm, n_steps=cfg.adam_steps, lr=1e-2, n_fisher_batch=N_FISHER,
        dtype=dtype, device=torch.device(device),
    )


# ===========================================================================
# fit-maps
# ===========================================================================

def _separation_report(x_refs: list[torch.Tensor], energies: list[float],
                        seeds: list[int]) -> bool:
    """Print pairwise MAP separation + energies. Returns True if every pair
    is at least MAP_MIN_REL_SEPARATION apart.

    Relative distance ||a-b|| / max(||a||,||b||) rather than raw L2: at
    D ~ 45,000 a raw norm is uninformative without a scale to compare to.
    Energies come along because two DISTINCT modes at very different
    energies are also a warning sign -- a chain started in a much worse
    mode may just be measuring how long it takes to leave it.
    """
    n = len(x_refs)
    print("\n  MAP summary:")
    print(f"    {'id':>3}  {'seed':>7}  {'energy':>14}  {'||x_ref||':>12}")
    for i, (x, e, s) in enumerate(zip(x_refs, energies, seeds)):
        print(f"    {i:>3}  {s:>7}  {e:>14.6g}  {float(x.norm()):>12.6g}")

    if n < 2:
        return True

    print("\n  pairwise separation (relative L2, ||a-b|| / max(||a||,||b||)):")
    ok = True
    for i in range(n):
        for j in range(i + 1, n):
            d = float((x_refs[i] - x_refs[j]).norm())
            scale = max(float(x_refs[i].norm()), float(x_refs[j].norm()), 1e-12)
            rel = d / scale
            flag = "" if rel >= MAP_MIN_REL_SEPARATION else "   <-- TOO CLOSE"
            if rel < MAP_MIN_REL_SEPARATION:
                ok = False
            print(f"    map_{i:02d} vs map_{j:02d}:  L2 {d:>12.6g}   rel {rel:>8.4f}{flag}")

    e_spread = max(energies) - min(energies)
    e_scale = max(abs(min(energies)), 1.0)
    print(f"\n  energy spread across MAPs: {e_spread:.6g} "
          f"({e_spread / e_scale:.2%} of |min energy|)")

    if not ok:
        print(
            f"\n  WARNING: at least one MAP pair is closer than "
            f"{MAP_MIN_REL_SEPARATION:.0%} relative L2. Chains started from "
            f"these are not meaningfully overdispersed, so between-chain "
            f"agreement will understate non-convergence. Consider re-running "
            f"fit-maps with a different --map-seed-base, or more --n-maps and "
            f"keeping the most separated subset."
        )
    return ok


def cmd_fit_maps(args) -> None:
    raw = load_raw_datasets(tuple(args.datasets))
    input_dims = {k: v[0].shape[1] for k, v in raw.items()}
    hidden = HIDDEN_VARIANTS[args.hidden_variant]
    cfgs = configs_for(input_dims, hidden, args.hidden_variant,
                       prior_inclusion_weight=args.prior_inclusion_weight)

    for ds in args.datasets:
        X, y = raw[ds]
        cfg = cfgs[ds]
        for split_id in args.splits:
            data = make_split(X, y, seed=BASE_SEED + split_id, dtype=DTYPE, device=DEVICE)
            bm = build_bm(data, cfg)

            md = maps_dir(args.out, args.hidden_variant, ds, split_id)
            md.mkdir(parents=True, exist_ok=True)

            print(f"\n--- fit-maps | {ds.upper()} split {split_id:02d} | "
                  f"layers={cfg.layer_sizes} | D={bm.D} | n_maps={args.n_maps} ---")

            x_refs, energies, seeds = [], [], []
            for map_id in range(args.n_maps):
                out_path = map_path(args.out, args.hidden_variant, ds, split_id, map_id)
                seed = args.map_seed_base + 1000 * split_id + map_id

                if args.resume and out_path.exists():
                    payload = torch.load(out_path, weights_only=False)
                    print(f"  [map_{map_id:02d}] skipping — exists at {out_path}")
                    x_refs.append(payload["x_ref"].to(dtype=DTYPE))
                    energies.append(float(payload["energy"]))
                    seeds.append(int(payload["map_seed"]))
                    continue

                t0 = time.perf_counter()
                x_ref, Sigma_inv = fit_one_reference(bm, cfg, seed)
                elapsed = time.perf_counter() - t0
                with torch.no_grad():
                    energy = float(bm.energy(x_ref))

                torch.save({
                    "dataset": ds,
                    "split_id": split_id,
                    "map_id": map_id,
                    "map_seed": seed,
                    "x_ref": x_ref.cpu(),
                    "Sigma_inv": Sigma_inv.cpu(),
                    "energy": energy,
                    "layer_sizes": cfg.layer_sizes,
                    "activation": cfg.activation,
                    "hidden_variant": args.hidden_variant,
                    "adam_steps": cfg.adam_steps,
                    "reference": REFERENCE,
                    "n_fisher": N_FISHER,
                    "D": int(bm.D),
                    "elapsed_sec": elapsed,
                }, out_path)
                print(f"  [map_{map_id:02d}] seed={seed}  energy={energy:.6g}  "
                      f"({elapsed:.1f}s) -> {out_path}")

                x_refs.append(x_ref)
                energies.append(energy)
                seeds.append(seed)

            _separation_report(x_refs, energies, seeds)


# ===========================================================================
# sample
# ===========================================================================

def _load_reference(out_dir: Path, hidden_variant: str, dataset: str, split_id: int,
                    chain_id: int, shared_sigma_inv: bool
                    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """(x_ref, Sigma_inv, provenance) for one chain.

    shared_sigma_inv=True takes Sigma_inv from map_00 while keeping this
    chain's own x_ref, so every chain runs the SAME Boomerang kernel and
    differs only in start point -- see the module docstring.
    """
    p = map_path(out_dir, hidden_variant, dataset, split_id, chain_id)
    if not p.exists():
        raise FileNotFoundError(
            f"No reference for chain {chain_id} at {p}. Run `fit-maps "
            f"--datasets {dataset} --splits {split_id} --n-maps {chain_id + 1}` "
            f"(or more) first."
        )
    payload = torch.load(p, weights_only=False)
    x_ref = payload["x_ref"].to(dtype=DTYPE, device=DEVICE)

    if shared_sigma_inv:
        p0 = map_path(out_dir, hidden_variant, dataset, split_id, 0)
        if not p0.exists():
            raise FileNotFoundError(
                f"--shared-sigma-inv needs map_00 at {p0} (it supplies the "
                f"common reference measure), but it does not exist."
            )
        src = torch.load(p0, weights_only=False)
        Sigma_inv = src["Sigma_inv"].to(dtype=DTYPE, device=DEVICE)
        sigma_src = 0
    else:
        Sigma_inv = payload["Sigma_inv"].to(dtype=DTYPE, device=DEVICE)
        sigma_src = chain_id

    prov = {
        "chain_id": chain_id,
        "map_seed": int(payload["map_seed"]),
        "map_energy": float(payload["energy"]),
        "sigma_inv_from_map": sigma_src,
        "shared_sigma_inv": bool(shared_sigma_inv),
    }
    return x_ref, Sigma_inv, prov


def cmd_sample(args) -> None:
    # deep_wide_uci.py's runners read these as MODULE GLOBALS on dw, so the
    # CLI values have to be pushed there rather than passed down.
    dw.N_SKELETON = args.n_skeleton
    dw.N_RESAMPLE = args.n_resample
    dw.STAGE_SIZE = args.stage_size
    dw.GRAD_BATCH_SIZE = args.grad_batch_size
    dw.TIME_WEIGHTED_RESAMPLE = not args.equal_draws_per_stage
    dw.POOL_PER_STAGE = args.pool_per_stage
    dw.STAGE_DIR = args.stage_dir if args.stage_dir is not None else (args.out / "chunks")

    if args.stage_size is None:
        raise SystemExit(
            "--stage-size is required (the staged sticky runners assert on it). "
            "e.g. --stage-size 5000"
        )

    raw = load_raw_datasets(tuple(args.datasets))
    input_dims = {k: v[0].shape[1] for k, v in raw.items()}
    hidden = HIDDEN_VARIANTS[args.hidden_variant]
    cfgs = configs_for(input_dims, hidden, args.hidden_variant,
                       prior_inclusion_weight=args.prior_inclusion_weight)

    for ds in args.datasets:
        X, y = raw[ds]
        cfg = cfgs[ds]
        for split_id in args.splits:
            data = make_split(X, y, seed=BASE_SEED + split_id, dtype=DTYPE, device=DEVICE)
            bm = build_bm(data, cfg)

            n_stages = math.ceil(dw.N_SKELETON / dw.STAGE_SIZE)
            print(f"\n--- sample | {ds.upper()} split {split_id:02d} | "
                  f"layers={cfg.layer_sizes} | D={bm.D} | chains={args.chains} | "
                  f"{dw.N_SKELETON} events x {n_stages} stages | "
                  f"sigma_inv={'shared(map_00)' if args.shared_sigma_inv else 'per-chain'} ---")

            for chain_id in args.chains:
                cd = chain_dir(args.out, args.hidden_variant, ds, split_id, chain_id)
                pending = [s for s in args.samplers
                           if not (args.resume and (cd / f"{s}.pt").exists())]
                for s in args.samplers:
                    if s not in pending:
                        print(f"  [chain {chain_id}][{s}] skipping — exists at {cd / f'{s}.pt'}")
                if not pending:
                    continue

                x_ref, Sigma_inv, prov = _load_reference(
                    args.out, args.hidden_variant, ds, split_id, chain_id,
                    args.shared_sigma_inv,
                )
                cd.mkdir(parents=True, exist_ok=True)
                print(f"  [chain {chain_id}] x_ref from map_{chain_id:02d} "
                      f"(seed={prov['map_seed']}, energy={prov['map_energy']:.6g}), "
                      f"Sigma_inv from map_{prov['sigma_inv_from_map']:02d}")

                # Stage dirs must not collide between concurrently-running
                # chains, so the chain id goes into the transient path too.
                dw.STAGE_DIR = (args.stage_dir if args.stage_dir is not None
                                else args.out / "chunks") / f"chain_{chain_id}"

                for sampler_name in pending:
                    print(f"  [chain {chain_id}][{sampler_name}]")
                    # Sampler randomness is tied to the chain, so two chains
                    # never share a trajectory even from an identical start.
                    seed = BASE_SEED + split_id + args.chain_seed_offset * chain_id
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    dw.SAMPLER_RUNNERS[sampler_name](
                        ds, split_id, data, cfg, cd, bm, x_ref, Sigma_inv,
                    )

                    # Chain provenance. save_run is imported verbatim from
                    # uci_bnn_grid.py and knows nothing about chains, so these
                    # are patched in afterwards (same pattern deep_wide_uci.py
                    # already uses for its staging fields).
                    out_path = cd / f"{sampler_name}.pt"
                    payload = torch.load(out_path, weights_only=False)
                    payload.update(prov)
                    payload["chain_seed"] = seed
                    payload["n_maps_available"] = len(
                        list(maps_dir(args.out, args.hidden_variant, ds, split_id)
                             .glob("map_*.pt"))
                    )
                    torch.save(payload, out_path)


# ===========================================================================
# CLI
# ===========================================================================

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--datasets", nargs="+", default=["boston"],
                   choices=list(UCI_DATASETS))
    p.add_argument("--splits", nargs="+", type=int, default=[0])
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--hidden-variant", choices=list(HIDDEN_VARIANTS),
                   default="deep_wide",
                   help="Defaults to deep_wide (this script's reason to exist). "
                        "Results go under <out>/<variant>/... for every variant "
                        "except 'small' (flat <out>/...).")
    p.add_argument("--prior-inclusion-weight", type=float, default=0.3,
                   help="Sticky spike-and-slab inclusion prob feeding kappa. Must "
                        "match between fit-maps and sample (it is part of cfg).")
    p.add_argument("--resume", action="store_true")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- fit-maps ---------------------------------------------------------
    pf = sub.add_parser("fit-maps", formatter_class=argparse.RawDescriptionHelpFormatter,
                        help="Fit N independent references (MAP + diagonal Laplace) "
                             "and save them as map_XX.pt.")
    _add_common(pf)
    pf.add_argument("--n-maps", type=int, default=DEFAULT_N_MAPS,
                    help=f"How many references to fit (default {DEFAULT_N_MAPS}). "
                         f"Chain i later uses map_i, so fit at least as many as you "
                         f"intend to run chains.")
    pf.add_argument("--map-seed-base", type=int, default=MAP_SEED_BASE,
                    help="map_i is fitted under torch.manual_seed(base + 1000*split + i). "
                         "Change it to get a different set of MAPs entirely (e.g. if the "
                         "separation report flags two as too close).")
    pf.set_defaults(func=cmd_fit_maps)

    # --- sample -----------------------------------------------------------
    ps = sub.add_parser("sample", formatter_class=argparse.RawDescriptionHelpFormatter,
                        help="Run one staged chain per --chains entry, each from its "
                             "own saved reference.")
    _add_common(ps)
    ps.add_argument("--chains", nargs="+", type=int, default=[0],
                    help="Which chains to run; chain i starts from map_i. Pass one id "
                         "to run a single chain (resumable, so chains can be run "
                         "sequentially or split across nodes), or several to do them "
                         "back-to-back in one process.")
    ps.add_argument("--samplers", nargs="+", default=list(SAMPLER_NAMES),
                    choices=list(SAMPLER_NAMES),
                    help="Sticky PDMP families only. NUTS/NUTS-HS are excluded: they "
                         "do their own internal multi-chain warmup and never consume "
                         "x_ref, so 'run NUTS from MAP_i' is not meaningful.")

    sig = ps.add_mutually_exclusive_group()
    sig.add_argument("--shared-sigma-inv", dest="shared_sigma_inv",
                     action="store_true", default=True,
                     help="(default) Every chain uses map_00's Sigma_inv with its OWN "
                          "x_ref, so all chains are the same Markov kernel started at "
                          "different points -- the cleaner convergence test. No effect "
                          "on ZigZag, which has no reference measure.")
    sig.add_argument("--per-chain-sigma-inv", dest="shared_sigma_inv",
                     action="store_false",
                     help="Each chain uses its own (x_ref, Sigma_inv), i.e. N fully "
                          "independent runs of the pipeline. Boomerang chains are then "
                          "different kernels, so cross-chain R-hat is harder to read.")

    ps.add_argument("--chain-seed-offset", type=int, default=10_000,
                    help="Chain i's sampler RNG is seeded BASE_SEED + split + "
                         "offset*i, so two chains never share a trajectory even from "
                         "an identical start. Set 0 to make sampler randomness "
                         "identical across chains (isolating the effect of the start "
                         "point alone).")

    # Staging / resampling -- same semantics as deep_wide_uci.py's.
    ps.add_argument("--n-skeleton", type=int, default=dw.N_SKELETON,
                    help="Total skeleton events per chain (summed across stages).")
    ps.add_argument("--n-resample", type=int, default=dw.N_RESAMPLE)
    ps.add_argument("--stage-size", "--skeleton-chunk-size", type=int, default=None,
                    dest="stage_size",
                    help="Skeleton events per stage. REQUIRED. Peak disk is "
                         "O(stage-size * D), not O(n-skeleton * D).")
    ps.add_argument("--stage-dir", "--skeleton-chunk-dir", type=Path, default=None,
                    dest="stage_dir",
                    help="Transient stage dirs. Default <out>/chunks. A chain_N/ level "
                         "is appended automatically so concurrent chains never collide.")
    ps.add_argument("--pool-per-stage", type=int, default=dw.POOL_PER_STAGE)
    ps.add_argument("--equal-draws-per-stage", action="store_true",
                    help="Revert to per-stage-uniform resampling instead of "
                         "time-weighted pooling (see deep_wide_uci.py).")
    ps.add_argument("--grad-batch-size", type=int, default=None,
                    help="Opt into minibatched gradients. Default None = full batch, "
                         "matching deep_wide_uci.py and keeping the chains comparable "
                         "to the existing single-chain results.")
    ps.set_defaults(func=cmd_sample)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
