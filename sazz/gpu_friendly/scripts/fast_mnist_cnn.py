"""
Usage:
    python -m sazz.gpu_friendly.scripts.mnist_cnn_grid --samplers grid_sticky_zigzag
    python -m sazz.gpu_friendly.scripts.mnist_cnn_grid --map-path results/maps/cnn_reference_N60000_steps10000.pt
    python -m sazz.gpu_friendly.scripts.mnist_cnn_grid --map-path results/maps/lenet_reference_N60000_steps10000.pt
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchvision import datasets, transforms

from sazz.gpu_friendly.models.neural_networks import CNN, LeNet5
from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.priors import (
    build_fan_in_prior_precision, build_kappa_from_inclusion, build_can_freeze_mask,
)
from sazz.gpu_friendly.utils.warmup import find_reference_bnn
from sazz.gpu_friendly.utils.resample import (
    resample_zigzag_path_sticky_torch, resample_boomerang_path_sticky_torch,
    resample_zigzag_path_sticky_chunked_torch, resample_boomerang_path_sticky_chunked_torch,
)
from sazz.gpu_friendly.samplers.fast_grid_sticky_zigzag import FastGridStickyZigZagSampler
from sazz.gpu_friendly.samplers.fast_grid_sticky_boomerang import FastGridStickyBoomerangSampler
from sazz.gpu_friendly.scripts.cnn_reference import eval_accuracy

ARCHITECTURES = {"cnn": CNN, "lenet5": LeNet5}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64
torch.set_default_dtype(torch.float32)

BASE_SEED = 42

N_TRAIN, N_TEST = 60_000, 500
N_SKELETON = 50
N_RESAMPLE = 2000
N_SAVE = 500

BURNIN_FRAC = 0.2
N_ACCURACY_DRAWS = 300      # subsample for softmax-averaging specifically 

PRIOR_STD_W = 2.0
PRIOR_STD_B = 2.0
PRIOR_INCLUSION_WEIGHT = 0.1
ACTIVATION = "tanh"
POOL = "avg"

PRUNE_ACC_DROP_TOLERANCE = 0.01   # max acceptable (baseline_acc - pruned_acc)
PRUNE_N_THRESHOLDS = 60
N_SWEEP = 2000 

GRID_CHUNK_SIZE = 4
GAMMA = 1e-6
GRID_T_MAX_INIT_ZIGZAG = 0.01
GRID_SPACING_ZIGZAG = 1e-5
GRID_T_MAX_INIT_BOOM = math.pi / 4  
GRID_SPACING_BOOM = math.pi / 64    
GRID_N_SEGMENTS = 100
GRID_ALPHA_PLUS = 1.02
GRID_ALPHA_MINUS = 1.04
GRID_ALPHA_VIOLATION = 1.1

# Skeleton-chunking knobs (new -- see radiant-finding-church.md). None
# disables chunking entirely (sample()'s original full-[N,D] behavior).
SKELETON_CHUNK_SIZE: Optional[int] = None
SKELETON_CHUNK_DIR: Optional[Path] = None

DATA_DIR = Path("datasets")
OUT_DIR = Path("results/grid/mnist_cnn")

SAMPLER_NAMES = ("grid_sticky_zigzag", "grid_sticky_boomerang")


@dataclass
class CNNConfig:
    activation: str = ACTIVATION
    pool: str = POOL
    prior_std_weight: float = PRIOR_STD_W
    prior_std_bias: float = PRIOR_STD_B
    fan_in_scaling: bool = True
    #adam_steps: int = 2000       # fallback find_reference_bnn path only --
                                    # the normal path is cnn_reference.py + --map-path
    prior_inclusion_weight: float = PRIOR_INCLUSION_WEIGHT


# ===========================================================================
# Data
# ===========================================================================

def load_mnist_subset(n_train: int, n_test: int, seed: int, data_dir: Path,
                       dtype: torch.dtype, device: str, n_sweep: int = 0) -> dict[str, Any]:
    """
    Converts X_train/y_train to (dtype, device) HERE, before BayesianModule.
    build is ever called. This matters: BayesianModule.build's categorical/
    bernoulli branches capture the raw X/y arguments into the likelihood
    closure BEFORE its own internal .to(dtype, device) call (which only
    produces bm.X/bm.y, a SEPARATE object from what the closure evaluates
    against) -- confirmed directly by reading model.py. On DEVICE=cuda,
    DTYPE=float32 this would otherwise silently hit a dtype/device mismatch
    on every energy() call unless the caller pre-converts, exactly as
    uci_bnn_grid.py's own build_target already does.

    n_sweep (if > 0): also draws X_sweep/y_sweep, n_sweep points from
    MNIST's TEST pool, disjoint from test_idx by construction (test_idx and
    sweep_idx are carved from a single permutation of test_full, not drawn
    independently). Used by prune_x_ref to pick the x_ref prune threshold
    on data that is BOTH held out from this script's own X_test AND
    (since it's drawn from the TEST pool, never touched by any
    cnn_reference.py/lenet_reference.py checkpoint's train/eval split, which
    only ever draws from MNIST's TRAIN pool) disjoint from whatever data a
    --map-path checkpoint's x_ref was fit on -- see prune_x_ref for why a
    slice of X_train would not satisfy either property.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_full = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(train_full), size=n_train, replace=False)

    test_pool_idx = rng.permutation(len(test_full))
    test_idx = test_pool_idx[:n_test]
    sweep_idx = test_pool_idx[n_test:n_test + n_sweep]

    def stack(ds, idx):
        Xs, ys = zip(*(ds[int(i)] for i in idx))
        X = torch.stack(Xs).to(dtype=dtype, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        return X, y

    X_train, y_train = stack(train_full, train_idx)
    X_test, y_test = stack(test_full, test_idx)

    print(f"  loaded MNIST subset: train={tuple(X_train.shape)}, test={tuple(X_test.shape)}")
    print(f"  train label counts:  {torch.bincount(y_train, minlength=10).tolist()}")
    out = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

    if n_sweep > 0:
        X_sweep, y_sweep = stack(test_full, sweep_idx)
        print(f"  loaded prune-sweep subset: sweep={tuple(X_sweep.shape)} "
              f"(disjoint from test, drawn from MNIST's TEST pool)")
        out["X_sweep"] = X_sweep
        out["y_sweep"] = y_sweep

    return out


# ===========================================================================
# Target builder
# ===========================================================================

def build_target(data: dict[str, Any], cfg: CNNConfig, dtype=DTYPE, device=DEVICE,
                  map_path: Optional[Path] = None, architecture: str = "cnn"):
    """
    Dispatches on the checkpoint's own `architecture` field (defaults to
    "cnn" for older checkpoints / the no-`--map-path` fallback, which has
    no checkpoint to dispatch from -- see the `architecture` parameter,
    settable via --architecture). Must load the checkpoint BEFORE building
    `module`/`bm`, not after -- picking the right nn.Module class is a
    precondition for BayesianModule.build, not something validated
    afterward like the other mismatch checks below.
    """
    if map_path is not None:
        print(f"\n  Loading reference checkpoint from {map_path} ...")
        ckpt = torch.load(map_path, weights_only=False)
        architecture = ckpt.get("architecture", "cnn")

    module_cls = ARCHITECTURES[architecture]
    module = module_cls(activation=cfg.activation, pool=cfg.pool)
    prec = build_fan_in_prior_precision(
        module, cfg.prior_std_weight, cfg.prior_std_bias,
        cfg.fan_in_scaling, dtype=dtype, device=device,
    )
    bm = BayesianModule.build(
        module, likelihood="categorical",
        X=data["X_train"], y=data["y_train"],
        prior_precision=prec, dtype=dtype, device=device,
    )

    if map_path is not None:
        mismatches = []
        if ckpt["D"] != bm.D:
            mismatches.append(f"D: ckpt={ckpt['D']} vs target={bm.D}")
        if ckpt.get("activation") != cfg.activation:
            mismatches.append(f"activation: ckpt={ckpt.get('activation')} vs target={cfg.activation}")
        if ckpt.get("pool") != cfg.pool:
            mismatches.append(f"pool: ckpt={ckpt.get('pool')} vs target={cfg.pool}")
        if ckpt.get("prior_std_weight") != cfg.prior_std_weight:
            mismatches.append(f"prior_std_weight: ckpt={ckpt.get('prior_std_weight')} vs target={cfg.prior_std_weight}")
        if ckpt.get("prior_std_bias") != cfg.prior_std_bias:
            mismatches.append(f"prior_std_bias: ckpt={ckpt.get('prior_std_bias')} vs target={cfg.prior_std_bias}")
        if ckpt.get("fan_in_scaling") != cfg.fan_in_scaling:
            mismatches.append(f"fan_in_scaling: ckpt={ckpt.get('fan_in_scaling')} vs target={cfg.fan_in_scaling}")
        assert not mismatches, (
            "Checkpoint is not a valid reference for this target -- mismatched fields:\n  "
            + "\n  ".join(mismatches)
        )
        x_ref = ckpt["x_ref"].to(dtype=dtype, device=device)
        Sigma_inv = ckpt["Sigma_inv"].to(dtype=dtype, device=device)
        cold_start_mask = ckpt.get("cold_start_mask")
        if cold_start_mask is not None:
            cold_start_mask = cold_start_mask.to(dtype=torch.bool, device=device)
        print(f"  loaded  architecture={architecture}  sigma_inv_source={ckpt.get('sigma_inv_source', 'UNKNOWN')}  "
              f"||x_ref||_inf={x_ref.abs().max():.3f}  "
              f"(train_acc={ckpt.get('train_acc', float('nan')):.3f})"
              + ("  [pre-pruned+refit checkpoint]" if cold_start_mask is not None else ""))

        # Sigma_inv = prior_precision + N_ckpt * fisher_mean was computed at
        # checkpoint time against N_ckpt = the CHECKPOINT's own training-set
        # size (cnn_reference.py::empirical_fisher_diag_scaled), not this
        # script's bm. Reusing it as-is overstates curvature in
        # Fisher-dominated coordinates by N_ckpt/N_bm -- rescale to the data
        # bm was actually built on (same convention, N = bm.X.shape[0]) so
        # the two are definitionally consistent. Keyed off bm.X.shape[0],
        # NOT the module-level N_TRAIN constant: --n-train only threads
        # into load_mnist_subset, N_TRAIN itself is never reassigned.
        N_bm = bm.X.shape[0]
        N_ckpt = ckpt["n_train"]
        ratio = N_bm / N_ckpt
        Sigma_inv = bm.prior_precision + ratio * (Sigma_inv - bm.prior_precision)
        print(f"  Sigma_inv rescale: n_train ckpt={N_ckpt} vs this run's bm.X={N_bm}  "
              f"-> Fisher term scaled by {ratio:.4f} (i.e. checkpoint's curvature "
              f"estimate divided by {1.0 / ratio:.1f}x, since it was fit on "
              f"{1.0 / ratio:.1f}x more data than this run uses)")
    else:
        print(
            "\n  WARNING: no --map-path given, falling back to "
            "utils/warmup.py::find_reference_bnn. Its MAP step starts from "
            "torch.randn(bm.D) (unit variance), which is far outside this "
            "target's fan-in-scaled prior std and known suboptimal -- run "
            "cnn_reference.py/lenet_reference.py once and pass --map-path "
            "for a real reference. Proceeding anyway, but expect a "
            "poor/near-chance MAP."
        )
        x_ref, Sigma_inv = find_reference_bnn(
            bm, n_steps=cfg.adam_steps, lr=1e-2, dtype=dtype, device=torch.device(device),
        )
        cold_start_mask = None

    return bm, x_ref, Sigma_inv, cold_start_mask


# ===========================================================================
# Samplers
# ===========================================================================

def _build_sticky_kappa_can_freeze(bm: BayesianModule, cfg: CNNConfig):
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


def build_sticky_zigzag_sampler(bm: BayesianModule, cfg: CNNConfig, cold_start_mask: Tensor):
    kappa, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
    return FastGridStickyZigZagSampler(
        grad_target=torch.func.grad(bm.energy),
        D=bm.D,
        kappa=kappa,
        can_freeze=can_freeze,
        cold_start_threshold=cold_start_mask,
        gamma=GAMMA,
        grid_t_max_init=GRID_T_MAX_INIT_ZIGZAG,
        n_segments=GRID_N_SEGMENTS,
        grid_spacing=GRID_SPACING_ZIGZAG,
        alpha_plus=GRID_ALPHA_PLUS,
        alpha_minus=GRID_ALPHA_MINUS,
        alpha_violation=GRID_ALPHA_VIOLATION,
        chunk_size=GRID_CHUNK_SIZE,
        dtype=DTYPE,
        device=bm.device,
    )


def build_sticky_boomerang_sampler(bm: BayesianModule, cfg: CNNConfig,
                                    x_ref: Tensor, Sigma_inv: Tensor, cold_start_mask: Tensor):
    kappa, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
    sampler = FastGridStickyBoomerangSampler(
        grad_target=torch.func.grad(bm.energy),
        D=bm.D,
        kappa=kappa,
        can_freeze=can_freeze,
        cold_start_threshold=cold_start_mask,
        grid_spacing=GRID_SPACING_BOOM,
        refresh_rate=1.0,
        grid_t_max_init=GRID_T_MAX_INIT_BOOM,
        n_segments=GRID_N_SEGMENTS,
        alpha_plus=GRID_ALPHA_PLUS,
        alpha_minus=GRID_ALPHA_MINUS,
        alpha_violation=GRID_ALPHA_VIOLATION,
        chunk_size=GRID_CHUNK_SIZE,
        dtype=DTYPE,
        device=bm.device,
    )
    sampler.preprocess(x_ref=x_ref, Sigma_inv=Sigma_inv)
    return sampler


# ===========================================================================
# Accuracy / diagnostics
# ===========================================================================

@torch.no_grad()
def evaluate_accuracy(bm: BayesianModule, samples: Tensor, X_test: Tensor, y_test: Tensor,
                       n_draws: int = N_ACCURACY_DRAWS) -> tuple[float, float]:
    """
    Average softmax probabilities across a SUBSAMPLE of posterior draws
    (not all of samples -- the posterior average converges well before
    N_RESAMPLE draws, and averaging softmax over tens of thousands of draws
    is unnecessary CNN-forward cost). Returns (test_accuracy, sparsity_frac)
    -- sparsity_frac uses the FULL samples tensor (cheap, no forward pass).
    """
    n = min(n_draws, samples.shape[0])
    idx = torch.randperm(samples.shape[0])[:n]
    sub = samples[idx]

    X_test = X_test.to(dtype=DTYPE, device=DEVICE)
    y_test = y_test.to(device=DEVICE)

    probs = []
    for beta in sub:
        logits = torch.func.functional_call(bm.module, bm.param_dict_fn(beta.to(DEVICE)), (X_test,))
        probs.append(torch.softmax(logits, dim=-1))
    mean_probs = torch.stack(probs).mean(0)
    pred_y = mean_probs.argmax(-1)
    acc = (pred_y == y_test).float().mean().item()

    sparsity = (samples.abs() < 1e-8).float().mean().item()
    return acc, sparsity


def prune_x_ref(bm: BayesianModule, x_ref: Tensor, X_sweep: Tensor, y_sweep: Tensor,
                 can_freeze: Tensor, acc_drop_tolerance: float = PRUNE_ACC_DROP_TOLERANCE,
                 n_thresholds: int = PRUNE_N_THRESHOLDS) -> tuple[Tensor, Tensor]:
    """
    Zero out x_ref coordinates below a per-coordinate threshold relative to
    that coordinate's own prior std (1/sqrt(prior_precision_i)) -- NOT one
    absolute threshold across all D, since fan-in scaling means weight
    scales differ hugely between e.g. conv1 and fc-type layers (an absolute
    threshold would just measure which layer happens to have larger
    weights). Picks the largest threshold (most sparsity) whose accuracy on
    X_sweep/y_sweep is still within acc_drop_tolerance of the unpruned
    baseline. Restricted to can_freeze coordinates throughout (biases never
    prune -- can_freeze is False for all bias coordinates), so the accuracy
    validated here matches the configuration the sampler actually holds:
    a bias coordinate the sampler moves off zero at t=0 must never count
    toward the validated prune fraction.

    X_sweep/y_sweep MUST be disjoint from both this run's own X_test (kept
    for final accuracy reporting) and from whatever data a --map-path
    checkpoint's x_ref was fit on -- see load_mnist_subset's n_sweep
    parameter, which draws sweep data from MNIST's TEST pool specifically
    because cnn_reference.py/lenet_reference.py checkpoints are only ever
    fit on MNIST's TRAIN pool. A slice of this script's own X_train would
    satisfy neither property (checkpoint fits and this script's X_train
    both sample from the same 60k MNIST train pool, so expected overlap at
    this script's default N_TRAIN is large) and would also be far too small
    for the tolerance to mean more than one or two images flipping.

    Returns (x_pruned, freeze_mask) -- freeze_mask (bool[D], True = frozen
    at cold start) is intended to be passed directly as a sticky sampler's
    cold_start_threshold.
    """
    prior_std = bm.prior_precision.clamp(min=1e-12).rsqrt()
    baseline_acc = eval_accuracy(bm, x_ref, X_sweep, y_sweep)

    sweep_multipliers = np.logspace(-3, 0, n_thresholds)
    best_frac, best_mask, best_acc, best_mult = 0.0, torch.zeros_like(can_freeze), baseline_acc, 0.0
    for m in sweep_multipliers:
        mask = (x_ref.abs() < m * prior_std) & can_freeze
        x_pruned = torch.where(mask, torch.zeros_like(x_ref), x_ref)
        acc = eval_accuracy(bm, x_pruned, X_sweep, y_sweep)
        if acc >= baseline_acc - acc_drop_tolerance:
            best_frac = float(mask.float().mean())
            best_mask = mask
            best_acc = acc
            best_mult = float(m)

    x_pruned = torch.where(best_mask, torch.zeros_like(x_ref), x_ref)
    n_freezable = int(can_freeze.sum())
    n_pruned = int(best_mask.sum())
    print(f"  prune_x_ref: threshold={best_mult:.4f}*std  "
          f"pruned {n_pruned}/{n_freezable} freezable coords ({best_frac:.3f} of freezable, "
          f"{n_pruned / bm.D:.3f} of all D)  params kept={bm.D - n_pruned}  "
          f"sweep_acc: baseline={baseline_acc:.4f} pruned={best_acc:.4f} "
          f"(drop={baseline_acc - best_acc:.4f}, tolerance={acc_drop_tolerance:.4f})")
    return x_pruned, best_mask


def print_preactivation_diagnostic(bm: BayesianModule, x_ref: Tensor, cfg: CNNConfig, n_prior_draws: int = 5):
    """
    Reports the fraction of pre-activations with |z|>2 at x_ref and a few
    prior draws -- diagnoses whether PRIOR_STD_W is putting tanh into its
    saturated regime (tanh'(2)~0.07, tanh'(5)~1.8e-4), independent of the
    target's actual curvature. A large fraction here means PRIOR_STD_W
    should be lowered before trusting downstream rate/equilibrium numbers.
    """
    hook_outputs = []

    def hook(module, inp, out):
        hook_outputs.append(inp[0].detach())

    handles = [bm.module.conv1.register_forward_hook(hook),
               bm.module.conv2.register_forward_hook(hook)]

    def frac_large(beta: Tensor) -> float:
        hook_outputs.clear()
        with torch.no_grad():
            torch.func.functional_call(bm.module, bm.param_dict_fn(beta), (bm.X[:32],))
        all_z = torch.cat([h.flatten() for h in hook_outputs])
        return (all_z.abs() > 2.0).float().mean().item()

    for h in handles:
        h.remove()

    # Re-register without removing immediately this time
    handles = [bm.module.conv1.register_forward_hook(hook),
               bm.module.conv2.register_forward_hook(hook)]
    frac_at_xref = frac_large(x_ref)

    fracs_prior = []
    D_network = bm.D - (1 if bm.learns_noise else 0)
    for _ in range(n_prior_draws):
        beta_prior = torch.randn(bm.D, dtype=DTYPE, device=DEVICE) / bm.prior_precision.clamp(min=1e-8).sqrt()
        fracs_prior.append(frac_large(beta_prior))

    for h in handles:
        h.remove()

    print(f"  pre-activation |z|>2 fraction: at x_ref={frac_at_xref:.3f}, "
          f"prior draws mean={np.mean(fracs_prior):.3f} "
          f"(prior_std_weight={cfg.prior_std_weight}) -- "
          f"large values mean tanh is saturated as an artifact of prior scale, "
          f"not target curvature")


# ===========================================================================
# Persistence
# ===========================================================================

def thin_to(samples: Tensor, n_keep: int) -> Tensor:
    n = samples.shape[0]
    if n <= n_keep:
        return samples
    idx = torch.linspace(0, n - 1, n_keep, device=samples.device).round().long()
    return samples[idx]


def save_run(out_path: Path, *, sampler: str, samples: Tensor, x_ref: Optional[Tensor],
             cfg: CNNConfig, elapsed_sec: float, n_events: int, bound_violations: int,
             gradient_evals: Optional[int] = None, grid_t_max_log: Optional[list[float]] = None,
             test_accuracy: Optional[float] = None, sparsity_frac: Optional[float] = None,
             prune_frac: Optional[float] = None, cold_start_mask: Optional[Tensor] = None,
             diagnostics: Optional[list[dict]] = None) -> None:
    """
    x_ref here is whatever the caller actually sampled from/anchored to --
    for the sticky runners this MUST be the pruned x_ref (x_pruned), not
    the raw checkpoint vector, so a saved run's recorded provenance matches
    the vector that was actually used.

    diagnostics: the full per-iteration diag_log list from sample()'s
    return dict (small dicts of floats/ints/strings, no tensors -- cheap,
    a few MB at typical N_SKELETON), or None. Only ever non-None for a
    non-chunked sample() call (chunk_size=None) -- chunked runs already
    flush their diag_log to chunk_dir/diag_*.pt periodically specifically
    to bound memory (see sample()'s docstring), so result["diagnostics"]
    is None in that mode and there is nothing here to save; the chunked
    diag_*.pt files are the analysis path for that case, not this field.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sampler": sampler,
        "samples": thin_to(samples, N_SAVE).cpu(),
        "x_ref": x_ref.cpu() if x_ref is not None else None,
        "activation": cfg.activation,
        "pool": cfg.pool,
        "n_events": n_events,
        "elapsed_sec": elapsed_sec,
        "bound_violations": bound_violations,
        "gradient_evals": gradient_evals,
        "grid_t_max_log": grid_t_max_log,
        "test_accuracy": test_accuracy,
        "sparsity_frac": sparsity_frac,
        "prune_frac": prune_frac,
        "cold_start_mask": cold_start_mask.cpu() if cold_start_mask is not None else None,
        "diagnostics": diagnostics,
    }, out_path)


def _resume_skip(out_path: Path, n_skeleton: int) -> bool:
    """
    --resume skips a sampler only if its output file exists AND was
    produced with the CURRENT N_SKELETON -- a plain existence check would
    let a stale smoke-test run (tiny N_SKELETON, meaningless accuracy)
    silently stand in for a real run after N_SKELETON is retuned upward.
    """
    if not out_path.exists():
        return False
    try:
        ckpt = torch.load(out_path, weights_only=False)
    except Exception:
        return False
    return ckpt.get("n_events") == n_skeleton


# ===========================================================================
# Per-sampler runners -- SAMPLER_RUNNERS dispatch calls every entry with the
# SAME signature (dataset_name, split_id, data, cfg, sd, bm, x_pruned,
# Sigma_inv, cold_start_mask). x_pruned/cold_start_mask come from
# prune_x_ref, called once per split in run_dataset -- NOT the raw
# checkpoint x_ref. For sticky Boomerang, x_pruned must reach every one of:
# the sampler builder (-> preprocess -> x_ref), sample(x0=...), and the
# resample_*_sticky_torch call -- see module docstring / plan for why a
# stale (unpruned) reference at any of those three sites reintroduces the
# bug pruning is meant to fix.
# ===========================================================================

def run_grid_sticky_zigzag(dataset_name: str, split_id: int, data: dict[str, Any], cfg: CNNConfig,
                            sd: Path, bm: BayesianModule, x_pruned: Tensor, Sigma_inv: Tensor,
                            cold_start_mask: Tensor) -> None:
    out_path = sd / "grid_sticky_zigzag.pt"
    if _resume_skip(out_path, N_SKELETON):
        print(f"      skipping — exists at {out_path}")
        return

    sampler = build_sticky_zigzag_sampler(bm, cfg, cold_start_mask)

    n_freezable = int(sampler.can_freeze.sum())
    n_frozen_init = int(cold_start_mask.sum())
    print(f"      cold start: {n_frozen_init}/{n_freezable} freezable coords "
          f"frozen ({100 * n_frozen_init / max(n_freezable, 1):.1f}%)")

    chunking_active = SKELETON_CHUNK_SIZE is not None
    sample_kwargs = {}
    if chunking_active:
        sample_kwargs["chunk_size"] = SKELETON_CHUNK_SIZE
        sample_kwargs["chunk_dir"] = SKELETON_CHUNK_DIR / dataset_name / f"split_{split_id:02d}" / "grid_sticky_zigzag"

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, x0=x_pruned, diagnostics=True, **sample_kwargs)
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
    acc, sparsity = evaluate_accuracy(bm, samples, data["X_test"], data["y_test"])

    final_sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, "
          f"final sparsity {final_sparsity:.2f}) "
          f"test_acc={acc:.3f} sample_sparsity={sparsity:.3f}")

    save_run(
        out_path, sampler="grid_sticky_zigzag", samples=samples, x_ref=x_pruned, cfg=cfg,
        elapsed_sec=elapsed, n_events=N_SKELETON,
        bound_violations=result["bound_violations"],
        gradient_evals=result["gradient_evals"], grid_t_max_log=result["grid_t_max_log"],
        test_accuracy=acc, sparsity_frac=sparsity,
        prune_frac=n_frozen_init / max(n_freezable, 1), cold_start_mask=cold_start_mask,
        diagnostics=result.get("diagnostics"),
    )
    print(f"      saved -> {out_path}")


def run_grid_sticky_boomerang(dataset_name: str, split_id: int, data: dict[str, Any], cfg: CNNConfig,
                               sd: Path, bm: BayesianModule, x_pruned: Tensor, Sigma_inv: Tensor,
                               cold_start_mask: Tensor) -> None:
    out_path = sd / "grid_sticky_boomerang.pt"
    if _resume_skip(out_path, N_SKELETON):
        print(f"      skipping — exists at {out_path}")
        return

    # x_pruned reaches the builder (-> preprocess -> self.x_ref), sample's
    # x0, AND the resample call below -- all three, not just one. See the
    # module docstring: an unpruned x_ref at preprocess() reintroduces a
    # permanent spurious term in the excess gradient at every frozen
    # coordinate, independent of what x0 is passed to sample().
    sampler = build_sticky_boomerang_sampler(bm, cfg, x_pruned, Sigma_inv, cold_start_mask)

    n_freezable = int(sampler.can_freeze.sum())
    n_frozen_init = int(cold_start_mask.sum())
    print(f"      cold start: {n_frozen_init}/{n_freezable} freezable coords "
          f"frozen ({100 * n_frozen_init / max(n_freezable, 1):.1f}%)")

    chunking_active = SKELETON_CHUNK_SIZE is not None
    sample_kwargs = {}
    if chunking_active:
        sample_kwargs["chunk_size"] = SKELETON_CHUNK_SIZE
        sample_kwargs["chunk_dir"] = SKELETON_CHUNK_DIR / dataset_name / f"split_{split_id:02d}" / "grid_sticky_boomerang"

    t0 = time.perf_counter()
    result = sampler.sample(N=N_SKELETON, x0=x_pruned, diagnostics=True, **sample_kwargs)
    elapsed = time.perf_counter() - t0

    if chunking_active:
        samples = resample_boomerang_path_sticky_chunked_torch(
            result["chunk_files"], x_pruned, N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
            manifest_path=result["manifest_path"], dtype=DTYPE, device=DEVICE,
        )
    else:
        samples = resample_boomerang_path_sticky_torch(
            result["positions"], result["velocities"], result["times"], x_pruned,
            N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
        )
    acc, sparsity = evaluate_accuracy(bm, samples, data["X_test"], data["y_test"])

    final_sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, "
          f"final sparsity {final_sparsity:.2f}) "
          f"test_acc={acc:.3f} sample_sparsity={sparsity:.3f}")

    save_run(
        out_path, sampler="grid_sticky_boomerang", samples=samples, x_ref=x_pruned, cfg=cfg,
        elapsed_sec=elapsed, n_events=N_SKELETON,
        bound_violations=result["bound_violations"],
        gradient_evals=result["gradient_evals"], grid_t_max_log=result["grid_t_max_log"],
        test_accuracy=acc, sparsity_frac=sparsity,
        prune_frac=n_frozen_init / max(n_freezable, 1), cold_start_mask=cold_start_mask,
        diagnostics=result.get("diagnostics"),
    )
    print(f"      saved -> {out_path}")


SAMPLER_RUNNERS = {
    "grid_sticky_zigzag": run_grid_sticky_zigzag,
    "grid_sticky_boomerang": run_grid_sticky_boomerang,
}


def run_dataset(split_id: int, data: dict[str, Any], cfg: CNNConfig, out_dir: Path,
                 samplers: list[str], map_path: Optional[Path], architecture: str) -> None:
    print(f"\n--- MNIST-CNN split {split_id:02d} | activation={cfg.activation} "
          f"pool={cfg.pool} | seed={BASE_SEED + split_id} ---")

    sd = out_dir / f"split_{split_id:02d}"
    seed = BASE_SEED + split_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, x_ref, Sigma_inv, cold_start_mask = build_target(data, cfg, map_path=map_path, architecture=architecture)
    print(f"  D = {bm.D}")

    print_preactivation_diagnostic(bm, x_ref, cfg)

    if cold_start_mask is not None:
        print("  using pre-pruned+refit checkpoint's x_ref/cold_start_mask directly (skipping prune_x_ref)")
        x_pruned = x_ref
    else:
        _, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
        x_pruned, cold_start_mask = prune_x_ref(
            bm, x_ref, data["X_sweep"], data["y_sweep"], can_freeze,
        )

    grad_target = torch.func.grad(bm.energy)
    grad_norm_unpruned = grad_target(x_ref).abs().max().item()
    grad_norm_pruned = grad_target(x_pruned).abs().max().item()
    print(f"  ||grad_target||_inf: at unpruned x_ref={grad_norm_unpruned:.4e}  "
          f"at x_pruned={grad_norm_pruned:.4e}  "
          f"(large values, esp. relative to the unpruned one, mean the excess "
          f"gradient is far from zero near the sampler's actual reference point "
          f"-- expect inflated bounce rate / bound_violations, and possibly that "
          f"the prune threshold was too aggressive for sampling even though "
          f"point-estimate accuracy still looked fine)")

    for sampler_name in samplers:
        print(f"  [{sampler_name}]")
        torch.manual_seed(seed)
        np.random.seed(seed)
        SAMPLER_RUNNERS[sampler_name](
            "mnist_cnn", split_id, data, cfg, sd, bm, x_pruned, Sigma_inv, cold_start_mask,
        )


# ===========================================================================
# CLI
# ===========================================================================

def main():
    global N_SKELETON, N_RESAMPLE, N_SAVE, SKELETON_CHUNK_SIZE, SKELETON_CHUNK_DIR

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--n-skeleton", type=int, default=N_SKELETON,
                         help="Overrides the module-level N_SKELETON default.")
    parser.add_argument("--n-resample", type=int, default=N_RESAMPLE,
                         help="Overrides N_RESAMPLE -- INDEPENDENT of --n-skeleton (see "
                              "radiant-finding-church.md sec 5; the original mnist_cnn_grid.py "
                              "coupled these, which reintroduces the memory ceiling chunking "
                              "exists to remove).")
    parser.add_argument("--n-save", type=int, default=N_SAVE,
                         help="Overrides N_SAVE -- INDEPENDENT of --n-resample. Controls saved "
                              "artifact size (thin_to(samples, N_SAVE) in save_run), not "
                              "sampling/resampling correctness.")
    parser.add_argument("--samplers", nargs="+", default=list(SAMPLER_NAMES),
                         choices=list(SAMPLER_NAMES))
    parser.add_argument("--splits", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--map-path", type=Path, default=None)
    parser.add_argument("--architecture", choices=list(ARCHITECTURES), default="cnn",
                         help="Only used when --map-path is not given -- with a checkpoint, "
                              "the architecture is always read from the checkpoint's own "
                              "'architecture' field.")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--skeleton-chunk-size", type=int, default=None,
                         help="Enables chunked skeleton flushing (bounds sampling memory to "
                              "O(chunk_size * D) instead of O(N_SKELETON * D)). None (default) "
                              "disables chunking entirely.")
    parser.add_argument("--skeleton-chunk-dir", type=Path, default=None,
                         help="Base directory for chunk files -- required if --skeleton-chunk-size "
                              "is set. Defaults to <out>/chunks if omitted.")
    args = parser.parse_args()

    N_SKELETON = args.n_skeleton
    N_RESAMPLE = args.n_resample
    N_SAVE = args.n_save
    SKELETON_CHUNK_SIZE = args.skeleton_chunk_size
    SKELETON_CHUNK_DIR = args.skeleton_chunk_dir if args.skeleton_chunk_dir is not None else args.out / "chunks"

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = CNNConfig()

    print(f"\nRunning MNIST-CNN | samplers: {args.samplers} | splits: {args.splits} | "
          f"N_SKELETON={N_SKELETON} | device={DEVICE} dtype={DTYPE}")

    # Sweep data (X_sweep/y_sweep) is only needed if run_dataset ends up
    # calling prune_x_ref -- i.e. the loaded checkpoint has no saved
    # cold_start_mask (an old, unpruned checkpoint, or no --map-path at
    # all). A pre-pruned+refit checkpoint (refit_pruned_lenet_reference.py's
    # output, the expected common case now) already carries its own
    # cold_start_mask, so build_target's skip-prune branch never touches
    # X_sweep/y_sweep -- loading n_sweep=N_SWEEP extra MNIST-test images in
    # that case is pure waste, not just unused. Peek at the checkpoint here
    # (cheap: a state dict, not the training data) purely to decide whether
    # sweep data is worth drawing; build_target reloads it properly below.
    needs_sweep = True
    if args.map_path is not None:
        _ckpt_peek = torch.load(args.map_path, map_location="cpu", weights_only=False)
        needs_sweep = _ckpt_peek.get("cold_start_mask") is None
    n_sweep = N_SWEEP if needs_sweep else 0
    if not needs_sweep:
        print("  checkpoint carries its own cold_start_mask -- skipping sweep-data load (no pruning needed)")

    for split_id in args.splits:
        data = load_mnist_subset(
            args.n_train, args.n_test, args.seed + split_id, args.data_dir, DTYPE, DEVICE,
            n_sweep=n_sweep,
        )
        run_dataset(split_id, data, cfg, args.out, args.samplers, args.map_path, args.architecture)


if __name__ == "__main__":
    main()
