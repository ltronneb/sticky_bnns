"""
fast_cheap_ffn_mnist.py -- the FFN counterpart to fast_cheap_mnist_cnn.py:
same staged/chunked sticky-PDMP driver (see that file's module docstring
for the full staging rationale -- STAGE_SIZE-sized sample() calls chained
via resume_state, each stage's chunk_*.pt files resampled then deleted so
transient disk usage never exceeds one stage's footprint), but targeting a
plain FFN (Izmailov et al.-style MLP: flattened 28x28 input, two 256-unit
hidden layers -- see ffn_mnist_reference.py) instead of CNN/LeNet5.

The conceptual shift from fast_cheap_mnist_cnn.py is ONLY the model:
  - module_cls(activation, pool) -> FFN(layer_sizes, activation) (no pool).
  - MNIST images flattened to [N, 784] before BayesianModule.build (FFN has
    no internal reshape, unlike LeNet5/CNN).
  - No --architecture dispatch (one architecture here), so build_target
    only ever reads a checkpoint's own architecture=="ffn" assertion.
  - print_preactivation_diagnostic (hooks bm.module.conv1/conv2) is CNN-
    specific and dropped -- no conv layers to hook here.
Everything else -- can_freeze (weights freezable, biases and the learned
noise coordinate never freeze, via build_can_freeze_mask + the
bm.learns_noise branch), kappa construction, minibatch grad_target, grid
constants (GAMMA/GRID_T_MAX_INIT_*/GRID_SPACING_*/GRID_N_SEGMENTS/
GRID_ALPHA_*), Sigma_inv handling (rescaled to this run's N, no extra
scale factor -- fast_cheap_mnist_cnn.py applies none either), staged
sampling/resampling/persistence -- is ported verbatim.

x_ref/cold_start_mask are handed straight to the PDMP samplers from the
--map-path checkpoint (expected to already be a pruned+refit checkpoint,
e.g. refit_pruned_ffn_reference.py's output) -- prune_x_ref is only a
fallback for an unpruned checkpoint, same as fast_cheap_mnist_cnn.py.

Usage:
    python -m sazz.gpu_friendly.scripts.fast_cheap_ffn_mnist \\
        --map-path results/maps/ffn_mnist_reference_N60000_steps10000_pruned_refit_tol03_longrefit.pt \\
        --n-skeleton 1_000_000 --stage-size 10_000 \\
        --n-resample 50_000 --n-save 10_000 --grad-batch-size 1024
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

from sazz.gpu_friendly.models.neural_networks import FFN
from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.priors import (
    build_fan_in_prior_precision, build_kappa_from_inclusion, build_can_freeze_mask,
    make_gaussian_prior,
)
from sazz.gpu_friendly.utils.warmup import find_reference_bnn
from sazz.gpu_friendly.utils.resample import (
    resample_zigzag_path_sticky_chunked_torch, resample_boomerang_path_sticky_chunked_torch,
)
from sazz.gpu_friendly.samplers.fast_grid_sticky_zigzag_cheap import FastGridStickyZigZagSampler_Cheap
from sazz.gpu_friendly.samplers.fast_grid_sticky_boomerang_cheap import FastGridStickyBoomerangSampler_Cheap
from sazz.gpu_friendly.scripts.ffn_mnist_reference import eval_accuracy, LAYER_SIZES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64
torch.set_default_dtype(torch.float32)

BASE_SEED = 42

N_TRAIN, N_TEST = 60_000, 1_000
N_SKELETON = 50_000
N_RESAMPLE = 5_000
N_SAVE = 1_000

BURNIN_FRAC = 0.2
N_ACCURACY_DRAWS = 300

PRIOR_STD_W = 2.0
PRIOR_STD_B = 2.0
PRIOR_INCLUSION_WEIGHT = 0.05
ACTIVATION = "relu"

PRUNE_ACC_DROP_TOLERANCE = 0.01
PRUNE_N_THRESHOLDS = 60
N_SWEEP = 2000

GRID_CHUNK_SIZE = 4

# Same grid constants as fast_cheap_mnist_cnn.py -- kept unchanged (per
# instruction: these don't matter too much for this target).
GAMMA = 1e-6
GRID_T_MAX_INIT_ZIGZAG = 0.001
GRID_SPACING_ZIGZAG = 1e-5

REFRESH_RATE = 5e2
GRID_T_MAX_INIT_BOOM = 0.01
GRID_SPACING_BOOM = 1e-3

GRID_N_SEGMENTS = 100
GRID_ALPHA_PLUS = 1.02
GRID_ALPHA_MINUS = 1.04
GRID_ALPHA_VIOLATION = 1.1

SKELETON_CHUNK_DIR: Optional[Path] = None
STAGE_SIZE: Optional[int] = None
GRAD_BATCH_SIZE: Optional[int] = None

DATA_DIR = Path("datasets")
OUT_DIR = Path("results/grid/ffn_mnist")

SAMPLER_NAMES = ("grid_sticky_zigzag", "grid_sticky_boomerang")


@dataclass
class FFNConfig:
    activation: str = ACTIVATION
    prior_std_weight: float = PRIOR_STD_W
    prior_std_bias: float = PRIOR_STD_B
    fan_in_scaling: bool = True
    prior_inclusion_weight: float = PRIOR_INCLUSION_WEIGHT


# ===========================================================================
# Data -- identical to fast_cheap_mnist_cnn.py::load_mnist_subset, plus the
# flatten step ffn_mnist_reference.py's loader also applies.
# ===========================================================================

def load_mnist_subset(n_train: int, n_test: int, seed: int, data_dir: Path,
                       dtype: torch.dtype, device: str, n_sweep: int = 0) -> dict[str, Any]:
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
        X = torch.flatten(X, 1)  # [N, 1, 28, 28] -> [N, 784]
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
# Target builder -- FFN counterpart of fast_cheap_mnist_cnn.py::build_target.
# Single architecture (no ARCHITECTURES dispatch), otherwise identical
# structure: checkpoint mismatch assertions, Sigma_inv rescale to this run's
# N, cold_start_mask passthrough for a pre-pruned+refit checkpoint.
# ===========================================================================

def build_target(data: dict[str, Any], cfg: FFNConfig, dtype=DTYPE, device=DEVICE,
                  map_path: Optional[Path] = None):
    if map_path is not None:
        print(f"\n  Loading reference checkpoint from {map_path} ...")
        ckpt = torch.load(map_path, weights_only=False)
        assert ckpt.get("architecture") == "ffn", (
            f"--map-path architecture mismatch: expected 'ffn', got {ckpt.get('architecture')!r}"
        )
        layer_sizes = ckpt["layer_sizes"]
    else:
        layer_sizes = LAYER_SIZES

    module = FFN(layer_sizes, activation=cfg.activation)
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
        print(f"  loaded  architecture=ffn  layer_sizes={layer_sizes}  "
              f"sigma_inv_source={ckpt.get('sigma_inv_source', 'UNKNOWN')}  "
              f"||x_ref||_inf={x_ref.abs().max():.3f}  "
              f"(train_acc={ckpt.get('train_acc', float('nan')):.3f})"
              + ("  [pre-pruned+refit checkpoint]" if cold_start_mask is not None else ""))

        # Same rescale as fast_cheap_mnist_cnn.py::build_target -- Sigma_inv
        # was computed against the CHECKPOINT's own training-set size, not
        # this run's bm; rescale the Fisher term to this run's actual N.
        N_bm = bm.X.shape[0]
        N_ckpt = ckpt["n_train"]
        ratio = N_bm / N_ckpt
        Sigma_inv = bm.prior_precision + ratio * (Sigma_inv - bm.prior_precision)
        print(f"  Sigma_inv rescale: n_train ckpt={N_ckpt} vs this run's bm.X={N_bm}  "
              f"-> Fisher term scaled by {ratio:.4f}")
    else:
        print(
            "\n  WARNING: no --map-path given, falling back to "
            "utils/warmup.py::find_reference_bnn. Its MAP step starts from "
            "torch.randn(bm.D), far outside this target's fan-in-scaled "
            "prior std -- run ffn_mnist_reference.py once and pass "
            "--map-path for a real reference. Proceeding anyway, but "
            "expect a poor/near-chance MAP."
        )
        x_ref, Sigma_inv = find_reference_bnn(
            bm, n_steps=2000, lr=1e-2, dtype=dtype, device=torch.device(device),
        )
        cold_start_mask = None

    return bm, x_ref, Sigma_inv, cold_start_mask


# ===========================================================================
# Samplers -- identical to fast_cheap_mnist_cnn.py's (architecture-agnostic;
# only touch bm/beta/masks).
# ===========================================================================

def build_minibatch_grad_target(bm: BayesianModule, batch_size: int):
    N_full = bm.X.shape[0]
    scale = N_full / batch_size
    log_prior_fn = make_gaussian_prior(bm.prior_precision)

    idx0 = torch.randint(0, N_full, (batch_size,), device=bm.device)
    cache = {"X": bm.X[idx0], "y": bm.y[idx0]}

    def energy_minibatch(beta: Tensor) -> Tensor:
        log_lik_batch = bm.log_likelihood.single(beta, cache["X"], cache["y"]) * scale
        return -(log_prior_fn(beta) + log_lik_batch)

    grad_target = torch.func.grad(energy_minibatch)

    def resample_fn() -> None:
        idx = torch.randint(0, N_full, (batch_size,), device=bm.device)
        cache["X"] = bm.X[idx]
        cache["y"] = bm.y[idx]

    return grad_target, resample_fn


def _build_sticky_kappa_can_freeze(bm: BayesianModule, cfg: FFNConfig):
    """Weights freezable, biases and the learned noise coordinate never
    freeze (build_can_freeze_mask: p.dim() != 1 -> weights only; the
    bm.learns_noise branch appends False for log_sigma)."""
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


def build_sticky_zigzag_sampler(bm: BayesianModule, cfg: FFNConfig, cold_start_mask: Tensor):
    kappa, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
    if GRAD_BATCH_SIZE is not None:
        grad_target, resample_grad_batch = build_minibatch_grad_target(bm, GRAD_BATCH_SIZE)
    else:
        grad_target, resample_grad_batch = torch.func.grad(bm.energy), None
    return FastGridStickyZigZagSampler_Cheap(
        grad_target=grad_target,
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
        resample_grad_batch=resample_grad_batch,
    )


def build_sticky_boomerang_sampler(bm: BayesianModule, cfg: FFNConfig,
                                    x_ref: Tensor, Sigma_inv: Tensor, cold_start_mask: Tensor):
    kappa, can_freeze = _build_sticky_kappa_can_freeze(bm, cfg)
    if GRAD_BATCH_SIZE is not None:
        grad_target, resample_grad_batch = build_minibatch_grad_target(bm, GRAD_BATCH_SIZE)
    else:
        grad_target, resample_grad_batch = torch.func.grad(bm.energy), None
    sampler = FastGridStickyBoomerangSampler_Cheap(
        grad_target=grad_target,
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
        resample_grad_batch=resample_grad_batch,
    )
    # No extra Sigma_inv scale factor -- matches fast_cheap_mnist_cnn.py,
    # which passes Sigma_inv straight through (post-rescale) with no
    # additional multiplier.
    sampler.preprocess(x_ref=x_ref, Sigma_inv=Sigma_inv)
    return sampler


# ===========================================================================
# Accuracy / diagnostics -- identical to fast_cheap_mnist_cnn.py's
# evaluate_accuracy/prune_x_ref (architecture-agnostic). No
# print_preactivation_diagnostic here -- that hooks bm.module.conv1/conv2,
# CNN-specific with no FFN analog.
# ===========================================================================

@torch.no_grad()
def evaluate_accuracy(bm: BayesianModule, samples: Tensor, X_test: Tensor, y_test: Tensor,
                       n_draws: int = N_ACCURACY_DRAWS) -> tuple[float, float]:
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


# ===========================================================================
# Persistence -- identical to fast_cheap_mnist_cnn.py's (no pool field).
# ===========================================================================

def thin_to(samples: Tensor, n_keep: int) -> Tensor:
    n = samples.shape[0]
    if n <= n_keep:
        return samples
    idx = torch.linspace(0, n - 1, n_keep, device=samples.device).round().long()
    return samples[idx]


def save_run(out_path: Path, *, sampler: str, samples: Tensor, x_ref: Optional[Tensor],
             cfg: FFNConfig, elapsed_sec: float, n_events: int, bound_violations: int,
             gradient_evals: Optional[int] = None, grid_t_max_log: Optional[list[float]] = None,
             test_accuracy: Optional[float] = None, sparsity_frac: Optional[float] = None,
             prune_frac: Optional[float] = None, cold_start_mask: Optional[Tensor] = None,
             diagnostics: Optional[list[dict]] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sampler": sampler,
        "samples": thin_to(samples, N_SAVE).cpu(),
        "x_ref": x_ref.cpu() if x_ref is not None else None,
        "activation": cfg.activation,
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
    if not out_path.exists():
        return False
    try:
        ckpt = torch.load(out_path, weights_only=False)
    except Exception:
        return False
    return ckpt.get("n_events") == n_skeleton


# ===========================================================================
# Per-sampler runners -- identical structure to fast_cheap_mnist_cnn.py's
# (staged, chunked; x_pruned/cold_start_mask threaded through the builder,
# sample()'s x0, and the resample_*_sticky_chunked_torch call).
# ===========================================================================

def run_grid_sticky_zigzag(dataset_name: str, split_id: int, data: dict[str, Any], cfg: FFNConfig,
                            sd: Path, bm: BayesianModule, x_pruned: Tensor, Sigma_inv: Tensor,
                            cold_start_mask: Tensor) -> None:
    out_path = sd / "grid_sticky_zigzag.pt"
    if _resume_skip(out_path, N_SKELETON):
        print(f"      skipping — exists at {out_path}")
        return

    assert STAGE_SIZE is not None, "fast_cheap_ffn_mnist.py requires --stage-size."

    sampler = build_sticky_zigzag_sampler(bm, cfg, cold_start_mask)

    n_freezable = int(sampler.can_freeze.sum())
    n_frozen_init = int(cold_start_mask.sum())
    print(f"      cold start: {n_frozen_init}/{n_freezable} freezable coords "
          f"frozen ({100 * n_frozen_init / max(n_freezable, 1):.1f}%)")

    chunk_dir_base = SKELETON_CHUNK_DIR / dataset_name / f"split_{split_id:02d}" / "grid_sticky_zigzag"

    t0 = time.perf_counter()
    n_stages = math.ceil(N_SKELETON / STAGE_SIZE)
    draws_per_stage = max(N_RESAMPLE // n_stages, 1)
    all_draws = []
    total_grad_evals = 0
    total_bound_violations = 0
    all_tmax_log: list[float] = []
    frozen_mask_final = None
    resume_state = None

    for stage in range(n_stages):
        stage_new_events = min(STAGE_SIZE, N_SKELETON - stage * STAGE_SIZE)
        stage_N = stage_new_events + 1
        stage_dir = chunk_dir_base / f"stage_{stage:04d}"
        print(f"      [stage {stage + 1}/{n_stages}] sampling {stage_new_events} skeleton points "
              f"({'cold start' if stage == 0 else 'resumed'}) -> {stage_dir}")

        result = sampler.sample(
            N=stage_N,
            x0=(x_pruned if stage == 0 else None),
            resume_state=resume_state,
            diagnostics=True,
            chunk_size=STAGE_SIZE, chunk_dir=stage_dir,
        )

        stage_draws = resample_zigzag_path_sticky_chunked_torch(
            result["chunk_files"], N_resample=draws_per_stage,
            burnin_frac=(BURNIN_FRAC if stage == 0 else 0.0),
            manifest_path=result["manifest_path"], dtype=DTYPE, device=DEVICE,
        )
        all_draws.append(stage_draws.cpu())

        total_grad_evals += result["gradient_evals"]
        total_bound_violations += result["bound_violations"]
        all_tmax_log.extend(result["grid_t_max_log"])
        frozen_mask_final = result["frozen_mask_final"]
        resume_state = result["resume_state"]

        n_deleted = 0
        for f in stage_dir.glob("chunk_*.pt"):
            f.unlink()
            n_deleted += 1
        print(f"      [stage {stage + 1}/{n_stages}] freed {n_deleted} chunk_*.pt files from {stage_dir}")

    elapsed = time.perf_counter() - t0
    samples = torch.cat(all_draws, dim=0)

    acc, sparsity = evaluate_accuracy(bm, samples, data["X_test"], data["y_test"])

    final_sparsity = float(frozen_mask_final.float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s across {n_stages} stages "
          f"({total_bound_violations} bound violations, "
          f"final sparsity {final_sparsity:.2f}) "
          f"test_acc={acc:.3f} sample_sparsity={sparsity:.3f}")

    save_run(
        out_path, sampler="grid_sticky_zigzag", samples=samples, x_ref=x_pruned, cfg=cfg,
        elapsed_sec=elapsed, n_events=N_SKELETON,
        bound_violations=total_bound_violations,
        gradient_evals=total_grad_evals, grid_t_max_log=all_tmax_log,
        test_accuracy=acc, sparsity_frac=sparsity,
        prune_frac=n_frozen_init / max(n_freezable, 1), cold_start_mask=cold_start_mask,
        diagnostics=None,
    )
    print(f"      saved -> {out_path}")


def run_grid_sticky_boomerang(dataset_name: str, split_id: int, data: dict[str, Any], cfg: FFNConfig,
                               sd: Path, bm: BayesianModule, x_pruned: Tensor, Sigma_inv: Tensor,
                               cold_start_mask: Tensor) -> None:
    out_path = sd / "grid_sticky_boomerang.pt"
    if _resume_skip(out_path, N_SKELETON):
        print(f"      skipping — exists at {out_path}")
        return

    assert STAGE_SIZE is not None, "fast_cheap_ffn_mnist.py requires --stage-size."

    sampler = build_sticky_boomerang_sampler(bm, cfg, x_pruned, Sigma_inv, cold_start_mask)

    n_freezable = int(sampler.can_freeze.sum())
    n_frozen_init = int(cold_start_mask.sum())
    print(f"      cold start: {n_frozen_init}/{n_freezable} freezable coords "
          f"frozen ({100 * n_frozen_init / max(n_freezable, 1):.1f}%)")

    chunk_dir_base = SKELETON_CHUNK_DIR / dataset_name / f"split_{split_id:02d}" / "grid_sticky_boomerang"

    t0 = time.perf_counter()
    n_stages = math.ceil(N_SKELETON / STAGE_SIZE)
    draws_per_stage = max(N_RESAMPLE // n_stages, 1)
    all_draws = []
    total_grad_evals = 0
    total_bound_violations = 0
    all_tmax_log: list[float] = []
    frozen_mask_final = None
    resume_state = None

    for stage in range(n_stages):
        stage_new_events = min(STAGE_SIZE, N_SKELETON - stage * STAGE_SIZE)
        stage_N = stage_new_events + 1
        stage_dir = chunk_dir_base / f"stage_{stage:04d}"
        print(f"      [stage {stage + 1}/{n_stages}] sampling {stage_new_events} skeleton points "
              f"({'cold start' if stage == 0 else 'resumed'}) -> {stage_dir}")

        result = sampler.sample(
            N=stage_N,
            x0=(x_pruned if stage == 0 else None),
            resume_state=resume_state,
            diagnostics=True,
            chunk_size=STAGE_SIZE, chunk_dir=stage_dir,
        )

        stage_draws = resample_boomerang_path_sticky_chunked_torch(
            result["chunk_files"], x_pruned, N_resample=draws_per_stage,
            burnin_frac=(BURNIN_FRAC if stage == 0 else 0.0),
            manifest_path=result["manifest_path"], dtype=DTYPE, device=DEVICE,
        )
        all_draws.append(stage_draws.cpu())

        total_grad_evals += result["gradient_evals"]
        total_bound_violations += result["bound_violations"]
        all_tmax_log.extend(result["grid_t_max_log"])
        frozen_mask_final = result["frozen_mask_final"]
        resume_state = result["resume_state"]

        n_deleted = 0
        for f in stage_dir.glob("chunk_*.pt"):
            f.unlink()
            n_deleted += 1
        print(f"      [stage {stage + 1}/{n_stages}] freed {n_deleted} chunk_*.pt files from {stage_dir}")

    elapsed = time.perf_counter() - t0
    samples = torch.cat(all_draws, dim=0)

    acc, sparsity = evaluate_accuracy(bm, samples, data["X_test"], data["y_test"])

    final_sparsity = float(frozen_mask_final.float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s across {n_stages} stages "
          f"({total_bound_violations} bound violations, "
          f"final sparsity {final_sparsity:.2f}) "
          f"test_acc={acc:.3f} sample_sparsity={sparsity:.3f}")

    save_run(
        out_path, sampler="grid_sticky_boomerang", samples=samples, x_ref=x_pruned, cfg=cfg,
        elapsed_sec=elapsed, n_events=N_SKELETON,
        bound_violations=total_bound_violations,
        gradient_evals=total_grad_evals, grid_t_max_log=all_tmax_log,
        test_accuracy=acc, sparsity_frac=sparsity,
        prune_frac=n_frozen_init / max(n_freezable, 1), cold_start_mask=cold_start_mask,
        diagnostics=None,
    )
    print(f"      saved -> {out_path}")


SAMPLER_RUNNERS = {
    "grid_sticky_zigzag": run_grid_sticky_zigzag,
    "grid_sticky_boomerang": run_grid_sticky_boomerang,
}


def run_dataset(split_id: int, data: dict[str, Any], cfg: FFNConfig, out_dir: Path,
                 samplers: list[str], map_path: Optional[Path]) -> None:
    print(f"\n--- FFN-MNIST split {split_id:02d} | activation={cfg.activation} "
          f"| seed={BASE_SEED + split_id} ---")

    sd = out_dir / f"split_{split_id:02d}"
    seed = BASE_SEED + split_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, x_ref, Sigma_inv, cold_start_mask = build_target(data, cfg, map_path=map_path)
    print(f"  D = {bm.D}")

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
          f"at x_pruned={grad_norm_pruned:.4e}")

    for sampler_name in samplers:
        print(f"  [{sampler_name}]")
        torch.manual_seed(seed)
        np.random.seed(seed)
        SAMPLER_RUNNERS[sampler_name](
            "ffn_mnist", split_id, data, cfg, sd, bm, x_pruned, Sigma_inv, cold_start_mask,
        )


# ===========================================================================
# CLI
# ===========================================================================

def main():
    global N_SKELETON, N_RESAMPLE, N_SAVE, SKELETON_CHUNK_DIR, STAGE_SIZE, GRAD_BATCH_SIZE

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--n-skeleton", type=int, default=N_SKELETON)
    parser.add_argument("--n-resample", type=int, default=N_RESAMPLE)
    parser.add_argument("--n-save", type=int, default=N_SAVE)
    parser.add_argument("--samplers", nargs="+", default=list(SAMPLER_NAMES),
                         choices=list(SAMPLER_NAMES))
    parser.add_argument("--splits", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--map-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--stage-size", type=int, required=True,
                         help="REQUIRED. See fast_cheap_mnist_cnn.py's identical flag.")
    parser.add_argument("--skeleton-chunk-dir", type=Path, default=None)
    parser.add_argument("--grad-batch-size", type=int, default=None,
                         help="If set, every grad_target(x) call is served from a fresh random "
                              "minibatch of this size instead of the full training set. None "
                              "(default) preserves exact full-batch behavior.")
    args = parser.parse_args()

    N_SKELETON = args.n_skeleton
    N_RESAMPLE = args.n_resample
    N_SAVE = args.n_save
    STAGE_SIZE = args.stage_size
    SKELETON_CHUNK_DIR = args.skeleton_chunk_dir if args.skeleton_chunk_dir is not None else args.out / "chunks"
    GRAD_BATCH_SIZE = args.grad_batch_size

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = FFNConfig()

    n_stages_preview = math.ceil(N_SKELETON / STAGE_SIZE)
    print(f"\nRunning FFN-MNIST (staged) | samplers: {args.samplers} | splits: {args.splits} | "
          f"N_SKELETON={N_SKELETON} | STAGE_SIZE={STAGE_SIZE} ({n_stages_preview} stages) | "
          f"device={DEVICE} dtype={DTYPE}")

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
        run_dataset(split_id, data, cfg, args.out, args.samplers, args.map_path)


if __name__ == "__main__":
    main()
