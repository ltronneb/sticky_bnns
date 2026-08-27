"""
CIFAR-10 ResNet-20 sticky-PDMP sampling driver -- the ResNet-20 analog of
fast_mnist_cnn.py. A NEW script, not an extension of fast_mnist_cnn.py's
ARCHITECTURES dict in place: fast_mnist_cnn.py's module-level grid/rate
constants (GAMMA, GRID_T_MAX_INIT_*, GRID_SPACING_*, ...) are MNIST/CNN-
tuned and near-certainly wrong scale for D=272,474 + BatchNorm's very
different curvature profile, and there is no CLI-level guard against
accidentally reusing them for a different-scale target. print_
preactivation_diagnostic also hardcodes bm.module.conv1/conv2 attribute
hooks that do not exist on ResNet20 as structured (see this file's own
version below, hooking stem_conv/stage3's last block instead).

REQUIRES a --map-path checkpoint from resnet20_reference.py (no torch.randn
fallback path, unlike fast_mnist_cnn.py's find_reference_bnn branch -- that
branch is already effectively dead for CNN-scale targets in the original
script, per its own printed warning, and ResNet-20 has no sane analog of a
"tiny default init happens to work" fallback).

grad_target/BatchNorm invariant: build_target below loads the checkpoint's
module_state_dict onto a fresh ResNet20 and calls .eval() BEFORE
BayesianModule.build -- this is the single most important ordering in this
file, directly enforcing the invariant established in diagnose_batchnorm_
eval.py (Phase 0) and documented on ResNet20 itself (neural_networks.py):
grad_target must be a fixed function of x for an entire sticky sampler
_grid_bound episode, which requires BatchNorm's running_mean/running_var to
never change after this point.

Usage:
    python -m sazz.gpu_friendly.scripts.fast_cifar_resnet \\
        --map-path results/maps/resnet20_reference_N50000_steps2000.pt \\
        --n-skeleton 5000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchvision import datasets, transforms

from sazz.gpu_friendly.models.neural_networks import ResNet20
from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.priors import (
    build_fan_in_prior_precision_resnet, build_kappa_from_inclusion_resnet,
    build_can_freeze_mask_resnet, assert_eval_mode_if_batchnorm,
)
from sazz.gpu_friendly.utils.resample import (
    resample_zigzag_path_sticky_torch, resample_boomerang_path_sticky_torch,
    resample_zigzag_path_sticky_chunked_torch, resample_boomerang_path_sticky_chunked_torch,
)
from sazz.gpu_friendly.samplers.fast_grid_sticky_zigzag import FastGridStickyZigZagSampler
from sazz.gpu_friendly.samplers.fast_grid_sticky_boomerang import FastGridStickyBoomerangSampler
from sazz.gpu_friendly.scripts.fast_mnist_cnn import (
    CNNConfig, build_minibatch_grad_target, save_run, thin_to, _resume_skip,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64
torch.set_default_dtype(torch.float32)

BASE_SEED = 42

N_TRAIN, N_TEST = 50_000, 1_000
N_SKELETON = 5_000
N_RESAMPLE = 2_000
N_SAVE = 1_000

BURNIN_FRAC = 0.2
N_ACCURACY_DRAWS = 300

PRIOR_STD_W = 2.0
PRIOR_STD_B = 2.0
PRIOR_STD_BN_WEIGHT = 1.0
PRIOR_INCLUSION_WEIGHT = 0.05
ACTIVATION = "relu"

# --- Grid/rate constants -- STARTING POINTS ONLY, not tuned values. Per
# the plan, these must be empirically retuned against a real reference
# checkpoint (diagnose_zigzag_rate.py-style rate inspection) before a real
# (non-smoke) run -- D=272,474 (vs LeNet5's 61,706) and BatchNorm's very
# different curvature profile mean fast_mnist_cnn.py's MNIST-tuned values
# are not expected to transfer. These starting points are deliberately
# conservative (smaller grid_t_max/grid_spacing than the MNIST script) as a
# first-pass smoke-test default, not a tuned recommendation. ---
GRID_CHUNK_SIZE = 4

GAMMA = 1e-6
GRID_T_MAX_INIT_ZIGZAG = 1e-4
GRID_SPACING_ZIGZAG = 1e-6

REFRESH_RATE = 5e2
GRID_T_MAX_INIT_BOOM = 1e-3
GRID_SPACING_BOOM = 1e-4

GRID_N_SEGMENTS = 100
GRID_ALPHA_PLUS = 1.02
GRID_ALPHA_MINUS = 1.04
GRID_ALPHA_VIOLATION = 1.1

SKELETON_CHUNK_SIZE: Optional[int] = None
SKELETON_CHUNK_DIR: Optional[Path] = None

GRAD_BATCH_SIZE: Optional[int] = None

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DATA_DIR = Path("datasets")
OUT_DIR = Path("results/grid/cifar_resnet")

SAMPLER_NAMES = ("grid_sticky_zigzag", "grid_sticky_boomerang")


# ===========================================================================
# Data -- per-script-local loader (own convention, matching resnet20_
# pretrain.py/resnet20_reference.py rather than fast_mnist_cnn.py's MNIST
# loader, since the data itself differs; same overall structural pattern).
# ===========================================================================

def load_cifar10_subset(n_train: int, n_test: int, seed: int, data_dir: Path,
                         dtype: torch.dtype, device: str) -> dict[str, Any]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_full = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(train_full), size=n_train, replace=False)
    test_idx = rng.choice(len(test_full), size=n_test, replace=False)

    def stack(ds, idx):
        Xs, ys = zip(*(ds[int(i)] for i in idx))
        X = torch.stack(Xs).to(dtype=dtype, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        return X, y

    X_train, y_train = stack(train_full, train_idx)
    X_test, y_test = stack(test_full, test_idx)

    print(f"  loaded CIFAR-10 subset: train={tuple(X_train.shape)}, test={tuple(X_test.shape)}")
    print(f"  train label counts:  {torch.bincount(y_train, minlength=10).tolist()}")
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


# ===========================================================================
# Target builder
# ===========================================================================

def build_target(data: dict[str, Any], cfg: CNNConfig, dtype=DTYPE, device=DEVICE,
                  map_path: Path = None):
    """
    Unlike fast_mnist_cnn.py::build_target, this REQUIRES map_path (no
    torch.randn/find_reference_bnn fallback -- see module docstring) and
    hard-asserts ckpt["architecture"] == "resnet20" rather than defaulting
    an absent/mismatched value to some other architecture, since there is
    no sane fallback architecture for this script to default to.

    Load-bearing ordering, do not reorder: (1) load checkpoint, (2) build
    ResNet20, (3) module.load_state_dict(ckpt["module_state_dict"]) --
    restores BatchNorm's running_mean/running_var (buffers, invisible to
    BayesianModule.build) alongside params (whose values are immediately
    superseded in meaning by beta anyway, per model.py's own docstring:
    "module's CURRENT parameter values are only used for names/shapes"),
    (4) module.eval() -- BEFORE BayesianModule.build, not after.
    """
    assert map_path is not None, (
        "fast_cifar_resnet.py requires --map-path (a resnet20_reference.py "
        "checkpoint) -- there is no torch.randn/find_reference_bnn fallback "
        "for a target this size; see module docstring."
    )
    print(f"\n  Loading reference checkpoint from {map_path} ...")
    ckpt = torch.load(map_path, weights_only=False)
    assert ckpt.get("architecture") == "resnet20", (
        f"--map-path checkpoint architecture mismatch: expected 'resnet20', "
        f"got {ckpt.get('architecture')!r}"
    )

    module = ResNet20(activation=cfg.activation)
    module.load_state_dict(ckpt["module_state_dict"])
    module.eval()  # -- BEFORE BayesianModule.build; see docstring above
    assert_eval_mode_if_batchnorm(module)
    module = module.to(dtype=dtype, device=device)

    prec = build_fan_in_prior_precision_resnet(
        module, cfg.prior_std_weight, cfg.prior_std_bias, PRIOR_STD_BN_WEIGHT,
        cfg.fan_in_scaling, dtype=dtype, device=device,
    )
    bm = BayesianModule.build(
        module, likelihood="categorical",
        X=data["X_train"], y=data["y_train"],
        prior_precision=prec, dtype=dtype, device=device,
    )

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
    # "pool" deliberately NOT compared -- ResNet20 has no pool constructor
    # parameter (Phase 2); resnet20_reference.py writes the sentinel
    # "pool": "n/a" specifically so this comparison is skipped here, not
    # so it silently passes/fails on an inapplicable field.
    assert not mismatches, (
        "Checkpoint is not a valid reference for this target -- mismatched fields:\n  "
        + "\n  ".join(mismatches)
    )
    x_ref = ckpt["x_ref"].to(dtype=dtype, device=device)
    Sigma_inv = ckpt["Sigma_inv"].to(dtype=dtype, device=device)
    cold_start_mask = ckpt.get("cold_start_mask")
    if cold_start_mask is not None:
        cold_start_mask = cold_start_mask.to(dtype=torch.bool, device=device)
    print(f"  loaded  architecture=resnet20  sigma_inv_source={ckpt.get('sigma_inv_source', 'UNKNOWN')}  "
          f"||x_ref||_inf={x_ref.abs().max():.3f}  "
          f"(train_acc={ckpt.get('train_acc', float('nan')):.3f})"
          + ("  [pre-pruned+refit checkpoint]" if cold_start_mask is not None else ""))

    # Same Sigma_inv rescale convention as fast_mnist_cnn.py::build_target
    # -- Sigma_inv was computed at checkpoint time against N_ckpt (the
    # CHECKPOINT's own training-set size), rescaled here to N_bm (this
    # run's own bm.X.shape[0]) so the two stay definitionally consistent.
    N_bm = bm.X.shape[0]
    N_ckpt = ckpt["n_train"]
    ratio = N_bm / N_ckpt
    Sigma_inv = bm.prior_precision + ratio * (Sigma_inv - bm.prior_precision)
    print(f"  Sigma_inv rescale: n_train ckpt={N_ckpt} vs this run's bm.X={N_bm}  "
          f"-> Fisher term scaled by {ratio:.4f}")

    return bm, x_ref, Sigma_inv, cold_start_mask


# ===========================================================================
# Samplers
# ===========================================================================

def _build_sticky_kappa_can_freeze_resnet(bm: BayesianModule, cfg: CNNConfig):
    """ResNet-specific analog of fast_mnist_cnn.py::_build_sticky_kappa_can_freeze
    -- uses the BatchNorm-aware _resnet builders (priors.py) instead of the
    originals, since ResNet20 has BatchNorm gamma/beta parameters the
    originals' dim()==1 heuristic would misclassify as ordinary biases."""
    kappa_net = build_kappa_from_inclusion_resnet(
        bm.module, cfg.prior_std_weight, cfg.prior_inclusion_weight,
        cfg.fan_in_scaling, dtype=DTYPE, device=bm.device,
    )
    can_freeze_net = build_can_freeze_mask_resnet(bm.module, device=bm.device)

    if bm.learns_noise:
        kappa = torch.cat([kappa_net, torch.zeros(1, dtype=DTYPE, device=bm.device)])
        can_freeze = torch.cat([can_freeze_net, torch.zeros(1, dtype=torch.bool, device=bm.device)])
    else:
        kappa = kappa_net
        can_freeze = can_freeze_net
    return kappa, can_freeze


def build_sticky_zigzag_sampler(bm: BayesianModule, cfg: CNNConfig, cold_start_mask: Tensor):
    kappa, can_freeze = _build_sticky_kappa_can_freeze_resnet(bm, cfg)
    if GRAD_BATCH_SIZE is not None:
        grad_target, resample_grad_batch = build_minibatch_grad_target(bm, GRAD_BATCH_SIZE)
    else:
        grad_target, resample_grad_batch = torch.func.grad(bm.energy), None
    return FastGridStickyZigZagSampler(
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


def build_sticky_boomerang_sampler(bm: BayesianModule, cfg: CNNConfig,
                                    x_ref: Tensor, Sigma_inv: Tensor, cold_start_mask: Tensor):
    kappa, can_freeze = _build_sticky_kappa_can_freeze_resnet(bm, cfg)
    if GRAD_BATCH_SIZE is not None:
        grad_target, resample_grad_batch = build_minibatch_grad_target(bm, GRAD_BATCH_SIZE)
    else:
        grad_target, resample_grad_batch = torch.func.grad(bm.energy), None
    sampler = FastGridStickyBoomerangSampler(
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
    sampler.preprocess(x_ref=x_ref, Sigma_inv=Sigma_inv)
    return sampler


# ===========================================================================
# Accuracy / diagnostics
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


def print_preactivation_diagnostic(bm: BayesianModule, x_ref: Tensor, cfg: CNNConfig, n_prior_draws: int = 5):
    """
    ResNet-specific analog of fast_mnist_cnn.py's version -- hooks
    stem_conv (the network's first conv, always present) and stage3's last
    block's conv2 (a late-stage conv, deepest point before the head)
    instead of the MNIST script's hardcoded conv1/conv2 attribute names,
    which do not exist on ResNet20's structure (Phase 2, neural_networks.py).
    """
    hook_outputs = []

    def hook(module, inp, out):
        hook_outputs.append(inp[0].detach())

    last_block = bm.module.stage3[-1]
    handles = [bm.module.stem_conv.register_forward_hook(hook),
               last_block.conv2.register_forward_hook(hook)]

    def frac_large(beta: Tensor) -> float:
        hook_outputs.clear()
        with torch.no_grad():
            torch.func.functional_call(bm.module, bm.param_dict_fn(beta), (bm.X[:32],))
        all_z = torch.cat([h.flatten() for h in hook_outputs])
        return (all_z.abs() > 2.0).float().mean().item()

    for h in handles:
        h.remove()

    handles = [bm.module.stem_conv.register_forward_hook(hook),
               last_block.conv2.register_forward_hook(hook)]
    frac_at_xref = frac_large(x_ref)

    fracs_prior = []
    D_network = bm.D - (1 if bm.learns_noise else 0)
    for _ in range(n_prior_draws):
        beta_prior = torch.randn(bm.D, dtype=DTYPE, device=DEVICE) / bm.prior_precision.clamp(min=1e-8).sqrt()
        fracs_prior.append(frac_large(beta_prior))

    for h in handles:
        h.remove()

    print(f"  pre-activation |z|>2 fraction (stem_conv + stage3[-1].conv2 inputs): "
          f"at x_ref={frac_at_xref:.3f}, prior draws mean={np.mean(fracs_prior):.3f} "
          f"(prior_std_weight={cfg.prior_std_weight}) -- "
          f"large values mean the prior scale is putting activations into a "
          f"saturated/unusual regime, independent of target curvature")


# ===========================================================================
# Per-sampler runners
# ===========================================================================

def run_grid_sticky_zigzag(dataset_name: str, split_id: int, data: dict[str, Any], cfg: CNNConfig,
                            sd: Path, bm: BayesianModule, x_ref: Tensor, Sigma_inv: Tensor,
                            cold_start_mask: Tensor) -> None:
    out_path = sd / "grid_sticky_zigzag.pt"
    if _resume_skip(out_path, N_SKELETON):
        print(f"      skipping — exists at {out_path}")
        return

    sampler = build_sticky_zigzag_sampler(bm, cfg, cold_start_mask)

    n_freezable = int(sampler.can_freeze.sum())
    n_frozen_init = int(cold_start_mask.sum()) if cold_start_mask is not None else 0
    print(f"      cold start: {n_frozen_init}/{n_freezable} freezable coords "
          f"frozen ({100 * n_frozen_init / max(n_freezable, 1):.1f}%)")

    chunking_active = SKELETON_CHUNK_SIZE is not None
    sample_kwargs = {}
    if chunking_active:
        sample_kwargs["chunk_size"] = SKELETON_CHUNK_SIZE
        sample_kwargs["chunk_dir"] = SKELETON_CHUNK_DIR / dataset_name / f"split_{split_id:02d}" / "grid_sticky_zigzag"

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
    acc, sparsity = evaluate_accuracy(bm, samples, data["X_test"], data["y_test"])

    final_sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, "
          f"final sparsity {final_sparsity:.2f}) "
          f"test_acc={acc:.3f} sample_sparsity={sparsity:.3f}")

    save_run(
        out_path, sampler="grid_sticky_zigzag", samples=samples, x_ref=x_ref, cfg=cfg,
        elapsed_sec=elapsed, n_events=N_SKELETON,
        bound_violations=result["bound_violations"],
        gradient_evals=result["gradient_evals"], grid_t_max_log=result["grid_t_max_log"],
        test_accuracy=acc, sparsity_frac=sparsity,
        prune_frac=n_frozen_init / max(n_freezable, 1), cold_start_mask=cold_start_mask,
        diagnostics=result.get("diagnostics"),
    )
    print(f"      saved -> {out_path}")


def run_grid_sticky_boomerang(dataset_name: str, split_id: int, data: dict[str, Any], cfg: CNNConfig,
                               sd: Path, bm: BayesianModule, x_ref: Tensor, Sigma_inv: Tensor,
                               cold_start_mask: Tensor) -> None:
    out_path = sd / "grid_sticky_boomerang.pt"
    if _resume_skip(out_path, N_SKELETON):
        print(f"      skipping — exists at {out_path}")
        return

    sampler = build_sticky_boomerang_sampler(bm, cfg, x_ref, Sigma_inv, cold_start_mask)

    n_freezable = int(sampler.can_freeze.sum())
    n_frozen_init = int(cold_start_mask.sum()) if cold_start_mask is not None else 0
    print(f"      cold start: {n_frozen_init}/{n_freezable} freezable coords "
          f"frozen ({100 * n_frozen_init / max(n_freezable, 1):.1f}%)")

    chunking_active = SKELETON_CHUNK_SIZE is not None
    sample_kwargs = {}
    if chunking_active:
        sample_kwargs["chunk_size"] = SKELETON_CHUNK_SIZE
        sample_kwargs["chunk_dir"] = SKELETON_CHUNK_DIR / dataset_name / f"split_{split_id:02d}" / "grid_sticky_boomerang"

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
    acc, sparsity = evaluate_accuracy(bm, samples, data["X_test"], data["y_test"])

    final_sparsity = float(result["frozen_mask_final"].float().mean())
    print(f"      sampled {N_SKELETON} skeleton events in {elapsed:.1f}s "
          f"({result['bound_violations']} bound violations, "
          f"final sparsity {final_sparsity:.2f}) "
          f"test_acc={acc:.3f} sample_sparsity={sparsity:.3f}")

    save_run(
        out_path, sampler="grid_sticky_boomerang", samples=samples, x_ref=x_ref, cfg=cfg,
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
                 samplers: list[str], map_path: Path) -> None:
    print(f"\n--- CIFAR10-ResNet20 split {split_id:02d} | activation={cfg.activation} "
          f"seed={BASE_SEED + split_id} ---")

    sd = out_dir / f"split_{split_id:02d}"
    seed = BASE_SEED + split_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, x_ref, Sigma_inv, cold_start_mask = build_target(data, cfg, map_path=map_path)
    print(f"  D = {bm.D}")

    print_preactivation_diagnostic(bm, x_ref, cfg)

    if cold_start_mask is None:
        # No pruning support in this script's first version (Phase 5,
        # deferred per the plan) -- cold_start_mask must come from the
        # checkpoint (a future refit_pruned_resnet20_reference.py output)
        # or every coordinate starts unfrozen.
        cold_start_mask = torch.zeros(bm.D, dtype=torch.bool, device=bm.device)
        print("  no cold_start_mask in checkpoint -- starting with all coordinates unfrozen "
              "(pruning/refit for ResNet-20 is deferred, see plan Phase 5)")
    else:
        print("  using checkpoint's cold_start_mask directly")

    # Deliberately a MINIBATCH gradient here, not torch.func.grad(bm.energy)
    # directly -- with --n-train at CIFAR-10 scale (up to 50,000 images), a
    # full-batch backward pass through ResNet-20's 19 conv/21 BatchNorm
    # layers is exactly what OOMs (same root cause found and fixed in
    # refit_pruned_resnet20_reference.py and fast_cheap_cifar_resnet.py's
    # identical diagnostic line). This print is informational only --
    # nothing downstream branches on its value -- so a minibatch estimate
    # is a fine substitute for a full-dataset one.
    diag_batch_size = min(1024, bm.X.shape[0])
    diag_grad_target, _ = build_minibatch_grad_target(bm, diag_batch_size)
    grad_norm = diag_grad_target(x_ref).abs().max().item()
    print(f"  ||grad_target||_inf: at x_ref (minibatch={diag_batch_size}) = {grad_norm:.4e}  "
          f"(large values mean the excess gradient is far from zero near the "
          f"sampler's actual reference point -- expect inflated bounce rate / "
          f"bound_violations)")

    for sampler_name in samplers:
        print(f"  [{sampler_name}]")
        torch.manual_seed(seed)
        np.random.seed(seed)
        SAMPLER_RUNNERS[sampler_name](
            "cifar_resnet20", split_id, data, cfg, sd, bm, x_ref, Sigma_inv, cold_start_mask,
        )


# ===========================================================================
# CLI
# ===========================================================================

def main():
    global N_SKELETON, N_RESAMPLE, N_SAVE, SKELETON_CHUNK_SIZE, SKELETON_CHUNK_DIR, GRAD_BATCH_SIZE

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
    parser.add_argument("--map-path", type=Path, required=True,
                         help="A resnet20_reference.py checkpoint (required -- see module docstring).")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--skeleton-chunk-size", type=int, default=None)
    parser.add_argument("--skeleton-chunk-dir", type=Path, default=None)
    parser.add_argument("--grad-batch-size", type=int, default=None,
                         help="If set, every grad_target(x) call is served from a fresh random "
                              "minibatch of this size instead of the full training set. None "
                              "(default) preserves exact full-batch behavior. Recommended to "
                              "include from the start (not defer) -- ResNet-20's full-batch "
                              "gradient cost is substantially higher than LeNet5's per call.")
    args = parser.parse_args()

    N_SKELETON = args.n_skeleton
    N_RESAMPLE = args.n_resample
    N_SAVE = args.n_save
    SKELETON_CHUNK_SIZE = args.skeleton_chunk_size
    SKELETON_CHUNK_DIR = args.skeleton_chunk_dir if args.skeleton_chunk_dir is not None else args.out / "chunks"
    GRAD_BATCH_SIZE = args.grad_batch_size

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = CNNConfig(
        activation=ACTIVATION, pool="n/a", prior_std_weight=PRIOR_STD_W,
        prior_std_bias=PRIOR_STD_B, fan_in_scaling=True,
        prior_inclusion_weight=PRIOR_INCLUSION_WEIGHT,
    )

    print(f"\nRunning CIFAR10-ResNet20 | samplers: {args.samplers} | splits: {args.splits} | "
          f"N_SKELETON={N_SKELETON} | device={DEVICE} dtype={DTYPE}")

    for split_id in args.splits:
        data = load_cifar10_subset(
            args.n_train, args.n_test, args.seed + split_id, args.data_dir, DTYPE, DEVICE,
        )
        run_dataset(split_id, data, cfg, args.out, args.samplers, args.map_path)


if __name__ == "__main__":
    main()
