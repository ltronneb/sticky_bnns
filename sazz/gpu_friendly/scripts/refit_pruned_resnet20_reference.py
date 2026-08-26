"""
Prune an existing resnet20_reference.py checkpoint and refit the surviving
(active) coordinates back toward stationarity, saving the result as a NEW
checkpoint alongside the original -- ResNet-20 analog of
refit_pruned_lenet_reference.py, same recipe, ported faithfully:

  1. prune_x_ref: zero out x_ref coordinates below a per-coordinate
     threshold relative to that coordinate's own prior std, sweeping the
     threshold to find the largest (most sparsity) one whose accuracy on a
     held-out sweep set stays within --prune-acc-drop-tolerance of the
     unpruned baseline. Restricted to can_freeze coordinates (biases AND,
     per priors.py's build_can_freeze_mask_resnet, BatchNorm gamma, never
     freeze/prune).
  2. refit_active_coords: a few Adam steps on bm.energy with the gradient
     masked to zero on frozen coordinates, so pruned coordinates stay
     exactly 0.0 while the surviving coordinates get pulled back toward a
     stationary point of the SAME target (this repairs the ||grad_target||
     jump a raw torch.where prune introduces -- see
     refit_pruned_lenet_reference.py's module docstring for the original
     motivating numbers).
  3. Save a new checkpoint with cold_start_mask set, so fast_cifar_resnet.py
     can load it directly via --map-path with no pruning logic of its own
     (mirrors run_dataset's existing cold_start_mask branch, already wired
     up in fast_cifar_resnet.py).

Two adaptations beyond a mechanical port:
  - build_can_freeze_mask_resnet (priors.py), not the original
    build_can_freeze_mask -- BatchNorm gamma is 1-D like a bias but is NOT
    one; the original's dim()==1 heuristic would otherwise let gamma
    prune/freeze, which is not what this pipeline wants (see priors.py's
    docstring on why BN gamma is deliberately never freezable).
  - The rebuilt module must go through the SAME eval()-before-
    BayesianModule.build ordering as resnet20_reference.py/
    fast_cifar_resnet.py's build_target -- BatchNorm's running_mean/
    running_var are buffers (invisible to BayesianModule.build, restored
    only via module.load_state_dict(ckpt["module_state_dict"])) and must
    never move during pruning/refitting, exactly as they must never move
    during MAP refinement or sampling. See diagnose_batchnorm_eval.py
    (Phase 0) / ResNet20's docstring (neural_networks.py) for why.

Usage:
    python -m sazz.gpu_friendly.scripts.refit_pruned_resnet20_reference \\
        --map-path results/maps/resnet20_reference_N10000_steps2000.pt \\
        --n-train 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchvision import datasets, transforms

from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.neural_networks import ResNet20
from sazz.gpu_friendly.models.priors import (
    build_fan_in_prior_precision_resnet, build_can_freeze_mask_resnet,
    assert_eval_mode_if_batchnorm,
)
from sazz.gpu_friendly.scripts.cnn_reference import eval_accuracy
from sazz.gpu_friendly.scripts.fast_mnist_cnn import build_minibatch_grad_target

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

PRUNE_ACC_DROP_TOLERANCE = 0.01
PRUNE_N_THRESHOLDS = 60
N_SWEEP = 2000

REFIT_N_STEPS = 200
REFIT_LR = 1e-3


# ===========================================================================
# Data -- per-script-local loader with a sweep split, mirroring
# fast_mnist_cnn.py::load_mnist_subset's n_sweep mechanism: sweep_idx is
# carved from the SAME permutation of the test pool as test_idx (disjoint
# by construction), never touched by resnet20_reference.py's own train/eval
# split (which only ever draws from CIFAR-10's TRAIN pool).
# ===========================================================================

def load_cifar10_subset(n_train: int, n_test: int, seed: int, data_dir: Path,
                         dtype: torch.dtype, device: str, n_sweep: int = 0) -> dict[str, Any]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_full = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

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

    print(f"  loaded CIFAR-10 subset: train={tuple(X_train.shape)}, test={tuple(X_test.shape)}")
    out = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

    if n_sweep > 0:
        X_sweep, y_sweep = stack(test_full, sweep_idx)
        print(f"  loaded prune-sweep subset: sweep={tuple(X_sweep.shape)} "
              f"(disjoint from test, drawn from CIFAR-10's TEST pool)")
        out["X_sweep"] = X_sweep
        out["y_sweep"] = y_sweep

    return out


# ===========================================================================
# Prune + refit -- ported verbatim from refit_pruned_lenet_reference.py,
# operating purely on bm/beta (architecture-agnostic).
# ===========================================================================

def prune_x_ref(bm: BayesianModule, x_ref: Tensor, X_sweep: Tensor, y_sweep: Tensor,
                 can_freeze: Tensor, acc_drop_tolerance: float = PRUNE_ACC_DROP_TOLERANCE,
                 n_thresholds: int = PRUNE_N_THRESHOLDS, eval_chunk: int = 256) -> tuple[Tensor, Tensor]:
    """
    eval_chunk (passed through to eval_accuracy's own chunk= param, NOT
    cnn_reference.py's default of 2048): ResNet-20's per-image activation
    memory (19 conv layers x 21 BatchNorm layers of intermediate feature
    maps) is much larger than LeNet5/CNN's, so 2048-image forward passes
    that were fine on MNIST-scale targets can OOM here -- this loop calls
    eval_accuracy ~n_thresholds+1 times (61 by default), and observed on
    a shared/contended A10 GPU to accumulate allocator fragmentation
    across repeated calls even under eval_accuracy's own @torch.no_grad(),
    eventually OOMing partway through the sweep despite nvidia-smi showing
    free memory beforehand. torch.cuda.empty_cache() after each call is a
    second, complementary mitigation -- releases cached-but-unused blocks
    back to the driver between iterations rather than letting them pile up
    across all 61 forward passes.
    """
    prior_std = bm.prior_precision.clamp(min=1e-12).rsqrt()
    baseline_acc = eval_accuracy(bm, x_ref, X_sweep, y_sweep, chunk=eval_chunk)
    if bm.device.type == "cuda":
        torch.cuda.empty_cache()

    sweep_multipliers = np.logspace(-3, 0, n_thresholds)
    best_frac, best_mask, best_acc, best_mult = 0.0, torch.zeros_like(can_freeze), baseline_acc, 0.0
    for m in sweep_multipliers:
        mask = (x_ref.abs() < m * prior_std) & can_freeze
        x_pruned = torch.where(mask, torch.zeros_like(x_ref), x_ref)
        acc = eval_accuracy(bm, x_pruned, X_sweep, y_sweep, chunk=eval_chunk)
        if bm.device.type == "cuda":
            torch.cuda.empty_cache()
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


def refit_active_coords(bm: BayesianModule, x_pruned: Tensor, frozen_mask: Tensor,
                         n_steps: int = REFIT_N_STEPS, lr: float = REFIT_LR,
                         batch_size: Optional[int] = None) -> Tensor:
    """
    batch_size=None (default): full-batch bm.energy(beta) every step, same
    as refit_pruned_lenet_reference.py's original behavior -- fine for
    LeNet5/CNN-scale targets and small MNIST images, but at CIFAR-10 scale
    (--full = 50,000 images) through ResNet-20's 19 conv/21 BatchNorm
    layers, a full-batch backward pass on EVERY one of n_steps (default
    200) iterations is what actually OOMs, not (as first suspected) the
    prune_x_ref sweep loop above. batch_size=B: minibatch gradient via the
    same unbiased-subsampling rescale run_map (cnn_reference.py) already
    uses for MAP fitting -- loss = -log_prior - (N/B)*log_lik_batch, fresh
    random batch drawn every step (plain SGD-style, no vmap involved here,
    so there's no fixed-batch-per-episode constraint like the sticky
    sampler's grad_target has).
    """
    active_mask = ~frozen_mask
    beta = x_pruned.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([beta], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.1)

    N = bm.X.shape[0]
    minibatch = batch_size is not None and batch_size < N
    if minibatch:
        assert not bm.learns_noise, "minibatch rescale double-counts the sigma prior"
        scale = N / batch_size

    for _ in range(n_steps):
        optimizer.zero_grad()
        if minibatch:
            idx = torch.randint(0, N, (batch_size,), device=bm.X.device)
            ll = bm.log_likelihood.single(beta, bm.X[idx], bm.y[idx])
            log_prior = -0.5 * (bm.prior_precision * beta ** 2).sum()
            loss = -log_prior - scale * ll
        else:
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
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--map-path", type=Path, required=True,
                         help="Existing resnet20_reference.py checkpoint to prune+refit.")
    parser.add_argument("--n-train", type=int, default=10_000,
                         help="Training pool size for rebuilding bm -- should match the "
                              "sampling script's --n-train so Sigma_inv's rescale and "
                              "prune/refit decisions are made against the same data bm "
                              "will actually be sampled against. Ignored if --full is given.")
    parser.add_argument("--full", action="store_true",
                         help="Use the full 50k CIFAR-10 training set (equivalent to "
                              "--n-train 50000). Overrides --n-train.")
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--n-sweep", type=int, default=N_SWEEP)
    parser.add_argument("--prune-acc-drop-tolerance", type=float, default=PRUNE_ACC_DROP_TOLERANCE)
    parser.add_argument("--refit-n-steps", type=int, default=REFIT_N_STEPS)
    parser.add_argument("--refit-lr", type=float, default=REFIT_LR)
    parser.add_argument("--refit-batch-size", type=int, default=256,
                         help="Minibatch size for refit_active_coords' Adam loop. Default "
                              "256, NOT None/full-batch -- at CIFAR-10 scale (--full), a "
                              "full-batch backward pass through ResNet-20 on every one of "
                              "--refit-n-steps iterations is what OOMs. Pass --refit-batch-size "
                              "0 (or any value >= --n-train) to force full-batch behavior.")
    parser.add_argument("--diag-batch-size", type=int, default=256,
                         help="Minibatch size for the three ||grad_target||_inf sanity-check "
                              "prints (informational only, nothing downstream depends on "
                              "them) -- avoids a full-batch backward pass just for a diagnostic.")
    parser.add_argument("--eval-chunk", type=int, default=256,
                         help="Forward-pass chunk size for prune_x_ref's threshold-sweep "
                              "eval_accuracy calls (passed through to eval_accuracy's own "
                              "chunk= param). Lower than cnn_reference.py's MNIST-tuned "
                              "default of 2048 -- ResNet-20's per-image activation memory "
                              "is much larger.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--out", type=Path, default=None,
                         help="Output checkpoint path. Default: alongside --map-path, "
                              "named '<original_stem>_pruned_refit.pt'.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_train = 50_000 if args.full else args.n_train

    print(f"Loading reference checkpoint from {args.map_path} ...")
    ckpt = torch.load(args.map_path, map_location="cpu", weights_only=False)
    assert ckpt.get("architecture") == "resnet20", (
        f"--map-path checkpoint architecture mismatch: expected 'resnet20', "
        f"got {ckpt.get('architecture')!r}"
    )
    activation = ckpt.get("activation", "relu")
    prior_std_weight = ckpt.get("prior_std_weight", 2.0)
    prior_std_bias = ckpt.get("prior_std_bias", 2.0)
    prior_std_bn_weight = ckpt.get("prior_std_bn_weight", 1.0)
    fan_in_scaling = ckpt.get("fan_in_scaling", True)

    print(f"  architecture=resnet20  activation={activation}  "
          f"prior_std_weight={prior_std_weight}  prior_std_bias={prior_std_bias}  "
          f"prior_std_bn_weight={prior_std_bn_weight}  "
          f"n_train={'full (50000)' if args.full else n_train}")

    data = load_cifar10_subset(n_train, args.n_test, args.seed, args.data_dir,
                                DTYPE, DEVICE, n_sweep=args.n_sweep)

    # Same load-bearing ordering as resnet20_reference.py/fast_cifar_resnet.py's
    # build_target: load module_state_dict (restores BatchNorm's running
    # stats, invisible to BayesianModule.build otherwise), THEN eval(),
    # THEN BayesianModule.build -- never reordered.
    module = ResNet20(activation=activation)
    module.load_state_dict(ckpt["module_state_dict"])
    module.eval()
    assert_eval_mode_if_batchnorm(module)
    module = module.to(dtype=DTYPE, device=DEVICE)

    prec = build_fan_in_prior_precision_resnet(
        module, prior_std_weight, prior_std_bias, prior_std_bn_weight,
        fan_in_scaling, dtype=DTYPE, device=DEVICE,
    )
    bm = BayesianModule.build(
        module, likelihood="categorical",
        X=data["X_train"], y=data["y_train"],
        prior_precision=prec, dtype=DTYPE, device=DEVICE,
    )
    print(f"  D = {bm.D}")

    assert ckpt["D"] == bm.D, f"D mismatch: ckpt={ckpt['D']} vs rebuilt bm={bm.D}"
    x_ref = ckpt["x_ref"].to(dtype=DTYPE, device=DEVICE)
    Sigma_inv = ckpt["Sigma_inv"].to(dtype=DTYPE, device=DEVICE)

    # ||grad_target||_inf diagnostic below is deliberately evaluated on a
    # MINIBATCH, not bm.energy's full training set -- with --full (50,000
    # images) through ResNet-20, a single full-batch torch.func.grad(bm.energy)
    # backward pass is exactly what was OOMing here (confirmed: crashed on
    # the FIRST such call, before prune_x_ref's sweep loop even starts).
    # These three prints are informational sanity checks only (nothing
    # downstream branches on their value), so a minibatch estimate is a
    # fine substitute -- same build_minibatch_grad_target helper
    # fast_mnist_cnn.py's sticky samplers use, just called for its
    # resample_fn side effect here (redraw before each print) rather than
    # threaded into a sampler.
    diag_batch_size = min(args.diag_batch_size, bm.X.shape[0])
    diag_grad_target, diag_resample = build_minibatch_grad_target(bm, diag_batch_size)

    diag_resample()
    grad_norm_unpruned = diag_grad_target(x_ref).abs().max().item()
    print(f"  ||grad_target||_inf at unpruned x_ref (minibatch={diag_batch_size}) = {grad_norm_unpruned:.4e}")

    can_freeze = build_can_freeze_mask_resnet(bm.module, device=bm.device)
    if bm.learns_noise:
        can_freeze = torch.cat([can_freeze, torch.zeros(1, dtype=torch.bool, device=bm.device)])

    print("\n[prune]")
    rm_before_prune = module.stem_bn.running_mean.clone()
    x_pruned, freeze_mask = prune_x_ref(
        bm, x_ref, data["X_sweep"], data["y_sweep"], can_freeze,
        acc_drop_tolerance=args.prune_acc_drop_tolerance, eval_chunk=args.eval_chunk,
    )
    diag_resample()
    grad_norm_pruned = diag_grad_target(x_pruned).abs().max().item()
    print(f"  ||grad_target||_inf at x_pruned (before refit, minibatch={diag_batch_size}) = {grad_norm_pruned:.4e}")

    print("\n[refit]")
    x_refit = refit_active_coords(bm, x_pruned, freeze_mask,
                                   n_steps=args.refit_n_steps, lr=args.refit_lr,
                                   batch_size=args.refit_batch_size or None)
    diag_resample()
    grad_norm_refit = diag_grad_target(x_refit).abs().max().item()
    print(f"  ||grad_target||_inf at x_pruned (after {args.refit_n_steps} refit steps, "
          f"minibatch={diag_batch_size}) = {grad_norm_refit:.4e}  "
          f"(unpruned baseline was {grad_norm_unpruned:.4e})")

    rm_after_prune = module.stem_bn.running_mean.clone()
    bn_frozen = torch.equal(rm_before_prune, rm_after_prune)
    print(f"  BatchNorm running_mean unchanged throughout prune+refit = {bn_frozen} (expect True)")
    assert bn_frozen, (
        "BatchNorm running_mean changed during prune/refit -- module was not in eval() "
        "mode throughout, or something else mutated its buffers."
    )

    with torch.no_grad():
        refit_acc = eval_accuracy(bm, x_refit, data["X_test"], data["y_test"])
    print(f"  test_acc after prune+refit = {refit_acc:.4f}")

    out_path = args.out or args.map_path.with_name(args.map_path.stem + "_pruned_refit.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_out = dict(ckpt)
    ckpt_out.update({
        "x_ref": x_refit.cpu(),
        "Sigma_inv": Sigma_inv.cpu(),
        "cold_start_mask": freeze_mask.cpu(),
        "module_state_dict": module.state_dict(),
        "source_checkpoint": str(args.map_path),
        "pruned": True,
        "refit": True,
        "prune_acc_drop_tolerance": args.prune_acc_drop_tolerance,
        "refit_n_steps": args.refit_n_steps,
        "refit_lr": args.refit_lr,
        "n_train_prune_refit": data["X_train"].shape[0],
        "test_acc_pruned_refit": refit_acc,
        "grad_norm_inf_unpruned": grad_norm_unpruned,
        "grad_norm_inf_pruned": grad_norm_pruned,
        "grad_norm_inf_pruned_refit": grad_norm_refit,
    })
    torch.save(ckpt_out, out_path)
    print(f"\nSaved pruned+refit checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
