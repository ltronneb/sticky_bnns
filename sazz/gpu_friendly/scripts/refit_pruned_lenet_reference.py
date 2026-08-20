"""
Prune an existing lenet_reference.py MAP checkpoint and refit the surviving
(active) coordinates back toward stationarity, saving the result as a NEW
checkpoint alongside the original -- so fast_mnist_cnn.py (or any other
sampling script) can just load the pruned+refit checkpoint directly via
--map-path, with no pruning/refit logic left in the sampling script itself.

Why this exists: prune_x_ref (previously inside fast_mnist_cnn.py, now
duplicated here since pruning moves to this script) zeroes out coordinates
via a straight torch.where -- no re-optimization. That moves the point off
the mode: on a real run, ||grad_target||_inf jumped from ~136 (unpruned
x_ref) to ~514 (pruned x_pruned). This script repairs that: after pruning,
a few Adam steps run on bm.energy with the gradient masked to zero on
frozen coordinates, so they never move off their pruned value of exactly
0.0 while active coordinates get pulled back toward a stationary point --
same target, same bm.energy loss used everywhere else in this tree, purely
a better reference point, not a distributional change. (Diagonal
preconditioning -- the OTHER fix for badly-conditioned per-coordinate step
sizes -- is a separate, orthogonal change and does not belong in this
script; it is a target-level transform applied at sampling time, not a
reference-point-fitting-time one.)

Usage:
    python -m sazz.gpu_friendly.scripts.refit_pruned_lenet_reference \\
        --map-path results/maps/lenet_reference_N60000_steps10000.pt \\
        --n-train 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.neural_networks import LeNet5, CNN
from sazz.gpu_friendly.models.priors import build_fan_in_prior_precision, build_can_freeze_mask
from sazz.gpu_friendly.scripts.fast_mnist_cnn import load_mnist_subset, eval_accuracy

ARCHITECTURES = {"cnn": CNN, "lenet5": LeNet5}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64

PRUNE_ACC_DROP_TOLERANCE = 0.01
PRUNE_N_THRESHOLDS = 60
N_SWEEP = 2000

REFIT_N_STEPS = 200
REFIT_LR = 1e-3


def prune_x_ref(bm: BayesianModule, x_ref: Tensor, X_sweep: Tensor, y_sweep: Tensor,
                 can_freeze: Tensor, acc_drop_tolerance: float = PRUNE_ACC_DROP_TOLERANCE,
                 n_thresholds: int = PRUNE_N_THRESHOLDS) -> tuple[Tensor, Tensor]:
    """
    Zero out x_ref coordinates below a per-coordinate threshold relative to
    that coordinate's own prior std -- picks the largest threshold (most
    sparsity) whose accuracy on X_sweep/y_sweep stays within
    acc_drop_tolerance of the unpruned baseline. Restricted to can_freeze
    coordinates (biases never prune). Ported verbatim from
    fast_mnist_cnn.py's prune_x_ref -- pruning now happens here, once,
    rather than inside the sampling script.
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


def refit_active_coords(bm: BayesianModule, x_pruned: Tensor, frozen_mask: Tensor,
                         n_steps: int = REFIT_N_STEPS, lr: float = REFIT_LR) -> Tensor:
    """
    A few restricted Adam steps on bm.energy, masking the gradient so
    frozen_mask coordinates never move off x_pruned's existing zero.
    Repairs stationarity in the ACTIVE subspace only. Zeroing the gradient
    on frozen coordinates every step keeps their Adam moment estimates at
    exactly 0 throughout (both are EMAs of a value that's always 0), so
    the update there is exactly 0 every step -- no drift, no
    divide-by-zero.
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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--map-path", type=Path, required=True,
                         help="Existing lenet_reference.py/cnn_reference.py checkpoint to prune+refit.")
    parser.add_argument("--n-train", type=int, default=10_000,
                         help="Training pool size for rebuilding bm -- should match the "
                              "sampling script's --n-train so Sigma_inv's rescale and "
                              "prune/refit decisions are made against the same data bm "
                              "will actually be sampled against. Ignored if --full is given.")
    parser.add_argument("--full", action="store_true",
                         help="Use the full 60k MNIST training set (equivalent to --n-train 60000, "
                              "MNIST's actual training-pool size). Overrides --n-train -- mirrors "
                              "lenet_reference.py's --full flag, so the pruning/refit step here "
                              "runs against the SAME training data the original MAP checkpoint was "
                              "fit on (when that checkpoint was itself produced with --full), "
                              "rather than a subsample that silently changes what Sigma_inv's "
                              "N-rescale and the prune/refit gradient are evaluated against.")
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--n-sweep", type=int, default=N_SWEEP)
    parser.add_argument("--prune-acc-drop-tolerance", type=float, default=PRUNE_ACC_DROP_TOLERANCE)
    parser.add_argument("--refit-n-steps", type=int, default=REFIT_N_STEPS)
    parser.add_argument("--refit-lr", type=float, default=REFIT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--out", type=Path, default=None,
                         help="Output checkpoint path. Default: alongside --map-path, "
                              "named '<original_stem>_pruned_refit.pt'.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_train = 60_000 if args.full else args.n_train

    print(f"Loading reference checkpoint from {args.map_path} ...")
    ckpt = torch.load(args.map_path, map_location="cpu", weights_only=False)
    architecture = ckpt.get("architecture", "cnn")
    activation = ckpt.get("activation", "tanh")
    pool = ckpt.get("pool", "avg")
    prior_std_weight = ckpt.get("prior_std_weight", 2.0)
    prior_std_bias = ckpt.get("prior_std_bias", 2.0)
    fan_in_scaling = ckpt.get("fan_in_scaling", True)

    print(f"  architecture={architecture}  activation={activation}  pool={pool}  "
          f"prior_std_weight={prior_std_weight}  prior_std_bias={prior_std_bias}  "
          f"n_train={'full (60000)' if args.full else n_train}")

    data = load_mnist_subset(n_train, args.n_test, args.seed, args.data_dir,
                              DTYPE, DEVICE, n_sweep=args.n_sweep)

    module_cls = ARCHITECTURES[architecture]
    module = module_cls(activation=activation, pool=pool)
    prec = build_fan_in_prior_precision(
        module, prior_std_weight, prior_std_bias, fan_in_scaling, dtype=DTYPE, device=DEVICE,
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

    grad_target = torch.func.grad(bm.energy)
    grad_norm_unpruned = grad_target(x_ref).abs().max().item()
    print(f"  ||grad_target||_inf at unpruned x_ref = {grad_norm_unpruned:.4e}")

    can_freeze = build_can_freeze_mask(bm.module, device=bm.device)
    if bm.learns_noise:
        can_freeze = torch.cat([can_freeze, torch.zeros(1, dtype=torch.bool, device=bm.device)])

    print("\n[prune]")
    x_pruned, freeze_mask = prune_x_ref(
        bm, x_ref, data["X_sweep"], data["y_sweep"], can_freeze,
        acc_drop_tolerance=args.prune_acc_drop_tolerance,
    )
    grad_norm_pruned = grad_target(x_pruned).abs().max().item()
    print(f"  ||grad_target||_inf at x_pruned (before refit) = {grad_norm_pruned:.4e}")

    print("\n[refit]")
    x_refit = refit_active_coords(bm, x_pruned, freeze_mask,
                                   n_steps=args.refit_n_steps, lr=args.refit_lr)
    grad_norm_refit = grad_target(x_refit).abs().max().item()
    print(f"  ||grad_target||_inf at x_pruned (after {args.refit_n_steps} refit steps) = "
          f"{grad_norm_refit:.4e}  (unpruned baseline was {grad_norm_unpruned:.4e})")

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
