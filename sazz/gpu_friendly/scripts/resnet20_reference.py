"""
MAP + Sigma_inv (empirical Fisher) reference finder for the CIFAR-10
ResNet20 target -- the ResNet-20 analog of cnn_reference.py/
lenet_reference.py, consuming resnet20_pretrain.py's checkpoint as its
starting point.

Deliberate divergences from cnn_reference.py, both load-bearing:
  - MAP init is the PRETRAINED weights (loaded from a resnet20_pretrain.py
    checkpoint), NOT the module's own untrained default init.
    cnn_reference.py's "module's own default init is already close enough
    to the fan-in-scaled prior's typical set" trick relies on CNN/LeNet5
    being tiny; it does not hold for ResNet-20's default init (and doesn't
    even make sense here regardless, since BatchNorm's running stats need
    real data to be populated in the first place -- see
    resnet20_pretrain.py and diagnose_batchnorm_eval.py).
  - The module is switched to eval() IMMEDIATELY after loading the
    pretrain checkpoint's state_dict, before BayesianModule.build is ever
    called, and NEVER switched back for the rest of this script's
    lifetime -- MAP refinement only ever moves beta (weights/biases/BN
    gamma/BN beta), never BatchNorm's running_mean/running_var. This is
    the load-bearing invariant the whole ResNet-20 pipeline depends on;
    see ResNet20's docstring (neural_networks.py) and
    diagnose_batchnorm_eval.py (Phase 0 of the CIFAR-10/ResNet-20 plan)
    for why.
  - Uses priors.py's build_fan_in_prior_precision_resnet (not the original
    build_fan_in_prior_precision) -- BatchNorm gamma is 1-D like a bias,
    but is NOT a bias (it's a multiplicative scale, not additive), so it
    needs its own prior_std_bn_weight-scaled branch instead of either the
    existing weight (fan-in-scaled) or bias branch.
  - run_map/eval_accuracy/full_log_lik/empirical_fisher_diag_scaled are
    imported directly from cnn_reference.py rather than re-implemented --
    all four are architecture-agnostic (operate only on bm/beta), and
    fast_mnist_cnn.py already sets the precedent of importing
    eval_accuracy from cnn_reference.py rather than duplicating it.
  - --n-fisher-batch default is 128 here, not cnn_reference.py's 1024:
    D=272,474 (vs LeNet5's 61,706) means a [n_fisher_batch, D] fp32
    per-example-gradient tensor is ~1.1GB at 1024 (or ~4.3GB at fp64 on a
    CPU dev fallback) -- tune upward via the flag once GPU memory headroom
    is confirmed.

Checkpoint schema additions beyond cnn_reference.py's fields (see
build_target-style consumers): "architecture": "resnet20" is written
EXPLICITLY (unlike cnn_reference.py, which omits it and relies on a
"cnn" default downstream -- a ResNet-20 checkpoint silently defaulting to
"cnn" would be a severe silent bug, not a harmless omission); "pool":
"n/a" (ResNet20 has no pool constructor parameter -- a sentinel, not
absence, so a downstream mismatch-assertion can special-case it rather
than crash on a missing key); "module_state_dict": module.state_dict()
wholesale (params AND buffers) -- x_ref remains the flattened
parameter-only vector (unchanged meaning), module_state_dict is what a
downstream script must load onto a fresh ResNet20() and call .eval() on
before any functional_call, since that's the only way BatchNorm's frozen
running stats survive into the sampling script.

Usage:
    python -m sazz.gpu_friendly.scripts.resnet20_reference \\
        --pretrain-checkpoint results/maps/resnet20_pretrain_N50000_epochs80.pt \\
        --n-steps 2000
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
    build_fan_in_prior_precision_resnet, assert_eval_mode_if_batchnorm,
)
from sazz.gpu_friendly.scripts.cnn_reference import (
    run_map, eval_accuracy, full_log_lik, empirical_fisher_diag_scaled,
)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
DTYPE = torch.float32 if DEVICE in ("cuda", "mps") else torch.float64

ACTIVATION = "relu"
PRIOR_STD_W = 2.0
PRIOR_STD_B = 2.0
PRIOR_STD_BN_WEIGHT = 1.0

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DATA_DIR = Path("datasets")
OUT_DIR = Path("results/maps")


# ===========================================================================
# Data -- per-script-local loader (same convention as resnet20_pretrain.py
# and this tree's MNIST scripts; each script owns its own loader).
# ===========================================================================

def load_cifar10(n: Optional[int], train_frac: float, seed: int, data_dir: Path,
                  dtype: torch.dtype, device: str) -> dict[str, Any]:
    """
    Mirrors cnn_reference.py::load_mnist's structure exactly: n (None = full
    50k CIFAR-10 training pool) split train_frac/(1-train_frac) into
    (X_train, y_train)/(X_eval, y_eval); CIFAR-10's own 10k test set loaded
    separately, in full, untouched by n/train_frac.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_full = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    pool_size = len(train_full) if n is None else n
    pool_idx = rng.choice(len(train_full), size=pool_size, replace=False)
    n_train = int(round(pool_size * train_frac))
    train_idx = pool_idx[:n_train]
    eval_idx = pool_idx[n_train:]
    test_idx = np.arange(len(test_full))

    def stack(ds, idx):
        Xs, ys = zip(*(ds[int(i)] for i in idx))
        X = torch.stack(Xs).to(dtype=dtype, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        return X, y

    X_train, y_train = stack(train_full, train_idx)
    X_eval, y_eval = stack(train_full, eval_idx)
    X_test, y_test = stack(test_full, test_idx)

    print(f"  loaded CIFAR-10: pool={pool_size} (train_frac={train_frac}) -> "
          f"train={tuple(X_train.shape)}, eval={tuple(X_eval.shape)}, "
          f"test={tuple(X_test.shape)} (CIFAR-10's full held-out test set)")
    print(f"  train label counts:  {torch.bincount(y_train, minlength=10).tolist()}")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_eval": X_eval, "y_eval": y_eval,
        "X_test": X_test, "y_test": y_test,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True,
                         help="Path to a resnet20_pretrain.py checkpoint -- provides "
                              "both the MAP init (pretrained weights) and BatchNorm's "
                              "running_mean/running_var (via module_state_dict... see "
                              "resnet20_pretrain.py's 'state_dict' key).")
    parser.add_argument("--n", type=int, default=10_000,
                         help="Pool size drawn from CIFAR-10's 50k training set, "
                              "split --train-frac/(1-train-frac) into train/eval. "
                              "Ignored if --full is given.")
    parser.add_argument("--full", action="store_true",
                         help="Use the full 50k CIFAR-10 training set as the pool "
                              "(equivalent to --n 50000). Overrides --n.")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Minibatch size for the MAP step. Default (None) is "
                              "full-batch gradients.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--n-fisher-batch", type=int, default=128,
                         help="Subsample size for the empirical Fisher diagonal. "
                              "Lower than cnn_reference.py's MNIST-tuned default of "
                              "1024 -- D=272,474 makes a [n_fisher_batch, D] fp32 "
                              "gradient tensor ~1.1GB at 1024 (~4.3GB at fp64 on a "
                              "CPU dev fallback). Raise once GPU memory headroom is "
                              "confirmed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    n = None if args.full else args.n

    print("=" * 72)
    print(f"ResNet20 reference (MAP + vmapped Fisher)  |  pool={'full (50000)' if n is None else n}  "
          f"train_frac={args.train_frac}  n_steps={args.n_steps}  device={DEVICE}  dtype={DTYPE}")
    print("=" * 72)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n[data]")
    data = load_cifar10(n, args.train_frac, args.seed, args.data_dir, DTYPE, DEVICE)

    print(f"\n[pretrain checkpoint] loading {args.pretrain_checkpoint} ...")
    pretrain_ckpt = torch.load(args.pretrain_checkpoint, map_location="cpu", weights_only=False)
    assert pretrain_ckpt.get("architecture") == "resnet20", (
        f"--pretrain-checkpoint architecture mismatch: expected 'resnet20', "
        f"got {pretrain_ckpt.get('architecture')!r}"
    )
    print(f"  pretrain checkpoint: train_acc={pretrain_ckpt.get('train_acc', float('nan')):.3f}  "
          f"test_acc={pretrain_ckpt.get('test_acc', float('nan')):.3f}  "
          f"n_epochs={pretrain_ckpt.get('n_epochs')}")

    print("\n[model]")
    module = ResNet20(activation=ACTIVATION).to(dtype=DTYPE, device=DEVICE)
    module.load_state_dict(pretrain_ckpt["state_dict"])
    module.eval()  # -- BEFORE BayesianModule.build; see module docstring above
    assert_eval_mode_if_batchnorm(module)

    prec = build_fan_in_prior_precision_resnet(
        module, PRIOR_STD_W, PRIOR_STD_B, PRIOR_STD_BN_WEIGHT,
        fan_in_scaling=True, dtype=DTYPE, device=DEVICE,
    )
    bm = BayesianModule.build(
        module, likelihood="categorical",
        X=data["X_train"], y=data["y_train"],
        prior_precision=prec, dtype=DTYPE, device=DEVICE,
    )
    print(f"  D = {bm.D}  activation={ACTIVATION}  "
          f"prior_std_weight={PRIOR_STD_W}  prior_std_bias={PRIOR_STD_B}  "
          f"prior_std_bn_weight={PRIOR_STD_BN_WEIGHT}")

    print("\n[MAP]  (init = pretrained weights, NOT default init)")
    rm_before = module.stem_bn.running_mean.clone()
    x_ref, map_history = run_map(bm, n_steps=args.n_steps, lr=args.lr,
                                  batch_size=args.batch_size, log_every=args.log_every)
    rm_after = module.stem_bn.running_mean.clone()
    bn_frozen = torch.equal(rm_before, rm_after)
    print(f"  BatchNorm running_mean unchanged throughout MAP = {bn_frozen} (expect True)")
    assert bn_frozen, (
        "BatchNorm running_mean changed during the MAP loop -- module was not in "
        "eval() mode throughout, or something else mutated its buffers. This "
        "breaks the sticky sampler's fixed-rate-function invariant downstream."
    )

    with torch.no_grad():
        train_acc = eval_accuracy(bm, x_ref, data["X_train"], data["y_train"])
        eval_acc = eval_accuracy(bm, x_ref, data["X_eval"], data["y_eval"])
        test_acc = eval_accuracy(bm, x_ref, data["X_test"], data["y_test"])
    print(f"  MAP done  train_acc={train_acc:.3f}  eval_acc={eval_acc:.3f}  test_acc={test_acc:.3f}")

    print("\n[Sigma_inv -- vmapped empirical Fisher, N-scaled]")
    t0 = time.perf_counter()
    fisher_diag = empirical_fisher_diag_scaled(bm, x_ref, n_batch=args.n_fisher_batch)
    fisher_elapsed = time.perf_counter() - t0
    Sigma_inv = (bm.prior_precision + fisher_diag).clamp(min=1e-8)
    print(f"  done in {fisher_elapsed:.2f}s  ||Sigma_inv||_inf={Sigma_inv.abs().max():.3f}")
    frac = (fisher_diag > bm.prior_precision).float().mean().item()
    print(f"  Fisher dominates prior in {frac:.1%} of coords  "
          f"median ratio={(fisher_diag / bm.prior_precision).median():.2e}")

    pool_size = 50000 if n is None else n
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"resnet20_reference_N{pool_size}_steps{args.n_steps}"
    out_path = args.out / f"{name}.pt"
    torch.save({
        "x_ref": x_ref.cpu(),
        "Sigma_inv": Sigma_inv.cpu(),
        "sigma_inv_source": "prior_plus_fisher",
        "module_state_dict": module.state_dict(),
        "D": bm.D,
        "architecture": "resnet20",
        "activation": ACTIVATION,
        "pool": "n/a",
        "fan_in_scaling": True,
        "prior_std_weight": PRIOR_STD_W,
        "prior_std_bias": PRIOR_STD_B,
        "prior_std_bn_weight": PRIOR_STD_BN_WEIGHT,
        "train_acc": train_acc,
        "eval_acc": eval_acc,
        "test_acc": test_acc,
        "n_train": data["X_train"].shape[0],
        "n_eval": data["X_eval"].shape[0],
        "n_test": data["X_test"].shape[0],
        "pool_size": pool_size,
        "train_frac": args.train_frac,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "map_history": map_history,
        "pretrain_checkpoint": str(args.pretrain_checkpoint),
    }, out_path)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
