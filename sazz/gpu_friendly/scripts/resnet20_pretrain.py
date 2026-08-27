"""
Ordinary (non-Bayesian) SGD pretrain pass for ResNet20 on CIFAR-10. Sole
purpose: populate BatchNorm's running_mean/running_var with real statistics
(PyTorch's default BatchNorm2d init gives running_mean=0/running_var=1 --
i.e. NO real statistics at all), and land the network at a reasonable point
in weight space -- a far better MAP initializer for resnet20_reference.py
than ResNet20's raw default init would be (unlike CNN/LeNet5, whose tiny
default inits happen to be close enough to a fan-in-scaled prior's typical
set for a MAP step alone to fix, per cnn_reference.py's own docstring).

This is the ONLY place in the whole ResNet-20 pipeline that calls
module.train() -- see diagnose_batchnorm_eval.py (Phase 0) and ResNet20's
docstring (neural_networks.py) for why every downstream stage (MAP
refinement, Fisher-diagonal estimation, PDMP sampling) requires the module
to be eval() and PERMANENTLY frozen there instead.

Produces a PRETRAIN checkpoint (raw module.state_dict(), params+buffers
together) -- NOT the final reference checkpoint resnet20_reference.py
build_target-style scripts consume. Keep the two schemas distinct:
resnet20_pretrain_*.pt (this script) vs resnet20_reference_*.pt
(resnet20_reference.py).

Usage:
    python -m sazz.gpu_friendly.scripts.resnet20_pretrain --n-epochs 80
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from sazz.gpu_friendly.models.neural_networks import ResNet20

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
DTYPE = torch.float32 if DEVICE in ("cuda", "mps") else torch.float64

ACTIVATION = "relu"

# Standard CIFAR-10 per-channel normalization constants -- this repo has no
# prior CIFAR-10 convention to match (grepped, confirmed: none exists), so
# these are the commonly used reference values, analogous in role to
# fast_mnist_cnn.py's transforms.Normalize((0.1307,), (0.3081,)).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DATA_DIR = Path("datasets")
OUT_DIR = Path("results/maps")


# ===========================================================================
# Data -- per-script-local loader, matching this tree's existing convention
# (cnn_reference.py, lenet_reference.py, fast_mnist_cnn.py each own their own
# MNIST loader rather than sharing one; do the same here for CIFAR-10).
#
# Standard CIFAR-10 augmentation (random 32x32 crop from a 4px-padded image,
# random horizontal flip) WAS deliberately deferred in this file's first
# version -- observed consequence: train_acc hit 1.000 with loss flatlined
# by ~epoch 45/80 while test_acc plateaued at 83-84%, the classic
# augmentation-starved memorization signature (standard ResNet-20/CIFAR-10
# recipes reach ~91-92% test_acc, precisely because augmentation prevents
# this). Implemented below as an on-GPU/on-tensor batch-time transform
# (pad+random-crop+random-flip applied to X_train batches inside the
# training loop -- see augment_batch), NOT by switching to PIL-based
# per-sample torchvision.transforms, so the loader below still returns one
# fixed, pre-materialized X_train tensor (same convention as every other
# script in this tree) -- augmentation happens fresh each epoch inside
# run_pretrain's loop instead, so it's genuine per-epoch randomness, not a
# single baked-in crop reused forever. X_test is NEVER augmented, in
# load_cifar10_subset or anywhere else -- only run_pretrain's train-batch
# path calls augment_batch.
# ===========================================================================

def load_cifar10_subset(n_train: int, n_test: int, seed: int, data_dir: Path,
                         dtype: torch.dtype, device: str) -> dict[str, Any]:
    """
    Converts X_train/y_train to (dtype, device) here, matching
    fast_mnist_cnn.py::load_mnist_subset's convention (BayesianModule.build's
    likelihood closures capture raw X/y before bm.X/bm.y's own .to(dtype,
    device) call -- a separate object -- so a caller must pre-convert; not
    load-bearing for THIS script specifically since it never builds a
    BayesianModule, but kept consistent so a downstream script reusing this
    loader's output doesn't have to guess).
    """
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
# Augmentation -- standard CIFAR-10 recipe (pad 4px reflect, random 32x32
# crop, random horizontal flip), applied per-batch inside run_pretrain's
# loop so every epoch sees a fresh random crop/flip of each image, not one
# fixed transform baked in at load time. Pure tensor ops (F.pad + indexing
# + torch.where), runs on whatever device X_batch is already on -- no PIL
# round-trip, no extra host<->device copies per batch.
# ===========================================================================

def augment_batch(X_batch: Tensor, pad: int = 4) -> Tensor:
    """X_batch: [B, C, H, W], already normalized. Returns a same-shape
    tensor with an independent random crop + random horizontal flip per
    sample in the batch."""
    import torch.nn.functional as F

    B, C, H, W = X_batch.shape
    padded = F.pad(X_batch, [pad, pad, pad, pad], mode="reflect")

    # Random crop: each sample gets its own random top-left offset in
    # [0, 2*pad]. Vectorized via advanced indexing rather than a Python
    # per-sample loop -- build per-sample row/col index grids and gather.
    device = X_batch.device
    max_offset = 2 * pad
    offset_h = torch.randint(0, max_offset + 1, (B,), device=device)
    offset_w = torch.randint(0, max_offset + 1, (B,), device=device)

    row_idx = offset_h.view(B, 1, 1, 1) + torch.arange(H, device=device).view(1, 1, H, 1)
    col_idx = offset_w.view(B, 1, 1, 1) + torch.arange(W, device=device).view(1, 1, 1, W)
    row_idx = row_idx.expand(B, C, H, W)
    col_idx = col_idx.expand(B, C, H, W)
    batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, C, H, W)
    chan_idx = torch.arange(C, device=device).view(1, C, 1, 1).expand(B, C, H, W)

    cropped = padded[batch_idx, chan_idx, row_idx, col_idx]

    # Random horizontal flip, independent per sample.
    flip_mask = torch.rand(B, device=device) < 0.5
    flipped = cropped.flip(dims=[3])
    return torch.where(flip_mask.view(B, 1, 1, 1), flipped, cropped)


# ===========================================================================
# Pretrain -- plain cross-entropy SGD, module.train() throughout
# ===========================================================================

@torch.no_grad()
def eval_accuracy(module: nn.Module, X: Tensor, y: Tensor, chunk: int = 2048) -> float:
    module.eval()
    correct = 0
    for i in range(0, X.shape[0], chunk):
        logits = module(X[i:i + chunk])
        correct += (logits.argmax(-1) == y[i:i + chunk]).sum().item()
    module.train()
    return correct / X.shape[0]


def run_pretrain(module: nn.Module, data: dict[str, Any], n_epochs: int, batch_size: int,
                  lr: float, momentum: float, weight_decay: float, log_every: int,
                  augment: bool = True) -> dict:
    module.train()
    optimizer = torch.optim.SGD(module.parameters(), lr=lr, momentum=momentum,
                                 weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(data["X_train"], data["y_train"])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        module.train()
        epoch_loss = 0.0
        n_seen = 0
        for X_batch, y_batch in loader:
            if augment:
                X_batch = augment_batch(X_batch)
            optimizer.zero_grad()
            logits = module(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.shape[0]
            n_seen += X_batch.shape[0]
        scheduler.step()

        if epoch % log_every == 0 or epoch == n_epochs:
            train_acc = eval_accuracy(module, data["X_train"], data["y_train"])
            test_acc = eval_accuracy(module, data["X_test"], data["y_test"])
            avg_loss = epoch_loss / n_seen
            history.append({"epoch": epoch, "loss": avg_loss, "train_acc": train_acc, "test_acc": test_acc})
            print(f"  epoch {epoch:>4}/{n_epochs}  loss={avg_loss:.4f}  "
                  f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={time.perf_counter()-t0:.1f}s")

    module.eval()
    return {"history": history}


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-test", type=int, default=10_000)
    parser.add_argument("--n-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--no-augment", action="store_true",
                         help="Disable random-crop+flip augmentation (default: enabled). "
                              "Without it, expect train_acc to hit 1.000 and test_acc to "
                              "plateau in the low-to-mid 80s (memorization) well before "
                              "n_epochs is reached, rather than the ~91-92% standard "
                              "ResNet-20/CIFAR-10 recipes reach with augmentation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    print("=" * 72)
    print(f"ResNet20 pretrain (plain SGD, populates BatchNorm running stats)  "
          f"n_epochs={args.n_epochs}  device={DEVICE}  dtype={DTYPE}")
    print("=" * 72)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n[data]")
    data = load_cifar10_subset(args.n_train, args.n_test, args.seed, args.data_dir, DTYPE, DEVICE)

    print("\n[model]")
    module = ResNet20(activation=ACTIVATION).to(dtype=DTYPE, device=DEVICE)
    D = sum(p.numel() for p in module.parameters())
    print(f"  D = {D}  activation={ACTIVATION}")

    print("\n[pretrain]")
    rm_before = module.stem_bn.running_mean.clone()
    result = run_pretrain(
        module, data, n_epochs=args.n_epochs, batch_size=args.batch_size,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        log_every=args.log_every, augment=not args.no_augment,
    )
    rm_after = module.stem_bn.running_mean.clone()
    moved = not torch.equal(rm_before, rm_after)
    print(f"  pretrain done. stem_bn.running_mean moved from init default = {moved} (expect True)")
    assert moved, (
        "BatchNorm running_mean did not move from its init default (all-zeros) -- "
        "pretraining did not run long enough (or at all) to populate real statistics."
    )

    final_train_acc = result["history"][-1]["train_acc"]
    final_test_acc = result["history"][-1]["test_acc"]

    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"resnet20_pretrain_N{args.n_train}_epochs{args.n_epochs}"
    out_path = args.out / f"{name}.pt"
    torch.save({
        "state_dict": module.state_dict(),
        "architecture": "resnet20",
        "activation": ACTIVATION,
        "train_acc": final_train_acc,
        "test_acc": final_test_acc,
        "n_epochs": args.n_epochs,
        "n_train": args.n_train,
        "augment": not args.no_augment,
        "history": result["history"],
    }, out_path)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
