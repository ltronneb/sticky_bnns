"""
Standalone pure-SGD LeNet5 trainer for the MNIST target -- the frequentist
counterpart to lenet_reference.py.

lenet_reference.py fits a MAP point (Adam, full-batch, Gaussian log-prior in
the loss) plus an empirical-Fisher Sigma_inv, for use as a Boomerang
reference measure. THIS script instead fits a plain SGD-with-momentum
classifier: minibatches, cross-entropy only, no log-prior term, no
Sigma_inv. It exists so lenet_pixel_noise.ipynb can plot a genuine
frequentist baseline (Izmailov et al. 2021, Fig 15, use SGD in exactly this
role) alongside the sticky posteriors and the MAP point.

Deliberately kept separate from lenet_reference.py rather than bolted on as
a flag, because almost nothing is shared: the optimiser, the objective, the
batching, and the output contents all differ. The only things held in
common (and copied, not imported, per this tree's per-script-local-loader
convention) are the architecture constants and the MNIST loader split.

The saved checkpoint stores the trained weights BOTH as
  - `state_dict`  : the plain nn.Module state_dict, and
  - `x_ref`       : a flat [D] parameter vector in BayesianModule
                    param-order (torch.cat over module.parameters()),
so the notebook can load it exactly like a MAP checkpoint
(`runs["sgd"] = {"samples": w.unsqueeze(0)}`) with no extra plumbing. The
key is called `x_ref` purely for drop-in compatibility with that loader --
it is an SGD solution, not a MAP.

Usage:
    python -m sazz.gpu_friendly.scripts.lenet_sgd --epochs 30
    python -m sazz.gpu_friendly.scripts.lenet_sgd --full --epochs 40 --batch-size 128
    python -m sazz.gpu_friendly.scripts.lenet_sgd --n 10000 --epochs 50 --lr 0.05
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms

from sazz.gpu_friendly.models.neural_networks import LeNet5


DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
DTYPE = torch.float32 if DEVICE in ("cuda", "mps") else torch.float64

# Same architecture constants as lenet_reference.py -- a checkpoint trained
# with one activation/pool and loaded into a target built with another is a
# silent-garbage bug. The SGD baseline must match the samplers' LeNet5
# exactly for the pixel-noise comparison to be meaningful.
ACTIVATION = "tanh"
POOL = "avg"

DATA_DIR = Path("datasets")
OUT_DIR = Path("results/maps")


# ===========================================================================
# Data -- same split convention as lenet_reference.py::load_mnist
# ===========================================================================

def load_mnist(n: Optional[int], train_frac: float, seed: int, data_dir: Path,
                dtype: torch.dtype, device: str) -> dict[str, Any]:
    """
    n points drawn from MNIST's 60k TRAIN set, split train_frac/(1-train_frac)
    into (X_train, y_train)/(X_eval, y_eval). MNIST's own 10k TEST set is
    loaded in full, untouched by n/train_frac. Identical to
    lenet_reference.py::load_mnist so the SGD baseline sees the same data as
    the MAP.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_full = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    pool_size = len(train_full) if n is None else n
    pool_idx = rng.choice(len(train_full), size=pool_size, replace=False)
    n_train = int(round(pool_size * train_frac))
    train_idx = pool_idx[:n_train]
    eval_idx = pool_idx[n_train:]

    def stack(ds, idx):
        Xs, ys = zip(*(ds[int(i)] for i in idx))
        X = torch.stack(Xs).to(dtype=dtype, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        return X, y

    X_train, y_train = stack(train_full, train_idx)
    X_eval, y_eval = stack(train_full, eval_idx)
    X_test, y_test = stack(test_full, np.arange(len(test_full)))

    print(f"  loaded MNIST: pool={pool_size} (train_frac={train_frac}) -> "
          f"train={tuple(X_train.shape)}, eval={tuple(X_eval.shape)}, "
          f"test={tuple(X_test.shape)}")
    print(f"  train label counts:  {torch.bincount(y_train, minlength=10).tolist()}")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_eval": X_eval, "y_eval": y_eval,
        "X_test": X_test, "y_test": y_test,
    }


# ===========================================================================
# Train -- plain SGD + momentum, cross-entropy only (no prior)
# ===========================================================================

@torch.no_grad()
def evaluate(module: LeNet5, X: Tensor, y: Tensor, chunk: int = 2048) -> tuple[float, float]:
    """Returns (accuracy, mean cross-entropy) over (X, y)."""
    module.eval()
    correct, ce_sum = 0, 0.0
    for i in range(0, X.shape[0], chunk):
        logits = module(X[i:i + chunk])
        correct += (logits.argmax(-1) == y[i:i + chunk]).sum().item()
        ce_sum += F.cross_entropy(logits, y[i:i + chunk], reduction="sum").item()
    return correct / X.shape[0], ce_sum / X.shape[0]


def train_sgd(module: LeNet5, data: dict[str, Any], epochs: int, lr: float,
              batch_size: int, momentum: float, weight_decay: float,
              seed: int, log_every: int = 1) -> list[dict[str, float]]:
    """
    Standard minibatch SGD with Nesterov momentum and cosine-annealed LR.
    Objective is mean cross-entropy over the minibatch -- no log-prior, no
    Sigma_inv. weight_decay defaults to 0.0 so this is a genuinely
    unregularised frequentist fit; set it explicitly if an L2-regularised
    baseline is wanted instead.

    Returns a per-epoch history of {"epoch", "train_ce", "train_acc",
    "eval_ce", "eval_acc", "lr"} so the notebook can plot the SGD
    trajectory the same way it plots map_history.
    """
    X_train, y_train = data["X_train"], data["y_train"]
    N = X_train.shape[0]

    optimizer = torch.optim.SGD(module.parameters(), lr=lr, momentum=momentum,
                                nesterov=momentum > 0, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    g = torch.Generator(device=X_train.device).manual_seed(seed)
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        module.train()
        perm = torch.randperm(N, generator=g, device=X_train.device)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(module(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == epochs:
            tr_acc, tr_ce = evaluate(module, X_train, y_train)
            ev_acc, ev_ce = evaluate(module, data["X_eval"], data["y_eval"])
            cur_lr = scheduler.get_last_lr()[0]
            history.append({"epoch": epoch, "train_ce": tr_ce, "train_acc": tr_acc,
                            "eval_ce": ev_ce, "eval_acc": ev_acc, "lr": cur_lr})
            print(f"  epoch {epoch:>4}/{epochs}  train_ce={tr_ce:.4f}  "
                  f"train_acc={tr_acc:.4f}  eval_ce={ev_ce:.4f}  "
                  f"eval_acc={ev_acc:.4f}  lr={cur_lr:.2e}  "
                  f"elapsed={time.perf_counter() - t0:.1f}s")

    return history


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--n", type=int, default=60_000,
                         help="Pool size drawn from MNIST's 60k training set, "
                              "split --train-frac/(1-train-frac) into train/eval. "
                              "Ignored if --full is given. Default is the full 60k.")
    parser.add_argument("--full", action="store_true",
                         help="Use the full 60k MNIST training set (== --n 60000).")
    parser.add_argument("--train-frac", type=float, default=0.8,
                         help="Fraction of the pool used for training; the "
                              "remainder is a held-out eval split.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                         help="L2 penalty. Default 0.0 -- an unregularised "
                              "frequentist fit. Set > 0 for an L2 baseline.")
    parser.add_argument("--log-every", type=int, default=1,
                         help="Print/record a progress line every this many epochs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    n = None if args.full else args.n

    print("=" * 72)
    print(f"LeNet5 pure SGD baseline  |  pool={'full (60000)' if n is None else n}  "
          f"train_frac={args.train_frac}  epochs={args.epochs}  bs={args.batch_size}  "
          f"lr={args.lr}  device={DEVICE}  dtype={DTYPE}")
    print("=" * 72)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n[data]")
    data = load_mnist(n, args.train_frac, args.seed, args.data_dir, DTYPE, DEVICE)

    print("\n[model]")
    module = LeNet5(activation=ACTIVATION, pool=POOL).to(dtype=DTYPE, device=DEVICE)
    D = sum(p.numel() for p in module.parameters())
    print(f"  D = {D}  activation={ACTIVATION}  pool={POOL}")

    print("\n[SGD]")
    history = train_sgd(module, data, epochs=args.epochs, lr=args.lr,
                        batch_size=args.batch_size, momentum=args.momentum,
                        weight_decay=args.weight_decay, seed=args.seed,
                        log_every=args.log_every)

    train_acc, train_ce = evaluate(module, data["X_train"], data["y_train"])
    eval_acc, eval_ce = evaluate(module, data["X_eval"], data["y_eval"])
    test_acc, test_ce = evaluate(module, data["X_test"], data["y_test"])
    print(f"\n  SGD done  train_acc={train_acc:.4f}  eval_acc={eval_acc:.4f}  "
          f"test_acc={test_acc:.4f}  (test_ce={test_ce:.4f})")

    # Flat [D] vector in BayesianModule param-order: torch.cat over
    # module.parameters(), the exact order BayesianModule.build /
    # param_dict_fn use. Stored as `x_ref` for drop-in compatibility with
    # the notebook's MAP loader -- it is an SGD solution, not a MAP.
    w_flat = torch.cat([p.detach().flatten() for p in module.parameters()]).cpu()

    pool_size = 60000 if n is None else n
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"lenet_sgd_N{pool_size}_epochs{args.epochs}"
    out_path = args.out / f"{name}.pt"
    torch.save({
        "x_ref": w_flat,
        "state_dict": {k: v.detach().cpu() for k, v in module.state_dict().items()},
        "source": "pure_sgd",
        "architecture": "lenet5",
        "D": D,
        "activation": ACTIVATION,
        "pool": POOL,
        "optimizer": "sgd_nesterov",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "train_acc": train_acc,
        "eval_acc": eval_acc,
        "test_acc": test_acc,
        "train_ce": train_ce,
        "eval_ce": eval_ce,
        "test_ce": test_ce,
        "n_train": data["X_train"].shape[0],
        "n_eval": data["X_eval"].shape[0],
        "n_test": data["X_test"].shape[0],
        "pool_size": pool_size,
        "train_frac": args.train_frac,
        "seed": args.seed,
        "history": history,
    }, out_path)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
