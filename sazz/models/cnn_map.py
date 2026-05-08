"""Standalone MAP finder for the MNIST CNN-BNN.

Trains the CNN to the MAP estimate using either Adam + cosine annealing or
SGD with momentum + linear warmup, then saves x_ref and Sigma_inv (diagonal
prior precision) to a .pt file that mnist_cnn.py can load directly.

Usage:
    # Adam + cosine annealing (default)
    python -m sazz.models.cnn_map --n-train 1000 --optimizer adam

    # SGD + momentum + warmup
    python -m sazz.models.cnn_map --n-train 1000 --optimizer sgd

    # Then load in mnist_cnn.py:
    python -m sazz.scripts.bnns.mnist_cnn --map-path results/maps/cnn_map.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from sazz.models.neural_networks import CNN
from sazz.models.priors import ModuleGaussianPrior
from sazz.models.likelihoods import ModuleCategoricalLikelihood
from sazz.models.model import BayesianModel
from sazz.utils.bnn_utils import ParamSpec, build_prior_precision

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

PRIOR_STD_W = 5.0
PRIOR_STD_B = 5.0
DATA_DIR    = Path("../datasets")
OUT_DIR     = Path("results/maps")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_mnist(n_train: int, n_test: int, seed: int = 0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_full  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(train_full), size=n_train, replace=False)
    test_idx  = rng.choice(len(test_full),  size=n_test,  replace=False)

    def stack(ds, idx):
        Xs, ys = zip(*(ds[int(i)] for i in idx))
        return (torch.stack(Xs).to(torch.float64),
                torch.tensor(ys, dtype=torch.long))

    X_train, y_train = stack(train_full, train_idx)
    X_test,  y_test  = stack(test_full,  test_idx)
    print(f"  train={tuple(X_train.shape)}  test={tuple(X_test.shape)}")
    print(f"  label counts: {torch.bincount(y_train, minlength=10).tolist()}")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_accuracy(model: BayesianModel, beta: torch.Tensor,
                  X: torch.Tensor, y: torch.Tensor) -> float:
    logits = model.likelihood.predict(beta, X)
    return (logits.argmax(-1) == y).float().mean().item()


# ---------------------------------------------------------------------------
# Adam + cosine annealing
# ---------------------------------------------------------------------------

def run_adam(model: BayesianModel, D: int,
             n_steps: int, lr: float, lr_min: float,
             batch_size: int | None, X: torch.Tensor, y: torch.Tensor,
             log_every: int = 500) -> torch.Tensor:
    """MAP via Adam with cosine LR annealing."""
    beta = torch.cat([p.detach().flatten()
                      for p in model.likelihood.module.parameters()])
    beta = beta.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([beta], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr_min
    )

    N = X.shape[0]
    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        if batch_size is not None and batch_size < N:
            idx = torch.randint(0, N, (batch_size,))
            ll = model.likelihood.log_prob_single(beta, X[idx], y[idx])
            scale = N / batch_size
            loss = -model.prior.log_prob(beta) - scale * ll
        else:
            loss = model.energy(beta)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == n_steps:
            with torch.no_grad():
                acc = eval_accuracy(model, beta.detach(), X, y)
            print(f"  step {step:>6}/{n_steps}  loss={loss.item():.4f}  "
                  f"train_acc={acc:.3f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.perf_counter()-t0:.1f}s")

    return beta.detach()


# ---------------------------------------------------------------------------
# SGD + momentum + linear warmup
# ---------------------------------------------------------------------------

def run_sgd(model: BayesianModel, D: int,
            n_steps: int, lr: float, momentum: float,
            warmup_steps: int, batch_size: int | None,
            X: torch.Tensor, y: torch.Tensor,
            log_every: int = 500) -> torch.Tensor:
    """MAP via SGD with momentum and linear LR warmup then cosine decay."""
    beta = torch.cat([p.detach().flatten()
                      for p in model.likelihood.module.parameters()])
    beta = beta.clone().requires_grad_(True)

    optimizer = torch.optim.SGD([beta], lr=lr, momentum=momentum)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    N = X.shape[0]
    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        if batch_size is not None and batch_size < N:
            idx = torch.randint(0, N, (batch_size,))
            ll = model.likelihood.log_prob_single(beta, X[idx], y[idx])
            scale = N / batch_size
            loss = -model.prior.log_prob(beta) - scale * ll
        else:
            loss = model.energy(beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([beta], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == n_steps:
            with torch.no_grad():
                acc = eval_accuracy(model, beta.detach(), X, y)
            print(f"  step {step:>6}/{n_steps}  loss={loss.item():.4f}  "
                  f"train_acc={acc:.3f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.perf_counter()-t0:.1f}s")

    return beta.detach()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--n-train",    type=int,   default=32)
    parser.add_argument("--n-test",     type=int,   default=10)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--optimizer",  default="adam", choices=["adam", "sgd"])
    parser.add_argument("--n-steps",    type=int,   default=10_000)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--lr-min",     type=float, default=1e-4,
                        help="Minimum LR for cosine annealing (Adam only).")
    parser.add_argument("--momentum",   type=float, default=0.9,
                        help="Momentum (SGD only).")
    parser.add_argument("--warmup",     type=int,   default=500,
                        help="Linear warmup steps (SGD only).")
    parser.add_argument("--batch-size", type=int,   default=None,
                        help="Minibatch size. Default: full data.")
    parser.add_argument("--log-every",  type=int,   default=500)
    parser.add_argument("--out",        type=Path,  default=OUT_DIR)
    parser.add_argument("--name",       type=str,   default=None,
                        help="Output filename stem. Default: auto-generated.")
    args = parser.parse_args()

    print("=" * 72)
    print(f"CNN MAP  |  optimizer={args.optimizer}  n_train={args.n_train}  "
          f"n_steps={args.n_steps}  batch_size={args.batch_size}")
    print("=" * 72)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n[data]")
    X_train, y_train, X_test, y_test = load_mnist(
        args.n_train, args.n_test, seed=args.seed
    )

    print("\n[model]")
    module = CNN(activation="relu").to(torch.float64)
    spec   = ParamSpec.from_module(module)
    prec   = build_prior_precision(spec, PRIOR_STD_W, PRIOR_STD_B,
                                   fan_in_scaling=True)
    prior      = ModuleGaussianPrior(prec)
    likelihood = ModuleCategoricalLikelihood(module, spec, X_train, y_train)
    model      = BayesianModel(prior, likelihood)
    print(f"  D = {spec.D}")

    print(f"\n[MAP — {args.optimizer}]")
    t0 = time.perf_counter()
    if args.optimizer == "adam":
        x_ref = run_adam(
            model, spec.D,
            n_steps=args.n_steps, lr=args.lr, lr_min=args.lr_min,
            batch_size=args.batch_size,
            X=X_train, y=y_train, log_every=args.log_every,
        )
    else:
        x_ref = run_sgd(
            model, spec.D,
            n_steps=args.n_steps, lr=args.lr, momentum=args.momentum,
            warmup_steps=args.warmup, batch_size=args.batch_size,
            X=X_train, y=y_train, log_every=args.log_every,
        )
    elapsed = time.perf_counter() - t0
    print(f"\n  MAP done in {elapsed:.1f}s  ||x_ref||_inf={x_ref.abs().max():.3f}")

    with torch.no_grad():
        train_acc = eval_accuracy(model, x_ref, X_train, y_train)
        test_acc  = eval_accuracy(model, x_ref, X_test,  y_test)
    print(f"  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    # Sigma_inv is the diagonal prior precision — the same object
    # find_reference_bnn returns, so mnist_cnn.py can load it directly.
    Sigma_inv = prec.clone()

    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or (
        f"cnn_map_{args.optimizer}_N{args.n_train}_steps{args.n_steps}"
    )
    out_path = args.out / f"{name}.pt"
    torch.save({
        "x_ref":       x_ref,
        "Sigma_inv":   Sigma_inv,
        "D":           spec.D,
        "n_train":     args.n_train,
        "optimizer":   args.optimizer,
        "n_steps":     args.n_steps,
        "train_acc":   train_acc,
        "test_acc":    test_acc,
        "prior_std_weight": PRIOR_STD_W,
        "prior_std_bias":   PRIOR_STD_B,
    }, out_path)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
