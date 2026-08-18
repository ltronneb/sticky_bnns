"""
Standalone MAP + Sigma_inv (empirical Fisher) finder for the MNIST CNN-BNN
target, kept separate from utils/warmup.py::find_reference_bnn (the generic
path every other gpu_friendly script uses) so that path stays untouched --
this script owns its own pipeline and is free to diverge for CNN-scale
throughput.

Two deliberate divergences from find_reference_bnn, both load-bearing:
  - MAP init is the module's own (PyTorch-default) parameter values, NOT
    torch.randn(bm.D) -- for conv2.weight the fan-in-scaled prior std is
    ~0.06-0.3 depending on prior_std_weight, and a unit-variance start is
    far outside that; sazz/models/cnn_map.py already established this.
  - The empirical-Fisher diagonal is computed with ONE
    torch.func.vmap(torch.func.grad(...)) call over a batch of per-point
    log-likelihoods, instead of warmup.py's Python loop of n_batch
    individual torch.autograd.grad calls -- same math, no per-call
    kernel-launch overhead. vmap is applied over X.unsqueeze(1)/
    y.unsqueeze(1) (shape [n,1,1,28,28]/[n,1]), NOT raw X/y directly --
    vmap(in_dims=0) over unsqueezed X strips the batch dim entirely and
    CNN.forward's torch.flatten(x,1) then produces the wrong shape
    (confirmed by direct repro during planning).

Usage:
    python -m sazz.gpu_friendly.scripts.cnn_reference --n-steps 2000
    python -m sazz.gpu_friendly.scripts.cnn_reference --n 5000 --n-steps 2000
    python -m sazz.gpu_friendly.scripts.cnn_reference --full --batch-size 512 --n-steps 10_000
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

from sazz.gpu_friendly.models.neural_networks import CNN
from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.priors import build_fan_in_prior_precision


DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
DTYPE = torch.float32 if DEVICE in ("cuda", "mps") else torch.float64

# Must match mnist_cnn_grid.py's CNNConfig defaults exactly -- a checkpoint
# produced with one activation/pool/prior scale and loaded into a target
# built with another is a silent-garbage bug (asserted on load in
# mnist_cnn_grid.py's build_target), not a mere inefficiency.
ACTIVATION = "tanh"
POOL = "avg"
PRIOR_STD_W = 2.0
PRIOR_STD_B = 2.0

DATA_DIR = Path("datasets")
OUT_DIR = Path("results/maps")


# ===========================================================================
# Data (duplicated, not shared, with mnist_cnn_grid.py -- matches the
# existing per-script-local-loader convention in this tree)
# ===========================================================================

def load_mnist(n: Optional[int], train_frac: float, seed: int, data_dir: Path,
                dtype: torch.dtype, device: str) -> dict[str, Any]:
    """
    n: pool size drawn from MNIST's 60k TRAINING set (None = use all 60k --
    "the full dataset"). That pool is then split train_frac/(1-train_frac)
    into (X_train, y_train)/(X_eval, y_eval) -- an 80/20 split of the same
    pool by default, not two independently-sampled subsets.

    MNIST's own 10k TEST set is loaded separately, in full, untouched by n/
    train_frac -- it stays a genuine held-out set distinct from the 80/20
    split of the training pool, not something n/train_frac ever subsamples.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_full = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)

    pool_size = len(train_full) if n is None else n
    # rng.choice(..., replace=False) already returns indices in random order,
    # so slicing it directly into train/eval halves is a valid random split
    # -- no separate shuffle needed.
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

    print(f"  loaded MNIST: pool={pool_size} (train_frac={train_frac}) -> "
          f"train={tuple(X_train.shape)}, eval={tuple(X_eval.shape)}, "
          f"test={tuple(X_test.shape)} (MNIST's full held-out test set)")
    print(f"  train label counts:  {torch.bincount(y_train, minlength=10).tolist()}")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_eval": X_eval, "y_eval": y_eval,
        "X_test": X_test, "y_test": y_test,
    }


# ===========================================================================
# MAP -- Adam, module-parameter init (NOT torch.randn)
# ===========================================================================

def run_map(bm: BayesianModule, n_steps: int, lr: float,
            batch_size: Optional[int] = None,
            log_every: int = 50) -> tuple[Tensor, list[dict[str, float]]]:
    """
    Returns (x_ref, history) -- history is a list of {"step", "loss",
    "train_acc"} dicts logged at the same log_every cadence already used
    for the printed progress line, so a caller (e.g. an evaluation
    notebook) can plot the MAP trajectory instead of only seeing a single
    post-hoc accuracy number.

    batch_size=None (default): full-batch gradient every step, identical
    to the original behavior. batch_size=B: minibatch gradient via the
    standard unbiased-subsampling rescale (loss = -log_prior -
    (N/B)*log_lik_batch), following sazz/models/cnn_map.py's run_adam.
    History fields (loss/neg_log_lik/neg_log_prior) are always computed
    from the FULL training set at the log_every cadence regardless of
    batch_size, so they mean the same thing in both branches and stay
    comparable across runs -- they are diagnostics, not the optimized
    objective.
    """
    beta = torch.cat([p.detach().flatten() for p in bm.module.parameters()])
    beta = beta.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([beta], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.1)

    N = bm.X.shape[0]
    minibatch = batch_size is not None and batch_size < N
    if minibatch:
        assert not bm.learns_noise, "minibatch rescale double-counts the sigma prior"

    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        if minibatch:
            idx = torch.randint(0, N, (batch_size,), device=bm.X.device)
            ll = bm.log_likelihood.single(beta, bm.X[idx], bm.y[idx])
            scale = N / batch_size
            log_prior = -0.5 * (bm.prior_precision * beta ** 2).sum()
            loss = -log_prior - scale * ll
            if step == 1:
                with torch.no_grad():
                    print(f"  rescale check: mb={scale * -ll.item():.1f}  "
                          f"full={-full_log_lik(bm, beta.detach()):.1f}")
        else:
            loss = bm.energy(beta)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == n_steps:
            with torch.no_grad():
                acc = eval_accuracy(bm, beta.detach(), bm.X, bm.y)
                neg_log_lik = -full_log_lik(bm, beta.detach())
                neg_log_prior = 0.5 * (bm.prior_precision * beta.detach() ** 2).sum().item()
            loss_full = neg_log_lik + neg_log_prior
            history.append({"step": step, "loss": loss_full, "train_acc": acc,
                            "neg_log_lik": neg_log_lik, "neg_log_prior": neg_log_prior})
            print(f"  step {step:>6}/{n_steps}  loss={loss_full:.4f}  "
                f"nll={neg_log_lik:.4f}  neg_log_prior={neg_log_prior:.4f}  "
                f"train_acc={acc:.3f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                f"elapsed={time.perf_counter()-t0:.1f}s")

    return beta.detach(), history


@torch.no_grad()
def eval_accuracy(bm: BayesianModule, beta: Tensor, X: Tensor, y: Tensor,
                   chunk: int = 2048) -> float:
    correct = 0
    pd = bm.param_dict_fn(beta)
    for i in range(0, X.shape[0], chunk):
        logits = torch.func.functional_call(bm.module, pd, (X[i:i + chunk],))
        correct += (logits.argmax(-1) == y[i:i + chunk]).sum().item()
    return correct / X.shape[0]


@torch.no_grad()
def full_log_lik(bm: BayesianModule, beta: Tensor, chunk: int = 2048) -> float:
    return sum(
        bm.log_likelihood.single(beta, bm.X[i:i + chunk], bm.y[i:i + chunk]).item()
        for i in range(0, bm.X.shape[0], chunk)
    )


# ===========================================================================
# Sigma_inv -- vmapped empirical Fisher diagonal (replaces warmup.py's loop)
# ===========================================================================

def empirical_fisher_diag_vmapped(bm: BayesianModule, x_ref: Tensor, n_batch: int) -> Tensor:
    """
    Sigma_inv_ii = prior_prec_i + mean_i(grad_i log p(y_i|beta))^2, matching
    warmup.py::_empirical_fisher_diag's math exactly -- only the execution
    strategy differs (one vmapped grad call vs. n_batch individual
    autograd.grad calls).
    """
    X, y = bm.X, bm.y
    N = X.shape[0]
    n = min(n_batch, N)
    idx = torch.randperm(N, device=X.device)[:n]

    X_batch = X[idx].unsqueeze(1)  # [n, 1, 1, 28, 28] -- verified shape, see
    y_batch = y[idx].unsqueeze(1)  # [n, 1]           -- module docstring

    log_prob_single = bm.log_likelihood.single

    def per_point_log_prob(beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        return log_prob_single(beta, X_i, y_i)

    grad_fn = torch.func.grad(per_point_log_prob, argnums=0)
    grads = torch.func.vmap(grad_fn, in_dims=(None, 0, 0))(x_ref, X_batch, y_batch)  # [n, D]

    fisher = (grads ** 2).mean(dim=0)
    return fisher


def empirical_fisher_diag_scaled(bm: BayesianModule, x_ref: Tensor, n_batch: int) -> Tensor:
    """
    Sigma_inv_ii = prior_prec_i + N * mean_i(grad_i log p(y_i|beta))^2 --
    an unbiased subsampled estimate of the full-data observed-information
    SUM (not the mean empirical_fisher_diag_vmapped returns), so Sigma_inv
    is comparable across pool sizes instead of implicitly assuming
    whatever n_train a given run happened to use. N = bm.X.shape[0] is the
    size of the training split bm was built on.
    """
    N = bm.X.shape[0]
    fisher_mean = empirical_fisher_diag_vmapped(bm, x_ref, n_batch)
    return N * fisher_mean


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--n", type=int, default=10_000,
                         help="Pool size drawn from MNIST's 60k training set, "
                              "split --train-frac/(1-train-frac) into train/eval. "
                              "Ignored if --full is given.")
    parser.add_argument("--full", action="store_true",
                         help="Use the full 60k MNIST training set as the pool "
                              "(equivalent to --n 60000). Overrides --n.")
    parser.add_argument("--train-frac", type=float, default=0.8,
                         help="Fraction of the pool used for training; the "
                              "remainder is held out as an eval split.")
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Minibatch size for the MAP step. Default (None) "
                              "is full-batch gradients, matching the original "
                              "behavior. When set and < n_train, uses the "
                              "standard unbiased-subsampling rescale.")
    parser.add_argument("--log-every", type=int, default=50,
                         help="Print/record a progress line every this many "
                              "MAP steps.")
    parser.add_argument("--n-fisher-batch", type=int, default=1024,
                         help="Subsample size for the empirical Fisher "
                              "diagonal. Now load-bearing for Sigma_inv's "
                              "magnitude (relative error ~ O(n^-1/2)), so "
                              "kept larger than the pre-rescale default of 128.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    n = None if args.full else args.n

    print("=" * 72)
    print(f"CNN reference (MAP + vmapped Fisher)  |  pool={'full (60000)' if n is None else n}  "
          f"train_frac={args.train_frac}  n_steps={args.n_steps}  device={DEVICE}  dtype={DTYPE}")
    print("=" * 72)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n[data]")
    data = load_mnist(n, args.train_frac, args.seed, args.data_dir, DTYPE, DEVICE)

    print("\n[model]")
    module = CNN(activation=ACTIVATION, pool=POOL)
    prec = build_fan_in_prior_precision(
        module, PRIOR_STD_W, PRIOR_STD_B, fan_in_scaling=True, dtype=DTYPE, device=DEVICE,
    )
    bm = BayesianModule.build(
        module, likelihood="categorical",
        X=data["X_train"], y=data["y_train"],
        prior_precision=prec, dtype=DTYPE, device=DEVICE,
    )
    print(f"  D = {bm.D}  activation={ACTIVATION}  pool={POOL}  "
          f"prior_std_weight={PRIOR_STD_W}  prior_std_bias={PRIOR_STD_B}")

    print("\n[MAP]")
    x_ref, map_history = run_map(bm, n_steps=args.n_steps, lr=args.lr,
                                  batch_size=args.batch_size, log_every=args.log_every)
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

    pool_size = 60000 if n is None else n
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"cnn_reference_N{pool_size}_steps{args.n_steps}"
    out_path = args.out / f"{name}.pt"
    torch.save({
        "x_ref": x_ref.cpu(),
        "Sigma_inv": Sigma_inv.cpu(),
        "sigma_inv_source": "prior_plus_fisher",
        "D": bm.D,
        "activation": ACTIVATION,
        "pool": POOL,
        "fan_in_scaling": True,
        "prior_std_weight": PRIOR_STD_W,
        "prior_std_bias": PRIOR_STD_B,
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
    }, out_path)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
