"""Minimal MNIST CNN-BNN smoke test.

Builds a v2 modular target backed by the standard CNN from networks.py,
runs a short sticky PDMP chain, prints timing, and reports test accuracy.
The point is to confirm the pipeline (ParamSpec + functional_call + sticky
sampler + path resampling) works end-to-end on a Conv2d architecture.

N=10 training images is intentionally tiny — the posterior is wildly
underdetermined, but the goal is "does it run, how fast" rather than
"are the predictions any good."

Usage:
    python -m sazz.scripts.mnist_cnn
    python -m sazz.scripts.mnist_cnn --sampler sticky_zigzag
    python -m sazz.scripts.mnist_cnn --n-train 50 --n-skeleton 5000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchvision import datasets, transforms

from sazz.samplers.StickyAutomaticBoomerangSampler import (
    StickyAutomaticBoomerangSampler,
)
from sazz.samplers.StickyAutomaticZigZagSampler_torch import (
    StickyAutomaticZigZagSampler,
)
from sazz.models.bnn_torch import ModuleGaussianPrior, ModuleCategoricalLikelihood
from sazz.models.make_models import TorchTarget
from sazz.models.models_torch import BayesianModel
from sazz.models.networks import CNN
from sazz.utils.bnn_modular_utils import (
    ParamSpec, build_prior_precision, make_kappa_from_inclusion,
)
from sazz.utils.warmup import find_reference_bnn, tune_refresh_rate
from sazz.utils.sampling import (
    resample_boomerang_path_sticky, resample_zigzag_path_sticky,
)


torch.set_default_dtype(torch.float64)

# Knobs — kept small so it actually finishes
N_TRAIN     = 100
N_TEST      = 100
N_SKELETON  = 2_000
N_RESAMPLE  = 500
BURNIN_FRAC = 0.2

T_MAX_ZZ    = 0.1
GAMMA_ZZ    = 0.01

PRIOR_STD_W = 1.0
PRIOR_STD_B = 1.0
INCL_W      = 0.5     # spike-and-slab inclusion weight per layer

DATA_DIR    = Path("../datasets")


# ===========================================================================
# Data
# ===========================================================================

def load_mnist_subset(n_train: int, n_test: int, seed: int = 0):
    """Load MNIST and take small balanced-ish subsets. ToTensor scales to
    [0, 1], Normalize uses the standard MNIST statistics."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST(DATA_DIR, train=True,  download=True,
                                transform=transform)
    test_full  = datasets.MNIST(DATA_DIR, train=False, download=True,
                                transform=transform)

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(train_full), size=n_train, replace=False)
    test_idx  = rng.choice(len(test_full),  size=n_test,  replace=False)

    def stack(ds, idx):
        Xs, ys = zip(*(ds[int(i)] for i in idx))
        X = torch.stack(Xs).to(torch.float64)        # [N, 1, 28, 28]
        y = torch.tensor(ys, dtype=torch.long)       # [N]
        return X, y

    X_train, y_train = stack(train_full, train_idx)
    X_test,  y_test  = stack(test_full,  test_idx)

    print(f"  loaded MNIST subset: train={tuple(X_train.shape)}, "
          f"test={tuple(X_test.shape)}")
    print(f"  train label counts:  {torch.bincount(y_train, minlength=10).tolist()}")
    return X_train, y_train, X_test, y_test


# ===========================================================================
# Target
# ===========================================================================

def build_target(X_train, y_train, dtype=torch.float64) -> TorchTarget:
    """Build a CNN-BNN target for MNIST classification."""
    module = CNN(activation="relu").to(dtype=dtype)
    spec = ParamSpec.from_module(module)

    print(f"\n  CNN parameter groups (D = {spec.D}):")
    for name, shape, fan_in, is_b, can_freeze in zip(
        spec.names, spec.shapes, spec.fan_ins, spec.is_bias, spec.can_freeze
    ):
        print(f"    {name:<20} shape={tuple(shape)}  fan_in={fan_in}  "
              f"is_bias={is_b}  can_freeze={can_freeze}")

    prec = build_prior_precision(
        spec, PRIOR_STD_W, PRIOR_STD_B, fan_in_scaling=True,
        dtype=dtype, device="cpu",
    )
    prior = ModuleGaussianPrior(prec)
    likelihood = ModuleCategoricalLikelihood(module, spec, X_train, y_train)
    model = BayesianModel(prior, likelihood)

    print("\n  Running find_reference_bnn ...")
    t0 = time.perf_counter()
    x_ref, Sigma_inv = find_reference_bnn(
        model.energy, spec.D, model=model, dtype=dtype, device="cpu",
        reference="laplace_diag", n_steps=2000, lr=1e-2,
    )
    print(f"  find_reference done in {time.perf_counter() - t0:.1f}s  "
          f"||x_ref||_inf = {x_ref.abs().max().item():.3f}")

    return TorchTarget(
        name="mnist_cnn_bnn",
        D=spec.D,
        grad_target=model.grad_energy,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={"model": model, "spec": spec, "module": module},
    )


# ===========================================================================
# Sampler
# ===========================================================================

def build_sampler(name: str, target: TorchTarget):
    spec = target.meta["spec"]
    common = dict(grad_target=target.grad_target, D=target.D, thinning="pli")
    kappa = make_kappa_from_inclusion(
        spec=spec,
        prior_std_weight=PRIOR_STD_W,
        prior_inclusion_weight=INCL_W,
        fan_in_scaling=True,
    )
    print(f"  kappa: min={kappa.min().item():.3f}  "
          f"max={kappa.max().item():.3e}  "
          f"(weights vs biases — biases should be ~1e6)")

    if name == "sticky_zigzag":
        return StickyAutomaticZigZagSampler(
            **common, t_max=T_MAX_ZZ, gamma=GAMMA_ZZ, kappa=kappa,
            can_freeze=torch.tensor(spec.can_freeze, dtype=torch.bool)
        )
    if name == "sticky_boomerang":
        s = StickyAutomaticBoomerangSampler(
            **common, refresh_rate=1.0, kappa=kappa,
            can_freeze=torch.tensor(spec.can_freeze, dtype=torch.bool)
        )
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
        info = tune_refresh_rate(s, n_pilot=200)
        print(f"  tuned refresh_rate: "
              f"{info['lambda_r_old']:.3f} -> {info['lambda_r_new']:.3f}")
        return s
    raise ValueError(name)


def resample(name: str, result: dict, x_ref_np: np.ndarray) -> np.ndarray:
    pos = result["positions"].cpu().numpy()
    vel = result["velocities"].cpu().numpy()
    tim = result["times"].cpu().numpy()
    if name == "sticky_zigzag":
        return resample_zigzag_path_sticky(
            pos, vel, tim, N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
        )
    if name == "sticky_boomerang":
        return resample_boomerang_path_sticky(
            pos, vel, tim, x_ref_np,
            N_resample=N_RESAMPLE, burnin_frac=BURNIN_FRAC,
        )
    raise ValueError(name)


# ===========================================================================
# Predictions
# ===========================================================================

@torch.no_grad()
def predict_class_probs(samples: Tensor, target: TorchTarget,
                        X_test: Tensor) -> Tensor:
    """Average softmax probabilities across posterior samples. [N_test, 10]."""
    likelihood = target.meta["model"].likelihood
    X_test = X_test.to(dtype=likelihood.X.dtype, device=likelihood.X.device)
    probs = []
    for beta in samples:
        logits = likelihood.predict(beta, X_test)
        probs.append(torch.softmax(logits, dim=-1))
    return torch.stack(probs).mean(0)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", default="sticky_boomerang",
                        choices=["sticky_zigzag", "sticky_boomerang"])
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test",  type=int, default=N_TEST)
    parser.add_argument("--n-skeleton", type=int, default=N_SKELETON)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 72)
    print(f"MNIST CNN-BNN smoke test  |  sampler={args.sampler}  |  "
          f"n_train={args.n_train}  n_skeleton={args.n_skeleton}")
    print("=" * 72)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("\n[data]")
    X_train, y_train, X_test, y_test = load_mnist_subset(
        args.n_train, args.n_test, seed=args.seed,
    )

    print("\n[target]")
    target = build_target(X_train, y_train)

    print(f"\n[sampler] building {args.sampler}")
    sampler = build_sampler(args.sampler, target)

    print(f"\n[sampling] {args.n_skeleton} skeleton points ...")
    t0 = time.perf_counter()
    result = sampler.sample(N=args.n_skeleton, diagnostics=True)
    sample_time = time.perf_counter() - t0
    print(f"  sampling done in {sample_time:.1f}s  "
          f"({args.n_skeleton / sample_time:.1f} skel/s)")

    print(f"\n[resampling] to {N_RESAMPLE} discrete-time draws ...")
    t0 = time.perf_counter()
    samples_np = resample(args.sampler, result, target.x_ref.cpu().numpy())
    samples = torch.tensor(samples_np)
    print(f"  resample done in {time.perf_counter() - t0:.1f}s  "
          f"shape={tuple(samples.shape)}")

    print("\n[prediction]")
    t0 = time.perf_counter()
    probs = predict_class_probs(samples, target, X_test)
    pred_y = probs.argmax(-1)
    acc = (pred_y == y_test).float().mean().item()
    print(f"  prediction done in {time.perf_counter() - t0:.1f}s")
    print(f"  test accuracy on {args.n_test} test images: {acc:.3f}")
    print(f"  (chance level = 0.10; with {args.n_train} train images, "
          f"don't expect much)")

    # Sparsity diagnostic — sticky samplers should freeze a chunk of weights
    near_zero = (samples.abs() < 1e-8).float().mean().item()
    print(f"\n[sparsity] fraction of (sample, coord) pairs at 0: {near_zero:.2%}")

    print("\n✓ end-to-end CNN-BNN sampling works")


if __name__ == "__main__":
    main()
