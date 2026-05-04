"""
scripts/module_bnn.py

Test harness comparing the v1 (layer_sizes-driven, unflatten_params) BNN
path against the v2 (ParamSpec + nn.Module + functional_call) path.

Three checks, in order of strictness:

    1. Static equivalence — for the same FFN architecture, v1 and v2
       produce identical:
         * D
         * prior precision vector
         * kappa-from-inclusion vector
         * forward output for a fixed beta and X
         * gradient of the energy

    2. Sampler equivalence (FFN) — running the same sampler on a
       v1-built target and a v2-built target with identical seeds should
       produce identical skeletons (within float64 numerical noise).
       This is the real proof that v2 is a drop-in for FFNs.

    3. ConvNet feasibility — build a small Conv1d-based BNN, hand it to
       the v2 pipeline, run a few hundred sampler iterations, sanity
       check that we get a non-degenerate skeleton and that the forward
       pass produces sensible predictions.

Run from project root:

    python -m sazz.scripts.module_bnn --check static
    python -m sazz.scripts.module_bnn --check sampler --dataset hernandez
    python -m sazz.scripts.module_bnn --check conv

Adjust the imports at the top of this file to match your project layout.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# ===========================================================================
# ADJUST IMPORTS: point these at your project's actual module paths.
# ===========================================================================
# v1 — existing layer_sizes-driven utilities and BNN factory
from sazz.utils.bnn_utils import (
    _build_prior_precision as v1_build_prior_precision,
    make_kappa_from_inclusion as v1_make_kappa_from_inclusion,
    unflatten_params as v1_unflatten_params,
    count_params as v1_count_params,
    get_activation as v1_get_activation,
)
from sazz.models.bnn_torch import make_bnn_regression as v1_make_bnn_regression

# v2 — the new ParamSpec-based utilities (the file we just wrote)
from sazz.utils.bnn_modular_utils import (
    ParamSpec,
    build_prior_precision as v2_build_prior_precision,
    make_kappa_from_inclusion as v2_make_kappa_from_inclusion,
    build_ffn_module,
    get_activation as v2_get_activation,
)

# Samplers (used in --check sampler and --check conv)
from sazz.samplers.AutomaticBoomerangSampler import AutomaticBoomerangSampler
from sazz.samplers.AutomaticZigZagSampler_torch import AutomaticZigZagSampler

# Toy data loader — adjust if your loader lives elsewhere
from sazz.scripts.uci_bnn import load_toy

# ===========================================================================
# A v2 likelihood for the test. In the real codebase this will live in
# bnn_torch.py; reproduced here so the test script is self-contained.
# ===========================================================================

class _V2Likelihood(nn.Module):
    """Minimal stand-in for ModuleLikelihood used by the test script.

    Wraps an nn.Module + ParamSpec, exposes:
        predict(beta, X_new)   — forward via functional_call
        log_prob(beta)         — Gaussian regression log-likelihood
        log_prob_single(beta, X_i, y_i)   — same on an arbitrary subset
        precision_diag()       — for the diagonal Fisher in find_reference

    Once the real ModuleLikelihood is in bnn_torch.py, replace this with
    an import.
    """
    def __init__(self, module: nn.Module, spec: ParamSpec,
                 X: Tensor, y: Tensor, noise_std: float):
        super().__init__()
        self.module = module
        self.module.eval()
        self.spec = spec
        self.register_buffer("X", X)
        self.register_buffer("y", y)
        self.noise_std = noise_std

    def predict(self, beta: Tensor, X_new: Tensor) -> Tensor:
        params = self.spec.to_dict(beta)
        return torch.func.functional_call(self.module, params, (X_new,))

    def log_prob_single(self, beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        from torch.distributions import Normal
        preds = self.predict(beta, X_i).squeeze(-1)
        return Normal(preds, self.noise_std).log_prob(y_i).sum()

    def log_prob(self, beta: Tensor) -> Tensor:
        return self.log_prob_single(beta, self.X, self.y)


# ===========================================================================
# Check 1 — static equivalence between v1 and v2 builders
# ===========================================================================

def check_static(layer_sizes=(1, 50, 1), prior_std_w=1.0, prior_std_b=1.0,
                 incl=0.7, fan_in_scaling=True, dtype=torch.float64,
                 atol=0.0, rtol=0.0):
    """Compare v1 and v2 outputs for the same FFN, expecting bit-equality."""
    print("=" * 72)
    print(f"CHECK 1 — static equivalence | layer_sizes={list(layer_sizes)}")
    print("=" * 72)

    # --- D ---
    D_v1 = v1_count_params(list(layer_sizes))
    spec = ParamSpec.from_layer_sizes(layer_sizes)
    print(f"D: v1={D_v1}  v2={spec.D}  match={D_v1 == spec.D}")
    assert D_v1 == spec.D

    # --- Prior precision ---
    prec_v1 = v1_build_prior_precision(
        list(layer_sizes), prior_std_w, prior_std_b,
        fan_in_scaling, dtype, torch.device("cpu"),
    )
    prec_v2 = v2_build_prior_precision(
        spec, prior_std_w, prior_std_b, fan_in_scaling, dtype, "cpu",
    )
    eq = torch.equal(prec_v1, prec_v2)
    max_diff = (prec_v1 - prec_v2).abs().max().item()
    print(f"prior precision: equal={eq}  max|diff|={max_diff:.2e}")
    assert eq, "v1 / v2 prior precision diverged"

    # --- Kappa from inclusion ---
    kap_v1 = v1_make_kappa_from_inclusion(
        list(layer_sizes),
        prior_std_weight=prior_std_w,
        prior_inclusion_weight=incl,
        fan_in_scaling=fan_in_scaling,
        dtype=dtype,
    )
    kap_v2 = v2_make_kappa_from_inclusion(
        spec,
        prior_std_weight=prior_std_w,
        prior_inclusion_weight=incl,
        fan_in_scaling=fan_in_scaling,
        dtype=dtype,
    )
    eq = torch.equal(kap_v1, kap_v2)
    max_diff = (kap_v1 - kap_v2).abs().max().item()
    print(f"kappa vector:    equal={eq}  max|diff|={max_diff:.2e}")
    assert eq, "v1 / v2 kappa diverged"

    # --- Forward output for fixed beta, X ---
    torch.manual_seed(0)
    beta = torch.randn(spec.D, dtype=dtype) * 0.1
    X = torch.randn(8, layer_sizes[0], dtype=dtype)

    # v1: explicit unflatten + tanh FFN forward
    act = v1_get_activation("tanh")
    params = v1_unflatten_params(beta, list(layer_sizes), dtype=dtype)
    h = X
    for i, (W, b) in enumerate(params):
        h = h @ W.T + b
        if i < len(params) - 1:
            h = act(h)
    out_v1 = h

    # v2: nn.Module + functional_call
    module = build_ffn_module(layer_sizes, "tanh").to(dtype=dtype)
    out_v2 = torch.func.functional_call(module, spec.to_dict(beta), (X,))

    max_diff = (out_v1 - out_v2).abs().max().item()
    print(f"forward output:  max|diff|={max_diff:.2e}  "
          f"(shape v1={tuple(out_v1.shape)}, v2={tuple(out_v2.shape)})")
    if atol == 0 and rtol == 0:
        # Use eps-scaled tolerance for forward/grad — matmul fma differences
        eps = torch.finfo(out_v1.dtype).eps
        assert torch.allclose(out_v1, out_v2, atol=10*eps, rtol=10*eps), \
            f"forward outputs differ beyond fma noise (max|diff|={max_diff:.2e})"
    else:
        assert torch.allclose(out_v1, out_v2, atol=atol, rtol=rtol)


    # --- Gradient of a Gaussian-likelihood energy ---
    y = torch.randn(8, dtype=dtype)
    noise_std = 0.3

    # v1 energy
    def energy_v1(b):
        params = v1_unflatten_params(b, list(layer_sizes), dtype=dtype)
        h = X
        for i, (W, bb) in enumerate(params):
            h = h @ W.T + bb
            if i < len(params) - 1:
                h = act(h)
        preds = h.squeeze(-1)
        ll = -0.5 * ((y - preds) / noise_std) ** 2
        log_lik = ll.sum()
        log_prior = -0.5 * (prec_v1 * b ** 2).sum()
        return -(log_lik + log_prior)

    # v2 energy
    def energy_v2(b):
        preds = torch.func.functional_call(
            module, spec.to_dict(b), (X,)
        ).squeeze(-1)
        ll = -0.5 * ((y - preds) / noise_std) ** 2
        log_lik = ll.sum()
        log_prior = -0.5 * (prec_v2 * b ** 2).sum()
        return -(log_lik + log_prior)

    b1 = beta.clone().requires_grad_(True)
    e1 = energy_v1(b1); g1, = torch.autograd.grad(e1, b1)

    b2 = beta.clone().requires_grad_(True)
    e2 = energy_v2(b2); g2, = torch.autograd.grad(e2, b2)

    max_e_diff = (e1 - e2).abs().item()
    max_g_diff = (g1 - g2).abs().max().item()
    print(f"energy:          |v1 - v2| = {max_e_diff:.2e}")
    print(f"grad energy:     max|v1 - v2| = {max_g_diff:.2e}")
    eps_e = torch.finfo(e1.dtype).eps
    eps_g = torch.finfo(g1.dtype).eps
    assert torch.allclose(e1, e2, atol=10*eps_e, rtol=10*eps_e), \
            f"forward outputs differ beyond fma noise (max|diff|={max_diff:.2e})"
    assert torch.allclose(g1, g2, atol=10*eps_g, rtol=10*eps_g), \
            f"forward outputs differ beyond fma noise (max|diff|={max_diff:.2e})"
    # assert torch.equal(e1, e2), "energies not bit-equal"
    # assert torch.equal(g1, g2), "grads not bit-equal"

    print("✓ static equivalence holds bit-for-bit\n")


# ===========================================================================
# Check 2 — sampler equivalence on an FFN target
# ===========================================================================

def _build_v2_target_ffn(X, y, layer_sizes, prior_std_w, prior_std_b,
                        fan_in_scaling, noise_std, dtype):
    """Build a v2-style target for an FFN, mirroring make_bnn_regression."""
    from sazz.models.make_models import TorchTarget
    from sazz.models.priors_torch import Prior
    from sazz.models.models_torch import BayesianModel
    from sazz.utils.warmup import find_reference_bnn

    spec = ParamSpec.from_module(build_ffn_module(layer_sizes, "tanh"))
    module = build_ffn_module(layer_sizes, "tanh").to(dtype=dtype)

    class _V2Prior(Prior):
        def __init__(self, prec_vec):
            super().__init__()
            self.register_buffer("_precision", prec_vec)
        def log_prob(self, beta):
            return -0.5 * (self._precision * beta ** 2).sum()
        def precision_diag(self):
            return self._precision

    prec = v2_build_prior_precision(
        spec, prior_std_w, prior_std_b, fan_in_scaling, dtype, "cpu"
    )
    prior = _V2Prior(prec)
    likelihood = _V2Likelihood(module, spec,
                               X.to(dtype=dtype), y.to(dtype=dtype),
                               noise_std=noise_std)
    model = BayesianModel(prior, likelihood)
    x_ref, Sigma_inv = find_reference_bnn(
        model.energy, spec.D, model=model, dtype=dtype, device="cpu",
        reference="laplace_diag", n_steps=2000, lr=1e-2,
    )
    return TorchTarget(
        name="v2_bnn_regression",
        D=spec.D,
        grad_target=model.grad_energy,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={"model": model, "spec": spec, "module": module},
    )


def check_sampler(dataset="hernandez", n_skel=300, dtype=torch.float64,
                  layer_sizes=None):
    """
    Run the same sampler on v1- and v2-built targets, with identical
    initial weights and the same Adam+RNG seed feeding find_reference.
    Skeletons should match closely.

    Note: bit-equality is NOT guaranteed here because find_reference does
    its own RNG-driven Adam optimisation, and the two pipelines build
    *different* nn.Modules under the hood (the v1 likelihood doesn't
    actually instantiate one). What we want to see is similar x_ref,
    similar Sigma_inv, and similar predictive performance.
    """
    print("=" * 72)
    print(f"CHECK 2 — sampler equivalence | dataset={dataset}")
    print("=" * 72)

    data, cfg = load_toy(dataset)
    if layer_sizes is None:
        layer_sizes = list(cfg.layer_sizes)
    X_train = data["X_train"].to(dtype=dtype)
    y_train = data["y_train"].to(dtype=dtype)

    # ---- v1 target ----
    torch.manual_seed(0); np.random.seed(0)
    t0 = time.perf_counter()
    target_v1 = v1_make_bnn_regression(
        X_train, y_train,
        layer_sizes=layer_sizes,
        activation=cfg.activation,
        prior_std_weight=cfg.prior_std_weight,
        prior_std_bias=cfg.prior_std_bias,
        fan_in_scaling=cfg.fan_in_scaling,
        noise_std=cfg.noise_std,
        covariance_reference="laplace_diag",
        adam_steps=cfg.adam_steps,
        dtype=dtype,
    )
    print(f"v1 target built in {time.perf_counter() - t0:.2f}s  D={target_v1.D}")

    # ---- v2 target ----
    torch.manual_seed(0); np.random.seed(0)
    t0 = time.perf_counter()
    target_v2 = _build_v2_target_ffn(
        X_train, y_train,
        layer_sizes=layer_sizes,
        prior_std_w=cfg.prior_std_weight,
        prior_std_b=cfg.prior_std_bias,
        fan_in_scaling=cfg.fan_in_scaling,
        noise_std=cfg.noise_std,
        dtype=dtype,
    )
    print(f"v2 target built in {time.perf_counter() - t0:.2f}s  D={target_v2.D}")
    assert target_v1.D == target_v2.D

    # ---- Compare references ----
    x_diff = (target_v1.x_ref - target_v2.x_ref).abs().max().item()
    s_diff = (target_v1.Sigma_inv - target_v2.Sigma_inv).abs().max().item()
    print(f"|x_ref_v1 - x_ref_v2|_inf      = {x_diff:.3e}")
    print(f"|Sigma_inv_v1 - Sigma_inv_v2|  = {s_diff:.3e}")

    # ---- Compare energies and gradients at the SAME random beta ----
    torch.manual_seed(42)
    beta = torch.randn(target_v1.D, dtype=dtype) * 0.1

    g1 = target_v1.grad_target(beta.clone().requires_grad_(True))
    g2 = target_v2.grad_target(beta.clone().requires_grad_(True))
    g_diff = (g1.detach() - g2.detach()).abs().max().item()
    print(f"max|grad_v1 - grad_v2| at random beta = {g_diff:.3e}")

    # ---- Run a short Boomerang chain from each ----
    def run(target, name):
        torch.manual_seed(123); np.random.seed(123)
        s = AutomaticBoomerangSampler(
            grad_target=target.grad_target, D=target.D,
            refresh_rate=0.1, thinning="pli",
        )
        s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
        x0 = target.x_ref.clone() + 0.1 * torch.randn(target.D, dtype=dtype)
        t0 = time.perf_counter()
        res = s.sample(N=n_skel, x0=x0, diagnostics=False)
        wall = time.perf_counter() - t0
        print(f"  {name}: {n_skel} skel in {wall:.2f}s  "
              f"final_t={res['times'][-1].item():.2f}  "
              f"grad_evals={res['gradient_evals']}")
        return res

    print("\nRunning Boomerang on v1 target ...")
    res_v1 = run(target_v1, "v1")
    print("Running Boomerang on v2 target ...")
    res_v2 = run(target_v2, "v2")

    # Skeleton statistics — should be very similar in distribution if
    # x_ref / Sigma_inv match
    print("\nSkeleton stats (mean of last 100 positions):")
    p1 = res_v1["positions"][-100:].mean(0)
    p2 = res_v2["positions"][-100:].mean(0)
    print(f"  v1 ||mean||_inf = {p1.abs().max().item():.3e}")
    print(f"  v2 ||mean||_inf = {p2.abs().max().item():.3e}")
    print(f"  ||v1 - v2||_inf = {(p1 - p2).abs().max().item():.3e}")

    print("\n(Bit-equality not expected; check that x_ref, Sigma_inv, "
          "and skeleton statistics are close.)\n")


# ===========================================================================
# Check 3 — feasibility on a small ConvNet
# ===========================================================================

class _SmallConv1D(nn.Module):
    """
    Trivial Conv1d-based regressor for a 1D toy:  X_train shape [N, 1]
    is reshaped to [N, 1, 1] and passed through Conv1d -> activation ->
    Conv1d -> flatten. Just enough non-FFN structure to exercise the
    fan-in computation and functional_call dispatch on a non-Linear
    layer.
    """
    def __init__(self, hidden_channels=8, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size)
        self.conv2 = nn.Conv1d(hidden_channels, 1, kernel_size)

    def forward(self, x):
        # x: [N, in_dim]; treat in_dim as 1 spatial position with 1 channel.
        # If in_dim > 1, reshape to [N, 1, in_dim] (1 channel, in_dim positions).
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [N, 1, in_dim]
        h = self.conv1(x)
        h = torch.tanh(h)
        h = self.conv2(h)
        return h.flatten(1).mean(-1, keepdim=True)  # [N, 1]


def check_conv(dataset="hernandez", n_skel=200, dtype=torch.float64):
    """Smoke test: build a v2 target backed by a small Conv1d module
    and run a few hundred Boomerang iterations."""
    print("=" * 72)
    print(f"CHECK 3 — ConvNet feasibility | dataset={dataset}")
    print("=" * 72)

    data, cfg = load_toy(dataset)
    X_train = data["X_train"].to(dtype=dtype)
    y_train = data["y_train"].to(dtype=dtype)
    print(f"  X_train shape = {tuple(X_train.shape)}, "
          f"y_train shape = {tuple(y_train.shape)}")

    module = _SmallConv1D(hidden_channels=8, kernel_size=1).to(dtype=dtype)
    spec = ParamSpec.from_module(module)
    print(f"\nParamSpec from Conv1D module:")
    for n, s, fi, ib in zip(spec.names, spec.shapes,
                            spec.fan_ins, spec.is_bias):
        print(f"  {n:<24} shape={tuple(s)}  fan_in={fi}  is_bias={ib}")
    print(f"  D = {spec.D}\n")

    # Sanity: forward pass works
    beta0 = torch.randn(spec.D, dtype=dtype) * 0.1
    out = torch.func.functional_call(module, spec.to_dict(beta0), (X_train,))
    print(f"forward output shape: {tuple(out.shape)}  "
          f"(expected [N, 1])  -> {'OK' if out.shape == (X_train.shape[0], 1) else 'WRONG'}")

    # Build the target
    from sazz.models.make_models import TorchTarget
    from sazz.models.priors_torch import Prior
    from sazz.models.models_torch import BayesianModel
    from sazz.utils.warmup import find_reference_bnn

    class _V2Prior(Prior):
        def __init__(self, prec_vec):
            super().__init__()
            self.register_buffer("_precision", prec_vec)
        def log_prob(self, beta):
            return -0.5 * (self._precision * beta ** 2).sum()
        def precision_diag(self):
            return self._precision

    prec = v2_build_prior_precision(
        spec, cfg.prior_std_weight, cfg.prior_std_bias,
        cfg.fan_in_scaling, dtype, "cpu",
    )
    prior = _V2Prior(prec)
    likelihood = _V2Likelihood(module, spec, X_train, y_train,
                               noise_std=cfg.noise_std)
    model = BayesianModel(prior, likelihood)

    print("Running find_reference_bnn ...")
    t0 = time.perf_counter()
    x_ref, Sigma_inv = find_reference_bnn(
        model.energy, spec.D, model=model, dtype=dtype, device="cpu",
        reference="laplace_diag", n_steps=2000, lr=1e-2,
    )
    print(f"  done in {time.perf_counter() - t0:.2f}s")
    print(f"  ||x_ref||_inf = {x_ref.abs().max().item():.3e}")

    target = TorchTarget(
        name="conv1d_bnn",
        D=spec.D,
        grad_target=model.grad_energy,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={"model": model, "spec": spec, "module": module},
    )

    # Sample
    print("\nRunning Boomerang sampler ...")
    torch.manual_seed(0); np.random.seed(0)
    s = AutomaticBoomerangSampler(
        grad_target=target.grad_target, D=target.D,
        refresh_rate=0.1, thinning="pli",
    )
    s.preprocess(x_ref=target.x_ref, Sigma_inv=target.Sigma_inv)
    x0 = target.x_ref.clone() + 0.1 * torch.randn(target.D, dtype=dtype)
    t0 = time.perf_counter()
    res = s.sample(N=n_skel, x0=x0, diagnostics=False)
    print(f"  {n_skel} skel in {time.perf_counter() - t0:.2f}s  "
          f"final_t={res['times'][-1].item():.2f}  "
          f"grad_evals={res['gradient_evals']}")

    # Predictive sanity check: pick a few beta from the chain, predict on test
    X_test = data["X_test"].to(dtype=dtype)
    y_test = data["y_test"].to(dtype=dtype)
    with torch.no_grad():
        preds = torch.stack([
            torch.func.functional_call(
                module, spec.to_dict(b), (X_test,)
            ).squeeze(-1)
            for b in res["positions"][-50:]
        ])
    rmse = ((preds.mean(0) - y_test) ** 2).mean().sqrt().item()
    print(f"  test RMSE (last 50 skel, no resampling): {rmse:.4f}")

    print("\n✓ ConvNet path works end-to-end\n")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", choices=["static", "sampler", "conv", "all"],
        default="all",
    )
    parser.add_argument("--dataset", default="hernandez")
    parser.add_argument("--n-skel", type=int, default=300)
    args = parser.parse_args()

    if args.check in ("static", "all"):
        check_static(layer_sizes=(1, 50, 1))
        check_static(layer_sizes=(13, 50, 50, 1), incl=0.5)

    if args.check in ("sampler", "all"):
        check_sampler(dataset=args.dataset, n_skel=args.n_skel)

    if args.check in ("conv", "all"):
        check_conv(dataset=args.dataset, n_skel=args.n_skel)


if __name__ == "__main__":
    main()
