"""
Phase 0 of the CIFAR-10/ResNet-20 plan: does this repo's torch.func pipeline
(BayesianModule.build -> functional_call -> torch.func.grad/vmap/jvp, and the
sticky PDMP samplers built on top of it) actually work with a frozen,
eval-mode BatchNorm layer sitting inside an otherwise-normal nn.Module?

Answered here in complete isolation from CIFAR-10/ResNet-20 -- a throwaway
toy module (one BasicBlock-shaped residual unit: Conv2d(bias=False) ->
BatchNorm2d -> ReLU -> Conv2d(bias=False) -> BatchNorm2d -> residual-add ->
ReLU -> AdaptiveAvgPool2d -> Linear) on tiny synthetic data, CPU, seconds to
run. Nothing here is imported by or shared with the real ResNet-20 pipeline
built in later phases -- this script's only job is to gate whether that
pipeline is worth building at all.

Load-bearing design decision this script exists to validate: BatchNorm runs
in train() only during an ordinary (non-Bayesian) pretrain pass; the module
is then switched to eval() and NEVER switched back for the rest of its
lifetime (MAP refinement, Fisher-diagonal estimation, PDMP sampling all run
against an eval()-mode module). BN's running_mean/running_var become fixed
data, not sampled parameters -- BayesianModule.build only ever flattens
module.named_parameters() into beta (confirmed by reading model.py), so
buffers are structurally outside beta and are read live off the module
object by functional_call. This is a deliberate, permanent divergence from
"BatchNorm adapts as beta moves through the posterior," accepted as the cost
of keeping the sticky sampler's grid-thinning bound valid: grad_target must
be a FIXED function of x for an entire _grid_bound episode (it's evaluated
both eagerly and inside torch.func.vmap/jvp over a batch of candidate grid
times within that one episode), and train-mode BatchNorm -- which recomputes
batch statistics and mutates its running buffers on every call -- would
violate that invariant.

Run with:
    python -m sazz.gpu_friendly.scripts.diagnose_batchnorm_eval
"""

import torch
import torch.nn as nn
from torch import Tensor

from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.priors import (
    build_can_freeze_mask, build_kappa_from_inclusion,
    build_can_freeze_mask_resnet, build_kappa_from_inclusion_resnet,
)
from sazz.gpu_friendly.samplers.fast_grid_sticky_zigzag import FastGridStickyZigZagSampler

DTYPE = torch.float64
DEVICE = "cpu"


class ToyBasicBlock(nn.Module):
    """
    Minimal stand-in for one ResNet-20 BasicBlock: bias-free conv -> BN ->
    activation, twice, with a residual add before the final activation.
    Throwaway -- not reused by any later phase's real ResNet20 class.
    """
    def __init__(self, channels: int = 4, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = torch.relu(h + identity)
        h = self.pool(h).flatten(1)
        return self.fc(h)


def _make_toy_bm(module: nn.Module, X: Tensor, y: Tensor) -> BayesianModule:
    return BayesianModule.build(
        module, likelihood="categorical", X=X, y=y,
        prior_precision=1.0, dtype=DTYPE, device=DEVICE,
    )


def check_buffer_mutation(module: ToyBasicBlock, X: Tensor, y: Tensor) -> bool:
    print("\n[1] buffer mutation: train() mutates running stats, eval() does not")
    bm = _make_toy_bm(module, X, y)
    beta0 = torch.cat([p.detach().flatten() for p in module.parameters()]).clone()

    module.train()
    rm_before = module.bn1.running_mean.clone()
    bm.energy(beta0)
    rm_after_train = module.bn1.running_mean.clone()
    mutated_in_train = not torch.equal(rm_before, rm_after_train)
    print(f"    train(): running_mean changed = {mutated_in_train} (expected True)")

    module.eval()
    rm_before_eval = module.bn1.running_mean.clone()
    for _ in range(5):
        bm.energy(beta0)
        torch.func.grad(bm.energy)(beta0)
    rm_after_eval = module.bn1.running_mean.clone()
    unmutated_in_eval = torch.equal(rm_before_eval, rm_after_eval)
    print(f"    eval(): running_mean unchanged over 5 calls = {unmutated_in_eval} (expected True)")

    ok = mutated_in_train and unmutated_in_eval
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_vmap_grad(module: ToyBasicBlock, X: Tensor, y: Tensor) -> bool:
    print("\n[2] vmap(grad(...)) over batch-of-1 lanes (mirrors empirical_fisher_diag_vmapped)")
    module.eval()
    bm = _make_toy_bm(module, X, y)
    beta0 = torch.cat([p.detach().flatten() for p in module.parameters()]).clone()

    def per_point_log_prob(beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        return bm.log_likelihood.single(beta, X_i, y_i)

    X_lanes = X.unsqueeze(1)  # [n, 1, C, H, W]
    y_lanes = y.unsqueeze(1)  # [n, 1]

    try:
        grads = torch.func.vmap(
            torch.func.grad(per_point_log_prob), in_dims=(None, 0, 0),
        )(beta0, X_lanes, y_lanes)
        no_nan = not torch.isnan(grads).any().item()
        right_shape = grads.shape == (X.shape[0], beta0.numel())
        ok = no_nan and right_shape
        print(f"    output shape={tuple(grads.shape)} (expected {(X.shape[0], beta0.numel())}), "
              f"no NaNs={no_nan}")
    except Exception as e:
        print(f"    raised: {e!r}")
        ok = False
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_jvp_grid_bound_composition(module: ToyBasicBlock, X: Tensor, y: Tensor) -> bool:
    print("\n[3] vmap(jvp(...)) composition (mirrors fast_grid_bound.py's rate_and_grad_fn contract)")
    module.eval()
    bm = _make_toy_bm(module, X, y)
    D = bm.D
    beta0 = torch.cat([p.detach().flatten() for p in module.parameters()]).clone()
    v = torch.randn(D, dtype=DTYPE, device=DEVICE)
    grad_target = torch.func.grad(bm.energy)

    def g_scalar(t: Tensor) -> Tensor:
        x_t = beta0 + t * v
        return torch.dot(v, grad_target(x_t))

    def rate_and_grad_fn(t_batch: Tensor):
        vmap_fn = torch.func.vmap(
            lambda ti: torch.func.jvp(g_scalar, (ti,), (torch.ones_like(ti),)),
        )
        return vmap_fn(t_batch)

    t_batch = torch.linspace(0.0, 1.0, 8, dtype=DTYPE, device=DEVICE)
    try:
        y_out1, dy_out1 = rate_and_grad_fn(t_batch)
        y_out2, dy_out2 = rate_and_grad_fn(t_batch)
        no_nan = not (torch.isnan(y_out1).any() or torch.isnan(dy_out1).any()).item()
        deterministic = torch.equal(y_out1, y_out2) and torch.equal(dy_out1, dy_out2)
        ok = no_nan and deterministic
        print(f"    no NaNs={no_nan}, deterministic across repeated calls at same t={deterministic}")
    except Exception as e:
        print(f"    raised: {e!r}")
        ok = False
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_can_freeze_kappa_misclassification(module: ToyBasicBlock) -> bool:
    print("\n[4] can_freeze/kappa: original (unmodified) vs new _resnet builders, side-by-side")
    can_freeze_orig = build_can_freeze_mask(module, device=DEVICE)
    kappa_orig = build_kappa_from_inclusion(module, prior_std_weight=2.0, prior_inclusion_weight=0.05)
    can_freeze_new = build_can_freeze_mask_resnet(module, device=DEVICE)
    kappa_new = build_kappa_from_inclusion_resnet(module, prior_std_weight=2.0, prior_inclusion_weight=0.05)

    offset = 0
    ok = True
    for name, p in module.named_parameters():
        n = p.numel()
        is_bn_weight = name.endswith("bn1.weight") or name.endswith("bn2.weight")
        if is_bn_weight:
            orig_freeze = bool(can_freeze_orig[offset].item())
            new_freeze = bool(can_freeze_new[offset].item())
            orig_kappa = float(kappa_orig[offset].item())
            new_kappa = float(kappa_new[offset].item())
            print(f"    {name}: original priors.py (unchanged) -> can_freeze={orig_freeze}, "
                  f"kappa={orig_kappa:.4g} (correct by accident: dim()==1 already lumps it with "
                  f"bias, which happens to give a plausible-looking non-freezable/high-kappa "
                  f"result here even though it's semantically wrong -- gamma is not a bias)")
            print(f"    {name}: new _resnet builder -> can_freeze={new_freeze} (expect False), "
                  f"kappa={new_kappa:.4g} (expect 1e6, explicitly via bn_weight_thaw, not "
                  f"silently inherited from the bias branch)")
            ok = ok and (new_freeze is False)
        offset += n
    print(f"    -> {'PASS' if ok else 'FAIL'} (new builders correctly mark BN gamma non-freezable; "
          f"original priors.py functions confirmed untouched -- still importable, still callable)")
    return ok


def check_end_to_end_sampler(module: ToyBasicBlock, X: Tensor, y: Tensor) -> bool:
    print("\n[5] end-to-end micro sticky-ZigZag sampler run under eval-mode BatchNorm")
    module.eval()
    bm = _make_toy_bm(module, X, y)
    D = bm.D
    grad_target = torch.func.grad(bm.energy)

    sampler = FastGridStickyZigZagSampler(
        grad_target=grad_target, D=D, gamma=1e-4, grid_t_max_init=0.05,
        n_segments=20, grid_spacing=0.005, dtype=DTYPE, device=DEVICE,
    )
    x0 = torch.randn(D, dtype=DTYPE, device=DEVICE) * 0.1

    rm1_before = module.bn1.running_mean.clone()
    rm2_before = module.bn2.running_mean.clone()
    try:
        result = sampler.sample(N=50, x0=x0, diagnostics=False)
        no_nan = not torch.isnan(result["positions"]).any().item()
        finite_violations = math_isfinite(result["bound_violations"])
        rm1_after = module.bn1.running_mean.clone()
        rm2_after = module.bn2.running_mean.clone()
        buffers_frozen = torch.equal(rm1_before, rm1_after) and torch.equal(rm2_before, rm2_after)
        ok = no_nan and finite_violations and buffers_frozen
        print(f"    sampled {result['positions'].shape[0]} rows, no NaNs={no_nan}, "
              f"bound_violations={result['bound_violations']} (finite={finite_violations}), "
              f"BN buffers frozen throughout={buffers_frozen}")
    except Exception as e:
        print(f"    raised: {e!r}")
        ok = False
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def math_isfinite(v) -> bool:
    import math
    return math.isfinite(v)


def main():
    torch.manual_seed(0)
    channels, n_classes, hw, n = 4, 2, 8, 12
    X = torch.randn(n, channels, hw, hw, dtype=DTYPE, device=DEVICE)
    y = torch.randint(0, n_classes, (n,), device=DEVICE)

    print("Phase 0 diagnostic: BatchNorm eval-mode mechanics under this repo's torch.func pipeline")
    print(f"Toy module: ToyBasicBlock(channels={channels}, n_classes={n_classes}), "
          f"data X={tuple(X.shape)} y={tuple(y.shape)}, dtype={DTYPE}, device={DEVICE}")

    module = ToyBasicBlock(channels=channels, n_classes=n_classes).to(dtype=DTYPE, device=DEVICE)

    r1 = check_buffer_mutation(module, X, y)
    r2 = check_vmap_grad(module, X, y)
    r3 = check_jvp_grid_bound_composition(module, X, y)
    r4 = check_can_freeze_kappa_misclassification(module)
    r5 = check_end_to_end_sampler(module, X, y)

    print("\n" + "=" * 70)
    gate0_checks = {"[1] buffer mutation": r1, "[2] vmap+grad": r2,
                     "[3] jvp+grid-bound": r3, "[5] end-to-end sampler": r5}
    gate0_pass = all(gate0_checks.values())
    for name, ok in gate0_checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nGate 0: {'PASS -- safe to proceed to Phase 1' if gate0_pass else 'FAIL -- STOP, re-open the BatchNorm design decision before writing ResNet-20/CIFAR code'}")

    print(f"\nGate 1 (priors.py _resnet builders): {'PASS' if r4 else 'FAIL'}")


if __name__ == "__main__":
    main()
