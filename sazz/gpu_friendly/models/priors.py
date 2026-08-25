"""Priors as pure functions of a flat parameter vector `beta` (no module
state), so they compose with torch.func.grad/vmap/jvp."""

import math

import torch
import torch.nn as nn
from torch import Tensor


def build_fan_in_prior_precision(
    module: nn.Module,
    prior_std_weight: float,
    prior_std_bias: float,
    fan_in_scaling: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """
    Diagonal Gaussian prior precision for module.named_parameters(), in
    that order. Weights: sigma = prior_std_weight / sqrt(fan_in) if
    fan_in_scaling else prior_std_weight. Biases: sigma = prior_std_bias
    (never fan-in scaled). precision = 1 / sigma**2.
    """
    D = sum(p.numel() for p in module.parameters())
    prec = torch.empty(D, dtype=dtype, device=device)
    offset = 0
    for p in module.parameters():
        n = p.numel()
        if p.dim() == 1:
            sigma = prior_std_bias
        else:
            if fan_in_scaling:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(p)
                sigma = prior_std_weight / math.sqrt(fan_in)
            else:
                sigma = prior_std_weight
        prec[offset: offset + n] = 1.0 / sigma ** 2
        offset += n
    return prec


def build_kappa_from_inclusion(
    module: nn.Module,
    prior_std_weight: float,
    prior_inclusion_weight: float = 0.5,
    fan_in_scaling: bool = True,
    bias_thaw: float = 1e6,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """
    Per-coordinate thaw-rate kappa for GridStickyBoomerangSampler, matching
    a spike-and-slab prior beta_i ~ w*N(0, sigma_w^2) + (1-w)*delta_0:
        kappa_weight = (w / (1-w)) / (sigma_w * sqrt(2*pi))
    with sigma_w computed via the same fan-in convention as
    build_fan_in_prior_precision (so the slab here and the prior's slab
    are the same object by construction). Biases get `bias_thaw` (a large
    constant -> thaw almost immediately, i.e. effectively never sticky) --
    mirrors sazz.utils.bnn_utils.make_kappa_from_inclusion, reimplemented
    directly against module.named_parameters() instead of a ParamSpec.
    """
    if not (0.0 < prior_inclusion_weight < 1.0):
        raise ValueError(f"prior_inclusion_weight must be strictly in (0, 1); got {prior_inclusion_weight}.")

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    D = sum(p.numel() for p in module.parameters())
    kappa = torch.empty(D, dtype=dtype, device=device)
    offset = 0
    for p in module.parameters():
        n = p.numel()
        if p.dim() == 1:
            val = bias_thaw
        else:
            if fan_in_scaling:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(p)
                sigma_w = prior_std_weight / math.sqrt(fan_in)
            else:
                sigma_w = prior_std_weight
            val = (prior_inclusion_weight / (1.0 - prior_inclusion_weight)) / (sigma_w * sqrt_2pi)
        kappa[offset: offset + n] = val
        offset += n
    return kappa


def build_can_freeze_mask(module: nn.Module, device: torch.device | str = "cpu") -> Tensor:
    """Bool[D]: True for weight coordinates, False for biases -- biases never freeze."""
    D = sum(p.numel() for p in module.parameters())
    mask = torch.empty(D, dtype=torch.bool, device=device)
    offset = 0
    for p in module.parameters():
        n = p.numel()
        mask[offset: offset + n] = p.dim() != 1
        offset += n
    return mask


# ===========================================================================
# BatchNorm-aware variants -- additive only, the three functions above are
# never modified. A nn.BatchNorm{1,2,3}d's `weight` (gamma, a multiplicative
# scale centered near 1.0 at init) and `bias` (beta, additive, centered near
# 0.0) are BOTH shape [C], i.e. 1-D -- so the p.dim() == 1 heuristic above
# (correct for a real Linear/Conv bias) silently lumps BN gamma in with
# biases too. That's wrong on two counts: gamma isn't additive like a bias,
# and fan-in scaling (meant for >=2-D conv/linear weight tensors) doesn't
# apply to a 1-D tensor either, so gamma needs a third, explicit branch, not
# either existing one. These _resnet variants add that third branch; they
# are used only by the ResNet-20 pipeline (never by CNN/LeNet5, which have
# no BatchNorm submodules), and are separate functions specifically so the
# original three above stay byte-for-byte unchanged for every existing
# caller (fast_mnist_cnn.py, cnn_reference.py, lenet_reference.py, ...).
# ===========================================================================

def _is_batchnorm_weight(module: nn.Module, param_name: str) -> bool:
    """True if module.get_parameter(param_name) is a BatchNorm{1,2,3}d's
    `weight` (gamma) -- NOT its `bias` (beta), which stays bias-like."""
    if not param_name.endswith(".weight"):
        return False
    submodule_name = param_name[: -len(".weight")]
    submodule = module.get_submodule(submodule_name) if submodule_name else module
    return isinstance(submodule, nn.modules.batchnorm._BatchNorm)


def build_fan_in_prior_precision_resnet(
    module: nn.Module,
    prior_std_weight: float,
    prior_std_bias: float,
    prior_std_bn_weight: float = 1.0,
    fan_in_scaling: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """
    Like build_fan_in_prior_precision, with a third branch: BatchNorm gamma
    gets its own prior_std_bn_weight (default 1.0, matching gamma's typical
    init value) instead of fan-in scaling (meaningless for a 1-D tensor) or
    prior_std_bias (gamma is centered near 1.0, not 0 -- a zero-mean
    Gaussian prior here is a modeling compromise, flagged explicitly since
    there's no zero-centered alternative in this repo's prior machinery).
    BatchNorm bias (beta) is unaffected -- still prior_std_bias, same as any
    other 1-D bias.
    """
    D = sum(p.numel() for p in module.parameters())
    prec = torch.empty(D, dtype=dtype, device=device)
    offset = 0
    for name, p in module.named_parameters():
        n = p.numel()
        if _is_batchnorm_weight(module, name):
            sigma = prior_std_bn_weight
        elif p.dim() == 1:
            sigma = prior_std_bias
        else:
            if fan_in_scaling:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(p)
                sigma = prior_std_weight / math.sqrt(fan_in)
            else:
                sigma = prior_std_weight
        prec[offset: offset + n] = 1.0 / sigma ** 2
        offset += n
    return prec


def build_kappa_from_inclusion_resnet(
    module: nn.Module,
    prior_std_weight: float,
    prior_inclusion_weight: float = 0.5,
    fan_in_scaling: bool = True,
    bias_thaw: float = 1e6,
    bn_weight_thaw: float = 1e6,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """
    Like build_kappa_from_inclusion, with a third branch: BatchNorm gamma
    gets its own bn_weight_thaw (default 1e6, matching bias_thaw's "thaw
    almost immediately, effectively never sticky" convention -- see
    build_can_freeze_mask_resnet for why gamma is never eligible to freeze
    in the first place, so its kappa value is inert but kept explicit and
    separately named for clarity/future tuning rather than silently
    inheriting bias_thaw's value).
    """
    if not (0.0 < prior_inclusion_weight < 1.0):
        raise ValueError(f"prior_inclusion_weight must be strictly in (0, 1); got {prior_inclusion_weight}.")

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    D = sum(p.numel() for p in module.parameters())
    kappa = torch.empty(D, dtype=dtype, device=device)
    offset = 0
    for name, p in module.named_parameters():
        n = p.numel()
        if _is_batchnorm_weight(module, name):
            val = bn_weight_thaw
        elif p.dim() == 1:
            val = bias_thaw
        else:
            if fan_in_scaling:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(p)
                sigma_w = prior_std_weight / math.sqrt(fan_in)
            else:
                sigma_w = prior_std_weight
            val = (prior_inclusion_weight / (1.0 - prior_inclusion_weight)) / (sigma_w * sqrt_2pi)
        kappa[offset: offset + n] = val
        offset += n
    return kappa


def build_can_freeze_mask_resnet(module: nn.Module, device: torch.device | str = "cpu") -> Tensor:
    """
    Like build_can_freeze_mask, with BatchNorm gamma also marked False
    (non-freezable), same as biases. Freezing gamma to exactly 0 would zero
    that entire channel's post-normalization output -- a much more drastic
    structural change than freezing one Linear/Conv weight coordinate, and
    not a sparsity pattern the reference architecture (Wenzel et al. 2020 /
    Goan et al. 2023) was ever evaluated under. Deliberate choice, not a
    side effect of reusing the bias branch.
    """
    D = sum(p.numel() for p in module.parameters())
    mask = torch.empty(D, dtype=torch.bool, device=device)
    offset = 0
    for name, p in module.named_parameters():
        n = p.numel()
        if _is_batchnorm_weight(module, name):
            mask[offset: offset + n] = False
        else:
            mask[offset: offset + n] = p.dim() != 1
        offset += n
    return mask


def assert_eval_mode_if_batchnorm(module: nn.Module) -> None:
    """
    Raises if `module` contains any BatchNorm{1,2,3}d submodule AND is
    currently in train() mode. No-op for a module with no BatchNorm (e.g.
    CNN/LeNet5 -- never called by their scripts anyway). Intended call site:
    immediately before BayesianModule.build(...), in any script building a
    target from a BatchNorm-bearing module (e.g. ResNet20) -- converts
    "forgot to call module.eval() before sampling/MAP/Fisher-estimation" from
    a silent statistics bug (buffers mutating on every forward call) into an
    immediate, obvious error at construction time. See ResNet20's docstring
    in neural_networks.py and diagnose_batchnorm_eval.py for why eval-mode
    is required throughout this pipeline, not just at this one call site.
    """
    has_batchnorm = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in module.modules())
    if has_batchnorm and module.training:
        raise RuntimeError(
            "assert_eval_mode_if_batchnorm: module contains a BatchNorm submodule "
            "and is in train() mode. BayesianModule.build/functional_call reads "
            "BatchNorm's running_mean/running_var live off the module object -- "
            "train-mode BatchNorm recomputes batch statistics and mutates those "
            "buffers on every forward call, which silently corrupts the sticky "
            "PDMP sampler's fixed-rate-function invariant. Call module.eval() "
            "(after pretraining/populating real running stats) before proceeding."
        )


def make_gaussian_prior(precision: Tensor):
    """Zero-mean Gaussian prior. precision: [D] (diagonal) or [D, D] (dense)."""
    dense = precision.dim() == 2

    def log_prob(beta: Tensor) -> Tensor:
        if dense:
            return -0.5 * (beta @ precision @ beta)
        return -0.5 * (precision * beta ** 2).sum()

    return log_prob
