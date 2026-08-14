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
    a spike-and-slab prior beta_i ~ w*delta_0 + (1-w)*N(0, sigma_w^2):
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


def make_gaussian_prior(precision: Tensor):
    """Zero-mean Gaussian prior. precision: [D] (diagonal) or [D, D] (dense)."""
    dense = precision.dim() == 2

    def log_prob(beta: Tensor) -> Tensor:
        if dense:
            return -0.5 * (beta @ precision @ beta)
        return -0.5 * (precision * beta ** 2).sum()

    return log_prob
