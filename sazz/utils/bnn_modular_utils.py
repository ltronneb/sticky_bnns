"""
utils/bnn_utils_v2.py

Module-aware utilities for BNNs. Replaces the layer_sizes-driven helpers in
bnn_utils.py with a single ParamSpec abstraction that is built from any
nn.Module.

The flatten convention is determined entirely by module.named_parameters()
iteration order. For a standard FFN built as nn.Sequential of nn.Linear
layers, this produces (weight, bias, weight, bias, ...), which matches
the existing layer_sizes-driven order bit-for-bit, so the v2 builders can
substitute for the v1 ones without numerical change.

Nothing in this file knows about layer_sizes. Everything talks ParamSpec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# ParamSpec — the canonical description of a flattened module
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    """
    Static description of how a flat beta vector maps to nn.Module parameters.

    Built once, at construction time. Per-iteration calls to `to_dict(beta)`
    produce a {name: tensor} dict using only torch.Tensor.view, which is
    O(num_params) Python work and zero data copies.

    Fields
    ------
    names    : ordered list of named_parameters() keys
    shapes   : torch.Size for each parameter tensor
    numels   : numel() for each parameter tensor
    fan_ins  : per-parameter fan-in (used by prior_std / kappa formulae)
    is_bias  : True if the parameter is treated as a "bias" — by convention
               1D tensors. This affects which prior std and which kappa
               formula are applied to the corresponding flat slice.
    D        : sum of numels — total parameter count
    """
    names: list[str]
    shapes: list[torch.Size]
    numels: list[int]
    fan_ins: list[int]
    is_bias: list[bool]
    can_freeze: list[bool] 
    D: int

    # --- Construction -----------------------------------------------------

    @classmethod
    def from_module(cls, module: nn.Module) -> "ParamSpec":
        """
        Build a ParamSpec by walking module.named_parameters().

        Convention: a parameter with .dim() == 1 is treated as a bias. This
        captures Linear.bias, Conv*.bias, embedding-free LayerNorm.bias, etc.
        2D+ parameters (Linear.weight, Conv*.weight, ...) are treated as
        "weights" and get fan-in scaling.

        Fan-in is computed via torch.nn.init._calculate_fan_in_and_fan_out
        for parameters with .dim() >= 2. This matches the same convention
        used by Kaiming/Glorot initialisation, so a Conv2d's fan-in is
        C_in * kH * kW. For 1D tensors (biases) fan_in is set to 1; the
        value is unused because biases get prior_std_bias.
        """
        names, shapes, numels, fan_ins, is_bias, can_freeze = [], [], [], [], [], []
        for name, p in module.named_parameters():
            names.append(name)
            shapes.append(p.shape)
            numels.append(p.numel())
            is_b = (p.dim() == 1)
            is_bias.append(is_b)
            cf = getattr(p, "can_freeze", not is_b)
            can_freeze.append(cf)
            if is_b:
                fan_ins.append(1)  # placeholder, never consumed
            else:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(p)
                fan_ins.append(int(fan_in))
        return cls(
            names=names,
            shapes=shapes,
            numels=numels,
            fan_ins=fan_ins,
            is_bias=is_bias,
            can_freeze=can_freeze,
            D=sum(numels),
        )

    # --- Hot path ---------------------------------------------------------

    def to_dict(self, beta: Tensor) -> dict[str, Tensor]:
        """
        View the flat beta as a {name: tensor} dict suitable for
        torch.func.functional_call(module, dict, args).

        Pure views — no data copies, autograd graph is preserved.
        """
        out: dict[str, Tensor] = {}
        idx = 0
        for name, shape, n in zip(self.names, self.shapes, self.numels):
            out[name] = beta[idx: idx + n].view(shape)
            idx += n
        return out

    # --- Utilities --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.names)

    @property
    def n_groups(self) -> int:
        """Number of distinct parameter tensors (== len(named_parameters))."""
        return len(self.names)
    
    def with_extra_scalar(self, name: str, can_freeze: bool = False) -> "ParamSpec":
        """Append a scalar latent (e.g. log_sigma) to this spec.
        
        The scalar is treated as a 1D tensor of size 1, gets the bias prior
        (prior_std_bias), and defaults to can_freeze=False since most uses
        (noise scale, observation precision) shouldn't go through the freeze
        machinery.
        """
        return ParamSpec(
            names=self.names + [name],
            shapes=self.shapes + [torch.Size([1])],
            numels=self.numels + [1],
            fan_ins=self.fan_ins + [1],
            is_bias=self.is_bias + [True],
            can_freeze=self.can_freeze + [can_freeze],
            D=self.D + 1,
        )


# ---------------------------------------------------------------------------
# Broadcast helpers
# ---------------------------------------------------------------------------

def _broadcast_per_group(
    val: Union[float, Sequence[float], dict],
    spec: ParamSpec,
    name_for_error: str,
    selector: Optional[list[bool]] = None,
) -> list[float]:
    """
    Broadcast a scalar / sequence / dict of values to one float per
    *parameter group* in `spec`.

    selector:
        Optional boolean mask of length spec.n_groups. If given, only the
        groups where selector[i] is True contribute to the length-checking
        for sequences. The returned list still has length spec.n_groups,
        with values for masked-out groups set to NaN (they should never
        be consumed).

    Accepted forms for `val`:
      * scalar           -> applied to every group in `spec`
      * sequence         -> length must equal the number of selected groups
      * dict[name -> v]  -> looked up by spec.names; selected groups not
                            present in the dict raise KeyError.
    """
    n_total = spec.n_groups
    if selector is None:
        selector = [True] * n_total
    n_selected = sum(selector)

    if isinstance(val, (int, float)):
        return [float(val) if sel else float("nan")
                for sel in selector]

    if isinstance(val, dict):
        out: list[float] = []
        for nm, sel in zip(spec.names, selector):
            if not sel:
                out.append(float("nan"))
                continue
            if nm not in val:
                raise KeyError(
                    f"{name_for_error} dict is missing entry for "
                    f"parameter '{nm}'."
                )
            out.append(float(val[nm]))
        return out

    seq = list(val)
    if len(seq) != n_selected:
        raise ValueError(
            f"{name_for_error} must be a scalar, a dict keyed by parameter "
            f"name, or a sequence of length {n_selected} (one per "
            f"selected group); got length {len(seq)}."
        )
    out: list[float] = []
    j = 0
    for sel in selector:
        if sel:
            out.append(float(seq[j]))
            j += 1
        else:
            out.append(float("nan"))
    return out


# ---------------------------------------------------------------------------
# Prior precision builder
# ---------------------------------------------------------------------------

def build_prior_precision(
    spec: ParamSpec,
    prior_std_weight: float,
    prior_std_bias: float,
    fan_in_scaling: bool = True,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu",
) -> Tensor:
    """
    Per-coordinate diagonal precision vector of length spec.D, matching the
    flatten layout in spec.

    Weights:
        sigma_l = prior_std_weight / sqrt(fan_in_l)   if fan_in_scaling
                = prior_std_weight                    otherwise
        prec    = 1 / sigma_l ** 2

    Biases:
        sigma   = prior_std_bias
        prec    = 1 / sigma ** 2
    """
    device = torch.device(device)
    prec = torch.empty(spec.D, dtype=dtype, device=device)
    offset = 0
    for numel, fan_in, is_b in zip(spec.numels, spec.fan_ins, spec.is_bias):
        if is_b:
            sigma = prior_std_bias
        else:
            sigma = (prior_std_weight / math.sqrt(fan_in)
                     if fan_in_scaling else prior_std_weight)
        prec[offset: offset + numel] = 1.0 / sigma ** 2
        offset += numel
    return prec


# ---------------------------------------------------------------------------
# Kappa builders
# ---------------------------------------------------------------------------

def make_kappa_vector(
    spec: ParamSpec,
    kappa_weights: Union[float, Sequence[float], dict] = 1.0,
    kappa_biases: Union[float, Sequence[float], dict] = 1e6,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu",
) -> Tensor:
    """
    Per-coordinate kappa vector for the sticky sampler.

    kappa_weights / kappa_biases each accept:
      * scalar             — same value for every weight / bias group
      * sequence           — one value per weight (resp. bias) group, in
                             the order they appear in spec
      * dict[name, value]  — keyed by spec.names; only weight (resp. bias)
                             groups need entries

    Larger kappa = larger thaw rate = LESS sticky.
    Biases default to a very large kappa so they are effectively never
    frozen.
    """
    device = torch.device(device)

    weight_mask = [not b for b in spec.is_bias]
    bias_mask = list(spec.is_bias)

    kw = _broadcast_per_group(
        kappa_weights, spec, "kappa_weights", selector=weight_mask
    )
    kb = _broadcast_per_group(
        kappa_biases, spec, "kappa_biases", selector=bias_mask
    )

    kappa = torch.empty(spec.D, dtype=dtype, device=device)
    offset = 0
    for numel, is_b, kw_i, kb_i in zip(spec.numels, spec.is_bias, kw, kb):
        val = kb_i if is_b else kw_i
        kappa[offset: offset + numel] = val
        offset += numel
    return kappa


def make_kappa_from_inclusion(
    spec: ParamSpec,
    prior_std_weight: float,
    prior_inclusion_weight: Union[float, Sequence[float], dict] = 0.5,
    fan_in_scaling: bool = True,
    bias_thaw: float = 1e6,
    dtype: torch.dtype = torch.float64,
    device: Union[torch.device, str] = "cpu",
) -> Tensor:
    """
    Per-coordinate kappa vector derived from a spike-and-slab prior with
    Gaussian slabs.

    For weight i with prior   beta_i ~ w * delta_0 + (1 - w) * N(0, sigma_w^2),
    the matching stickiness parameter is

        kappa_w = (w / (1 - w)) * (1 / (sigma_w * sqrt(2*pi))).

    The slab std `sigma_w` is computed from prior_std_weight using the same
    fan-in convention as build_prior_precision (so the slab in the prior
    and the slab in this formula are the same object by construction).

    Biases use the separate `bias_thaw` constant — there is typically no
    reason to put a Dirac spike at zero on a bias.

    `prior_inclusion_weight` accepts the same forms as kappa_weights in
    make_kappa_vector: scalar, sequence (one per weight group, in spec
    order), or dict keyed by spec.names.
    """
    device = torch.device(device)

    weight_mask = [not b for b in spec.is_bias]
    w = _broadcast_per_group(
        prior_inclusion_weight, spec,
        "prior_inclusion_weight", selector=weight_mask,
    )
    for w_l, sel in zip(w, weight_mask):
        if sel and not (0.0 < w_l < 1.0):
            raise ValueError(
                f"prior_inclusion_weight must be strictly in (0, 1); "
                f"got {w_l}."
            )

    sqrt_2pi = math.sqrt(2.0 * math.pi)

    kappa = torch.empty(spec.D, dtype=dtype, device=device)
    offset = 0
    for numel, fan_in, is_b, w_l in zip(
        spec.numels, spec.fan_ins, spec.is_bias, w
    ):
        if is_b:
            val = bias_thaw
        else:
            sigma_w = (prior_std_weight / math.sqrt(fan_in)
                       if fan_in_scaling else prior_std_weight)
            val = (w_l / (1.0 - w_l)) * (1.0 / (sigma_w * sqrt_2pi))
        kappa[offset: offset + numel] = val
        offset += numel
    return kappa

