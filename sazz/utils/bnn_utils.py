"""
utils/bnn_utils.py

Lightweight utilities for functional BNNs.
No nn.Module — everything operates on flat parameter vectors.
"""

from typing import List
import torch
import math
from torch import Tensor
from typing import Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def get_activation(name: str):
    """
    Return a torch activation callable from a string name.
    Supported: "relu", "tanh", "sigmoid", "leaky_relu", "elu"
    """
    _activations = {
        "relu":       torch.relu,
        "tanh":       torch.tanh,
        "sigmoid":    torch.sigmoid,
        "leaky_relu": torch.nn.functional.leaky_relu,
        "elu":        torch.nn.functional.elu,
    }
    if name not in _activations:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {list(_activations)}"
        )
    return _activations[name]


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

def layer_shapes(layer_sizes: List[int]) -> List[tuple]:
    """
    Given [d_in, h1, h2, ..., d_out], return a list of (W_shape, b_shape)
    for each layer.

    Example: [2, 32, 1] -> [((32,2), (32,)), ((1,32), (1,))]
    """
    return [
        ((layer_sizes[i + 1], layer_sizes[i]), (layer_sizes[i + 1],))
        for i in range(len(layer_sizes) - 1)
    ]


def count_params(layer_sizes: List[int]) -> int:
    """Total number of scalar parameters for a fully-connected network."""
    return sum(
        out * inp + out
        for inp, out in zip(layer_sizes[:-1], layer_sizes[1:])
    )


def _build_prior_precision(
    layer_sizes: List[int],
    prior_std_weight: float,
    prior_std_bias: float,
    fan_in_scaling: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Build a per-coordinate prior precision vector matching the flatten
    convention in unflatten_params / layer_slices_from_sizes.

    Layout (per layer): weights first, biases second.
    """
    D = count_params(layer_sizes)
    prec = torch.empty(D, dtype=dtype, device=device)
    offset = 0

    for layer_idx, (W_shape, b_shape) in enumerate(layer_shapes(layer_sizes)):
        n_W = W_shape[0] * W_shape[1]
        n_b = b_shape[0]
        n_in = layer_sizes[layer_idx]   # fan-in for this layer

        if fan_in_scaling:
            sigma_w_l = prior_std_weight / (n_in ** 0.5)
        else:
            sigma_w_l = prior_std_weight

        prec[offset : offset + n_W] = 1.0 / sigma_w_l ** 2
        offset += n_W
        prec[offset : offset + n_b] = 1.0 / prior_std_bias ** 2
        offset += n_b

    return prec


# ---------------------------------------------------------------------------
# Flat vector <-> list of (W, b) tensors
# ---------------------------------------------------------------------------

def unflatten_params(
    beta: Tensor,
    layer_sizes: List[int],
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> List[tuple]:
    """
    Split a flat parameter vector into a list of (W, b) tuples, one per layer.

    Parameters
    ----------
    beta : Tensor [D]
        Flat parameter vector.
    layer_sizes : list of int
        e.g. [2, 32, 32, 1]

    Returns
    -------
    list of (W: Tensor [out, in], b: Tensor [out])
    """
    params = []
    idx = 0
    for (W_shape, b_shape) in layer_shapes(layer_sizes):
        W_size = W_shape[0] * W_shape[1]
        b_size = b_shape[0]
        W = beta[idx: idx + W_size].reshape(W_shape)
        idx += W_size
        b = beta[idx: idx + b_size]
        idx += b_size
        params.append((W, b))
    return params

# ---------------------------------------------------------------------------
# Kappa for weights and biases in different layers
# ---------------------------------------------------------------------------


def _broadcast_per_layer(val, n_layers: int, name: str) -> List[float]:
    if isinstance(val, (int, float)):
        return [float(val)] * n_layers
    val = list(val)
    if len(val) != n_layers:
        raise ValueError(
            f"{name} must be a scalar or a sequence of length "
            f"{n_layers} (got length {len(val)})."
        )
    return [float(v) for v in val]

def make_kappa_vector_bnn(
    layer_sizes: List[int],
    kappa_weights: Union[float, Sequence[float]] = 1.0,
    kappa_biases: Union[float, Sequence[float]] = 1e6,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Per-coordinate kappa vector for the sticky sampler.

    kappa_weights and kappa_biases can each be a scalar (same value applied
    to every layer) or a sequence of length len(layer_sizes) - 1 (one value
    per layer). Biases default to a very large kappa so they are effectively
    never stuck.
    """
    n_layers = len(layer_sizes) - 1
    kw = _broadcast_per_layer(kappa_weights, n_layers, name="kappa_weights")
    kb = _broadcast_per_layer(kappa_biases, n_layers, name="kappa_biases")

    kappas = []
    for layer_idx, (W_shape, b_shape) in enumerate(layer_shapes(layer_sizes)):
        n_weights = W_shape[0] * W_shape[1]
        n_biases = b_shape[0]
        kappas.append(
            torch.full((n_weights,), kw[layer_idx], dtype=dtype, device=device)
        )
        kappas.append(
            torch.full((n_biases,), kb[layer_idx], dtype=dtype, device=device)
        )
    return torch.cat(kappas)


def make_kappa_from_inclusion(
    layer_sizes: List[int],
    prior_std_weight: float,
    prior_inclusion_weight: Union[float, Sequence[float]] = 0.5,
    fan_in_scaling: bool = True,
    bias_thaw: float = 1e6,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Per-coordinate kappa vector for the sticky sampler, derived from a
    spike-and-slab prior with Gaussian slabs.

    For a spike-and-slab prior on weight i,
        beta_i ~ w_i * delta_0 + (1 - w_i) * N(0, sigma_w_i^2),
    the correct stickiness parameter is

        kappa_w_i = (w_i / (1 - w_i)) * (1 / (sigma_w_i * sqrt(2*pi))).

    Larger kappa = larger thaw rate = shorter freeze time = LESS sticky.

    The slab std `sigma_w_l` is taken from `prior_std_weight` with the
    same optional fan-in scaling used by `_build_prior_precision`, so the
    slab in the prior and the slab in this formula are the *same object*
    by construction.

    Biases use a separate mechanism: there is typically no reason to
    place a Dirac spike at zero on a bias, so they are kept effectively
    never-sticky via `bias_thaw` (a numerical stand-in for kappa -> inf).
    This is NOT the formula above evaluated at w_b = 1 — that would be
    infinite — it is a separate modelling choice.

    Parameters
    ----------
    layer_sizes : list of int
        Network architecture, e.g. [d_in, h1, h2, d_out].
    prior_std_weight : float
        Slab std for weights. Must match what was passed to
        BNNGaussianPrior.
    prior_inclusion_weight : float or sequence of float
        Prior probability that a weight is non-zero, w_l. Either a
        scalar (same w across layers) or a sequence of length
        len(layer_sizes) - 1 (one w per layer). Must be in (0, 1).
        Default 0.5 = "no sparsity opinion".
    fan_in_scaling : bool
        If True, sigma_w_l = prior_std_weight / sqrt(d_in_l). Must
        match the value used in BNNGaussianPrior.
    bias_thaw : float
        Thaw rate used for bias coordinates. Large values keep biases
        effectively always active. Default 1e6.

    Returns
    -------
    Tensor [D]
        Flat kappa vector aligned with the (W_1, b_1, W_2, b_2, ...)
        layout used by `unflatten_params`.
    """
    n_layers = len(layer_sizes) - 1
    w = _broadcast_per_layer(
        prior_inclusion_weight, n_layers, name="prior_inclusion_weight"
    )
    for w_l in w:
        if not (0.0 < w_l < 1.0):
            raise ValueError(
                f"prior_inclusion_weight must be strictly in (0, 1); got {w_l}."
            )

    sqrt_2pi = math.sqrt(2.0 * math.pi)

    kappas = []
    for layer_idx, (W_shape, b_shape) in enumerate(layer_shapes(layer_sizes)):
        n_weights = W_shape[0] * W_shape[1]
        n_biases = b_shape[0]
        n_in = layer_sizes[layer_idx]

        sigma_w_l = (
            prior_std_weight / (n_in ** 0.5) if fan_in_scaling else prior_std_weight
        )

        w_l = w[layer_idx]
        kappa_w_l = (w_l / (1.0 - w_l)) * (1.0 / (sigma_w_l * sqrt_2pi))

        kappas.append(
            torch.full((n_weights,), kappa_w_l, dtype=dtype, device=device)
        )
        kappas.append(
            torch.full((n_biases,), bias_thaw, dtype=dtype, device=device)
        )

    return torch.cat(kappas)

# ---------------------------------------------------------------------------
# Find reference BNNs
# ---------------------------------------------------------------------------


# def layer_slices_from_sizes(layer_sizes: Sequence[int]) -> list[slice]:
#     """
#     Helper: build layer_slices for a standard FFN with weights then biases
#     per layer, matching the flattening convention used by make_kappa_vector.

#     Returns one slice per parameter group (weights of layer 1, biases of
#     layer 1, weights of layer 2, ...).
#     """
#     slices = []
#     offset = 0
#     for i in range(len(layer_sizes) - 1):
#         n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
#         n_W = n_in * n_out
#         n_b = n_out
#         slices.append(slice(offset, offset + n_W))
#         offset += n_W
#         slices.append(slice(offset, offset + n_b))
#         offset += n_b
#     return slices
