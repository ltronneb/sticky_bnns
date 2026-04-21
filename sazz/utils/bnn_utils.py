"""
utils/bnn_utils.py

Lightweight utilities for functional BNNs.
No nn.Module — everything operates on flat parameter vectors.
"""

from typing import List
import torch
from torch import Tensor


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


def flatten_params(params: List[tuple]) -> Tensor:
    """
    Concatenate a list of (W, b) tuples into a single flat vector.
    Inverse of unflatten_params.
    """
    parts = []
    for W, b in params:
        parts.append(W.reshape(-1))
        parts.append(b.reshape(-1))
    return torch.cat(parts)

# ---------------------------------------------------------------------------
# Freeze only weights
# ---------------------------------------------------------------------------
def make_kappa_vector(
    layer_sizes: List[int],
    kappa_weights: float = 1.0,
    kappa_biases: float = 1e6,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    kappas = []
    for (W_shape, b_shape) in layer_shapes(layer_sizes):
        n_weights = W_shape[0] * W_shape[1]
        n_biases  = b_shape[0]
        kappas.append(torch.full((n_weights,), kappa_weights, dtype=dtype, device=device))
        kappas.append(torch.full((n_biases,),  kappa_biases,  dtype=dtype, device=device))
    return torch.cat(kappas)
