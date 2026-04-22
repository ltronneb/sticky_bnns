"""
utils/bnn_utils.py

Lightweight utilities for functional BNNs.
No nn.Module — everything operates on flat parameter vectors.
"""

from typing import List
import torch
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
# Find reference BNNs
# ---------------------------------------------------------------------------

def find_reference_bnn(
    energy_fn,
    D: int,
    layer_slices: Optional[Sequence[slice]] = None,
    prior_precision: Optional[Tensor] = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    n_steps: int = 2000,
    lr: float = 1e-2,
    n_samples_fisher: int = 50,
    #prec_min: float = 1e-3,
    #prec_max: float = 1e4,
    per_layer_normalise: bool = False,
    per_layer_clip: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Adam-based MAP + diagonal empirical Fisher, with optional per-layer
    normalisation and clipping.

    Parameters
    ----------
    layer_slices : list of slice, optional
        Index ranges for each parameter group (e.g. [slice(0, 784*32),
        slice(784*32, 784*32+32), ...] for a weight/bias split per layer).
        If None, falls back to a single global group (old behaviour).
    prior_precision : Tensor [D], optional
        Per-coordinate prior precision (1 / prior_std^2). If provided,
        used as the per-coordinate floor on the diagonal precision,
        so posterior precision is never below prior precision.
    per_layer_normalise : bool
        If True, divide each layer's precisions by the layer median before
        clipping, so clip bounds are interpreted relative to each layer's
        typical curvature rather than absolute.
    per_layer_clip : bool
        If True, clip after normalisation (if used). If False, only the
        normalisation is applied and no clipping is done.
    """
    # --- MAP via Adam ---
    beta = torch.randn(D, dtype=dtype, device=device) * 0.01
    beta.requires_grad_(True)
    optimizer = torch.optim.Adam([beta], lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = energy_fn(beta)
        loss.backward()
        optimizer.step()
    x_ref = beta.detach().clone()

    # --- Empirical Fisher diagonal ---
    grad_sq = torch.zeros(D, dtype=dtype, device=device)
    for _ in range(max(n_samples_fisher, 1)):
        b = x_ref.clone().requires_grad_(True)
        E = energy_fn(b)
        g, = torch.autograd.grad(E, b)
        grad_sq += g ** 2
    grad_sq /= max(n_samples_fisher, 1)

    # --- Default to single global group if no slices given ---
    if layer_slices is None:
        layer_slices = [slice(0, D)]

    diag_prec = grad_sq.clone()

    # --- Per-layer processing ---
    for sl in layer_slices:
        block = diag_prec[sl]
        if block.numel() == 0:
            continue
        med = block.median().clamp(min=1e-12)
        if per_layer_normalise:
            # Median-based normalisation is robust to outlier gradients
            #med = block.median().clamp(min=1e-12)
            block = block / med
            # After normalisation, the block is dimensionless and centred on 1

        if per_layer_clip:
            if prior_precision is not None:
                floor = prior_precision[sl].to(block.dtype).to(block.device)
                block = torch.maximum(block, floor).clamp(max=1e5)
            else:
                block = block.clamp(min=1.0, max=1e5)

        diag_prec[sl] = block

    Sigma_inv = torch.diag(diag_prec)
    return x_ref, Sigma_inv


def layer_slices_from_sizes(layer_sizes: Sequence[int]) -> list[slice]:
    """
    Helper: build layer_slices for a standard FFN with weights then biases
    per layer, matching the flattening convention used by make_kappa_vector.

    Returns one slice per parameter group (weights of layer 1, biases of
    layer 1, weights of layer 2, ...).
    """
    slices = []
    offset = 0
    for i in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        n_W = n_in * n_out
        n_b = n_out
        slices.append(slice(offset, offset + n_W))
        offset += n_W
        slices.append(slice(offset, offset + n_b))
        offset += n_b
    return slices

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