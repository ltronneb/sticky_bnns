"""Target = prior * likelihood, exposed as a pure function `energy(beta)` --
no detaching, no module-state -- so it composes with torch.func.grad/vmap/jvp.
(sazz/models/model.py's BayesianModel always detaches before differentiating,
which would sever the graph from a grid time `t` through to `beta`.)"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .priors import make_gaussian_prior
from .likelihoods import (
    make_gaussian_likelihood,
    make_bernoulli_likelihood,
    make_categorical_likelihood,
)


def make_energy(log_prior: Callable[[Tensor], Tensor],
                 log_likelihood: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Combine prior + likelihood log-densities into energy(beta) = -log_posterior(beta)."""
    def energy(beta: Tensor) -> Tensor:
        return -(log_prior(beta) + log_likelihood(beta))

    return energy


def to_param_dict(names: list[str], shapes: list[torch.Size]) -> Callable[[Tensor], dict]:
    """
    Build a beta -> {name: tensor} unflattening closure for functional_call.
    Safe under vmap/jvp as long as `beta` is never the batched argument --
    every sampler here only ever batches over grid time, never over beta.
    """
    numels = [int(torch.Size(s).numel()) for s in shapes]

    def param_dict_fn(beta: Tensor) -> dict:
        out = {}
        idx = 0
        for name, shape, n in zip(names, shapes, numels):
            out[name] = beta[idx: idx + n].view(shape)
            idx += n
        return out

    return param_dict_fn


@dataclass
class BayesianModule:
    """
    Target = prior * likelihood built from an nn.Module written the normal
    torch way (subclass nn.Module, define forward()). Flattens its
    parameters into a single beta vector and exposes a pure,
    torch.func-composable `energy(beta)`.

    Usage
    -----
        class MyNet(nn.Module):
            def __init__(self, d_in, d_hidden):
                super().__init__()
                self.fc1 = nn.Linear(d_in, d_hidden)
                self.fc2 = nn.Linear(d_hidden, 1)
            def forward(self, x):
                return self.fc2(torch.tanh(self.fc1(x))).squeeze(-1)

        module = MyNet(d_in=8, d_hidden=16).double()

        # Default: the observation noise is LEARNED (one extra `log_sigma`
        # coordinate appended to beta, HalfNormal(1.0) prior on sigma).
        bm = BayesianModule.build(module, likelihood="gaussian", X=X, y=y)
        energy = bm.energy
        D = bm.D              # == network params + 1

        # Override: fixed, known noise (e.g. synthetic data with a known
        # true noise level) -- pass noise_std explicitly to skip learning it.
        bm_fixed = BayesianModule.build(
            module, likelihood="gaussian", X=X, y=y, noise_std=0.1,
        )
    """
    module: nn.Module
    D: int
    param_dict_fn: Callable[[Tensor], dict]
    energy: Callable[[Tensor], Tensor]
    log_likelihood: Callable[[Tensor], Tensor]
    prior_precision: Tensor
    X: Tensor
    y: Tensor
    learns_noise: bool = False
    prior_sigma_scale: Optional[float] = None
    device: torch.device = torch.device("cpu")

    @classmethod
    def build(
        cls,
        module: nn.Module,
        likelihood: Literal["gaussian", "bernoulli", "categorical"],
        X: Tensor,
        y: Tensor,
        prior_precision: float | Tensor = 1.0,
        noise_std: Optional[float] = None,
        prior_sigma_scale: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> "BayesianModule":
        """
        module's CURRENT parameter values are only used for names/shapes
        (the sampler supplies beta externally at every evaluation).
        prior_precision is always length D_network (network weights only);
        when noise is learned, the log_sigma entry is appended and zeroed
        here (its prior lives in the likelihood term instead).
        noise_std=None (default) learns noise: appends `log_sigma` to beta
        (D = D_network + 1) with a HalfNormal(prior_sigma_scale) prior on
        sigma. Pass a float to fix the noise instead (e.g. synthetic data
        with a known true noise, as in scripts/toy_bnn_grid.py).
        """
        device = torch.device(device)
        module = module.to(dtype=dtype, device=device)
        names = [n for n, _ in module.named_parameters()]
        shapes = [p.shape for _, p in module.named_parameters()]
        D_network = sum(s.numel() for s in shapes)
        param_dict_fn = to_param_dict(names, shapes)

        if isinstance(prior_precision, Tensor):
            base_precision = prior_precision.to(dtype=dtype, device=device)
        else:
            base_precision = torch.full((D_network,), float(prior_precision), dtype=dtype, device=device)

        learns_noise = (likelihood == "gaussian" and noise_std is None)

        if learns_noise:
            D = D_network + 1
            # log_sigma's prior lives in the likelihood (HalfNormal-on-sigma
            # term below), not here -- zero its entry to avoid double-counting.
            precision = torch.cat([
                base_precision, torch.zeros(1, dtype=dtype, device=device),
            ])
        else:
            D = D_network
            precision = base_precision

        log_prior = make_gaussian_prior(precision)

        if likelihood == "gaussian":
            log_lik = make_gaussian_likelihood(
                module, param_dict_fn, X, y,
                noise_std=noise_std, prior_sigma_scale=prior_sigma_scale,
            )
        elif likelihood == "bernoulli":
            log_lik = make_bernoulli_likelihood(module, param_dict_fn, X, y)
        elif likelihood == "categorical":
            log_lik = make_categorical_likelihood(module, param_dict_fn, X, y)
        else:
            raise ValueError(f"Unknown likelihood={likelihood!r}")

        energy = make_energy(log_prior, log_lik)
        return cls(
            module=module, D=D, param_dict_fn=param_dict_fn, energy=energy,
            log_likelihood=log_lik, prior_precision=precision,
            X=X.to(dtype=dtype, device=device),
            y=y.to(dtype=dtype, device=device) if likelihood == "gaussian" else y.to(device=device),
            learns_noise=learns_noise,
            prior_sigma_scale=prior_sigma_scale if learns_noise else None,
            device=device,
        )
