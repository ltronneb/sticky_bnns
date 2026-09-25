"""Bayesian neural network targets.

The network parameters are flattened into one vector beta (plus a trailing
log_sigma when the Gaussian noise is learned). `BNN.energy(beta)` is the
negative log posterior as a pure function, so it composes with torch.func
(grad, vmap, jvp). Priors are zero-mean Gaussians with fan-in scaled weight
standard deviations. The sticky samplers add a spike at zero through kappa.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical, Normal


def _is_bn_weight(module: nn.Module, name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    return isinstance(module.get_submodule(name[:-7]), nn.modules.batchnorm._BatchNorm)


def prior_std(module: nn.Module, std_weight: float, std_bias: float,
              std_bn: float = 1.0) -> Tensor:
    """Per-parameter prior std, std_weight / sqrt(fan_in) for weights,
    std_bias for biases, std_bn for BatchNorm scales."""
    out = []
    for name, p in module.named_parameters():
        if _is_bn_weight(module, name):
            s = std_bn
        elif p.dim() == 1:
            s = std_bias
        else:
            s = std_weight / math.sqrt(nn.init._calculate_fan_in_and_fan_out(p)[0])
        out.append(torch.full((p.numel(),), s, dtype=torch.float64))
    return torch.cat(out)


def freezable(module: nn.Module) -> Tensor:
    """Weights of linear and convolutional layers, never biases or BatchNorm."""
    return torch.cat([torch.full((p.numel(),), p.dim() > 1 and not _is_bn_weight(module, n))
                      for n, p in module.named_parameters()])


@dataclass
class BNN:
    module: nn.Module
    D: int
    unflatten: Callable[[Tensor], dict]
    energy: Callable[[Tensor], Tensor]
    log_lik: Callable
    prior_precision: Tensor
    X: Tensor
    y: Tensor
    learns_noise: bool
    prior_sigma_scale: Optional[float]
    device: torch.device

    @classmethod
    def build(cls, module: nn.Module, likelihood: str, X: Tensor, y: Tensor, prior_std: Tensor,
              noise_std: Optional[float] = None, prior_sigma_scale: float = 1.0,
              dtype=torch.float64, device="cpu") -> "BNN":
        """likelihood is "gaussian" (noise learned with a HalfNormal(prior_sigma_scale)
        prior unless noise_std is given) or "categorical"."""
        device = torch.device(device)
        module = module.to(dtype=dtype, device=device)
        names = [n for n, _ in module.named_parameters()]
        shapes = [p.shape for _, p in module.named_parameters()]
        sizes = [s.numel() for s in shapes]

        def unflatten(beta: Tensor) -> dict:
            return {n: b.view(s) for n, s, b in zip(names, shapes, beta.split(sizes))}

        X = X.to(dtype=dtype, device=device)
        y = y.to(dtype=dtype, device=device) if likelihood == "gaussian" else y.to(device).long()
        precision = (1.0 / prior_std ** 2).to(dtype=dtype, device=device)
        learns_noise = likelihood == "gaussian" and noise_std is None
        if learns_noise:
            precision = torch.cat([precision, precision.new_zeros(1)])
            s0 = torch.tensor(prior_sigma_scale, dtype=dtype, device=device)

        def log_lik_single(beta: Tensor, Xb: Tensor, yb: Tensor) -> Tensor:
            if likelihood == "categorical":
                logits = torch.func.functional_call(module, unflatten(beta), (Xb,))
                return Categorical(logits=logits, validate_args=False).log_prob(yb).sum()
            w = beta[:-1] if learns_noise else beta
            pred = torch.func.functional_call(module, unflatten(w), (Xb,)).squeeze(-1)
            if not learns_noise:
                return Normal(pred, noise_std, validate_args=False).log_prob(yb).sum()
            sigma = beta[-1].exp()
            # HalfNormal(s0) prior on sigma with the log-Jacobian of log_sigma
            return (Normal(pred, sigma, validate_args=False).log_prob(yb).sum()
                    - 0.5 * (sigma / s0) ** 2 + beta[-1])

        def log_lik(beta: Tensor) -> Tensor:
            return log_lik_single(beta, X, y)

        log_lik.single = log_lik_single

        def energy(beta: Tensor) -> Tensor:
            return 0.5 * (precision * beta ** 2).sum() - log_lik(beta)

        return cls(module, len(precision), unflatten, energy, log_lik, precision, X, y,
                   learns_noise, prior_sigma_scale if learns_noise else None, device)

    def sticky_prior(self, std_weight: float, inclusion: float):
        """Thaw rates kappa and freezable mask for a spike-and-slab prior
        w N(0, s_i^2) + (1 - w) delta_0 on the weights:
        kappa_i = w / (1 - w) / (s_i sqrt(2 pi)), the slab density at zero."""
        mask = freezable(self.module).to(self.device)
        slab = prior_std(self.module, std_weight, 1.0).to(self.X.dtype).to(self.device)
        kappa = torch.where(mask, inclusion / (1 - inclusion) / (slab * math.sqrt(2 * math.pi)),
                            torch.zeros_like(slab))
        if self.learns_noise:
            mask = torch.cat([mask, mask.new_zeros(1)])
            kappa = torch.cat([kappa, kappa.new_zeros(1)])
        return kappa, mask

    def minibatch_grad(self, batch_size: int):
        """Returns (grad_target, resample), the gradient of the energy with the likelihood
        estimated on a minibatch, rescaled by N / batch_size (the noise prior
        is not rescaled). resample() draws a new minibatch."""
        N = self.X.shape[0]
        cache = {}

        def resample():
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            cache["X"], cache["y"] = self.X[idx], self.y[idx]

        def energy_mb(beta: Tensor) -> Tensor:
            ll = self.log_lik.single(beta, cache["X"], cache["y"])
            if self.learns_noise:
                sigma = beta[-1].exp()
                noise_prior = -0.5 * (sigma / self.prior_sigma_scale) ** 2 + beta[-1]
                ll = (ll - noise_prior) * (N / batch_size) + noise_prior
            else:
                ll = ll * (N / batch_size)
            return 0.5 * (self.prior_precision * beta ** 2).sum() - ll

        resample()
        return torch.func.grad(energy_mb), resample

    @torch.no_grad()
    def predict(self, beta: Tensor, X: Tensor, chunk: int = 2048) -> Tensor:
        w = beta[:-1] if self.learns_noise else beta
        params = self.unflatten(w.to(self.device))
        return torch.cat([torch.func.functional_call(self.module, params, (X[i:i + chunk],))
                          for i in range(0, X.shape[0], chunk)])
