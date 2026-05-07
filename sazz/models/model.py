import torch
import torch.nn as nn
from torch import Tensor

from .priors import Prior
from .likelihoods import Likelihood

class TorchTarget:
    """Minimal container matching what the sampler needs."""
    def __init__(self, name, D, grad_target, x_ref, Sigma_inv, 
                 marginal_grids=None, meta=None):
        self.name = name
        self.D = D
        self.grad_target = grad_target        # callable: Tensor -> Tensor
        self.x_ref = x_ref                    # Tensor [D]
        self.Sigma_inv = Sigma_inv            # Tensor [D,D]
        self.marginal_grids = marginal_grids  # dict[int, {grid, pdf, label}]
        self.meta = meta or {}


class BayesianModel(nn.Module):
    """
    Target = prior * likelihood. Exposes energy (negative log posterior)
    and gradient as the PDMP sampler needs them.
    """
    def __init__(self, prior: Prior, likelihood: Likelihood):
        super().__init__()
        self.prior = prior
        self.likelihood = likelihood

    def log_posterior(self, beta: Tensor) -> Tensor:
        return self.prior.log_prob(beta) + self.likelihood.log_prob(beta)

    def energy(self, beta: Tensor) -> Tensor:
        return -self.log_posterior(beta)

    def grad_energy(self, beta: Tensor) -> Tensor:
        with torch.enable_grad():
            b = beta.detach().requires_grad_(True)
            E = self.energy(b)
            g, = torch.autograd.grad(E, b)
        return g

