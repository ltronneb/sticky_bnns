import torch
import torch.nn as nn
from torch import Tensor

from .priors_torch import Prior
from .likelihoods_torch import Likelihood


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

