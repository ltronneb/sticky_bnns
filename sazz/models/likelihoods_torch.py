import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, Bernoulli, Poisson, Gamma


class Likelihood(nn.Module):
    """
    Likelihood given a flat parameter vector.
    Must provide:
        log_prob(beta: Tensor) -> Tensor (scalar, summed over data)
    """
    def log_prob(self, beta: Tensor) -> Tensor:
        raise NotImplementedError


class GaussianLikelihood(Likelihood):
    def __init__(self, X_aug: Tensor, y: Tensor, noise_std: float = 1.0):
        super().__init__()
        self.register_buffer("X_aug", X_aug)
        self.register_buffer("y", y)
        self.noise_std = noise_std

    def log_prob(self, beta: Tensor) -> Tensor:
        pred = self.X_aug @ beta
        dist = Normal(pred, self.noise_std)
        return dist.log_prob(self.y).sum()


class BernoulliLikelihood(Likelihood):
    def __init__(self, X_aug: Tensor, y: Tensor):
        super().__init__()
        self.register_buffer("X_aug", X_aug)
        self.register_buffer("y", y.to(X_aug.dtype))

    def log_prob(self, beta: Tensor) -> Tensor:
        logits = self.X_aug @ beta
        dist = Bernoulli(logits=logits)
        return dist.log_prob(self.y).sum()
