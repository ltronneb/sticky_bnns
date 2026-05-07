import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, Categorical, Bernoulli
from ..utils.bnn_utils import ParamSpec

class Likelihood(nn.Module):
    """
    Likelihood given a flat parameter vector.
    Must provide:
        log_prob(beta: Tensor) -> Tensor (scalar, summed over data)
    """
    def log_prob(self, beta: Tensor) -> Tensor:
        raise NotImplementedError

class ModuleLikelihood(Likelihood):
    def __init__(self, module: nn.Module, spec: ParamSpec, 
                 X: Tensor, y: Tensor):
        super().__init__()
        self.module = module
        self.module.eval()              # turn off dropout/BN updates
        self.spec = spec
        self.register_buffer("X", X)
        self.register_buffer("y", y)
    
    def predict(self, beta: Tensor, X_new: Tensor) -> Tensor:
        return torch.func.functional_call(
            self.module, self.spec.to_dict(beta), (X_new,)
        )
  
        
class ModuleGaussianLikelihood(ModuleLikelihood):
    """BNN regression likelihood evaluated via torch.func.functional_call.

    Wraps an nn.Module + ParamSpec. The forward pass uses functional_call so
    the module's parameters are taken from the flat beta vector instead of
    its own .parameters().
    """

    def __init__(self, module: nn.Module, spec: ParamSpec,
                 X: Tensor, y: Tensor, noise_std: float):
        super().__init__(module, spec, X, y) 
        self.noise_std = noise_std

    def log_prob_single(self, beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        preds = self.predict(beta, X_i).squeeze(-1)
        return Normal(preds, self.noise_std).log_prob(y_i).sum()

    def log_prob(self, beta: Tensor) -> Tensor:
        return self.log_prob_single(beta, self.X, self.y)
 

class ModuleGaussianLikelihoodLearnedNoise(ModuleLikelihood):
    """Gaussian regression with a learned noise scale.

    Sampled coordinate is log_sigma (last entry of beta). The implied prior
    on sigma = exp(log_sigma) is HalfNormal(prior_sigma_scale). The
    half-normal density on sigma, transformed to log_sigma via the standard
    change-of-variables, contributes:

        log p(log_sigma) = -sigma**2 / (2 * tau**2) + log_sigma + const

    The corresponding entry in ModuleGaussianPrior's precision vector
    should be 0, since the prior on log_sigma lives entirely here.
    """
    def __init__(self, module, base_spec, X, y, prior_sigma_scale=1.0):
        extended_spec = base_spec.with_extra_scalar(
            "log_sigma", can_freeze=False,
        )
        super().__init__(module, extended_spec, X, y)
        self.base_spec = base_spec
        self.prior_sigma_scale = prior_sigma_scale

    def _split(self, beta):
        return beta[:-1], beta[-1]

    def predict(self, beta, X_new):
        weights, _ = self._split(beta)
        params = self.base_spec.to_dict(weights)
        return torch.func.functional_call(self.module, params, (X_new,))

    def log_prob_single(self, beta, X_i, y_i):
        _, log_sigma = self._split(beta)
        sigma = log_sigma.exp()
        preds = self.predict(beta, X_i).squeeze(-1)
        log_lik = Normal(preds, sigma).log_prob(y_i).sum()
        # HalfNormal(tau) on sigma, with Jacobian to log_sigma
        log_prior = -0.5 * (sigma / self.prior_sigma_scale) ** 2 + log_sigma
        return log_lik + log_prior

    def log_prob(self, beta):
        return self.log_prob_single(beta, self.X, self.y)
    
 
class ModuleBernoulliLikelihood(ModuleLikelihood):
    """Binary classification likelihood via functional_call.

    The module should return raw logits of shape [N] or [N, 1].
    """

    def __init__(self, module: nn.Module, spec: ParamSpec,
                 X: Tensor, y: Tensor):
        super().__init__(module, spec, X, y.to(X.dtype))

    def log_prob_single(self, beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        logits = self.predict(beta, X_i).squeeze(-1)
        return Bernoulli(logits=logits).log_prob(y_i).sum()

    def log_prob(self, beta: Tensor) -> Tensor:
        return self.log_prob_single(beta, self.X, self.y)


class ModuleCategoricalLikelihood(ModuleLikelihood):
    """Multiclass classification likelihood evaluated via functional_call."""

    def __init__(self, module: nn.Module, spec: ParamSpec,
                 X: Tensor, y: Tensor):
        super().__init__(module, spec, X, y.long())

    def log_prob_single(self, beta: Tensor, X_i: Tensor, y_i: Tensor) -> Tensor:
        logits = self.predict(beta, X_i)
        return Categorical(logits=logits).log_prob(y_i.long()).sum()

    def log_prob(self, beta: Tensor) -> Tensor:
        return self.log_prob_single(beta, self.X, self.y)


# class Likelihood(nn.Module):
#     """
#     Likelihood given a flat parameter vector.
#     Must provide:
#         log_prob(beta: Tensor) -> Tensor (scalar, summed over data)
#     """
#     def log_prob(self, beta: Tensor) -> Tensor:
#         raise NotImplementedError


# class GaussianLikelihood(Likelihood):
#     def __init__(self, X_aug: Tensor, y: Tensor, noise_std: float = 1.0):
#         super().__init__()
#         self.register_buffer("X_aug", X_aug)
#         self.register_buffer("y", y)
#         self.noise_std = noise_std

#     def log_prob(self, beta: Tensor) -> Tensor:
#         pred = self.X_aug @ beta
#         dist = Normal(pred, self.noise_std)
#         return dist.log_prob(self.y).sum()


# class BernoulliLikelihood(Likelihood):
#     def __init__(self, X_aug: Tensor, y: Tensor):
#         super().__init__()
#         self.register_buffer("X_aug", X_aug)
#         self.register_buffer("y", y.to(X_aug.dtype))

#     def log_prob(self, beta: Tensor) -> Tensor:
#         logits = self.X_aug @ beta
#         dist = Bernoulli(logits=logits)
#         return dist.log_prob(self.y).sum()
