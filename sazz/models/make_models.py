import torch
import torch.nn as nn
from torch import Tensor

#from sazz.models.smoke_test import TorchTarget
from sazz.models.models_torch import BayesianModel
from .priors_torch import GaussianPrior, StudentTPrior
from .likelihoods_torch import GaussianLikelihood, BernoulliLikelihood
from sazz.utils.warmup import find_reference_glm

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

def _prepend_intercept(X: Tensor) -> Tensor:
    """Add a column of 1s at position 0 for the implicit intercept."""
    N = X.shape[0]
    ones = torch.ones(N, 1, dtype=X.dtype, device=X.device)
    return torch.cat([ones, X], dim=1)
    
    
def make_linear_regression(X, y, prior_dist="Gauss", prior_std=1.0, intercept_prior_std=10.0,
                           noise_std=1.0, diagonal_only=True, adam_steps=1000, adam_lr=1e-2,
                           dtype=torch.float64, device="cpu"):
    X = X.to(dtype=dtype, device=device)
    y = y.to(dtype=dtype, device=device)
    
    X_aug = _prepend_intercept(X)
    D = X_aug.shape[1]
    
    if prior_dist == "Gauss":
        prior = GaussianPrior(
            D, prior_std=prior_std,
            intercept_prior_std=intercept_prior_std,
            dtype=dtype, device=device,
        )
    elif prior_dist == "StudT":
        prior = StudentTPrior(
            D, prior_scale=prior_std, df=3.0,
            intercept_prior_std=intercept_prior_std,
            dtype=dtype, device=device,
        )
    else:
        raise ValueError(f"Unknown prior_dist: {prior_dist}")
    likelihood = GaussianLikelihood(X_aug, y, noise_std)
    model = BayesianModel(prior, likelihood)

    # Reference
    x_ref, Sigma_inv = find_reference_glm(model.energy, D, diagonal_only=diagonal_only, n_steps=adam_steps, lr=adam_lr)

    return TorchTarget(
        name=f"linear_regression_D{D-1}",
        D=D,
        grad_target=model.grad_energy,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={"model": model, "X_aug": X_aug, "y": y, "has_intercept": True},
    )
    

def make_logistic_regression(X, y, prior_dist="Gauss", prior_std=1.0, intercept_prior_std=10.0,
                           noise_std=1.0, diagonal_only=True, adam_steps=1000, adam_lr=1e-2,
                           dtype=torch.float64, device="cpu"):
    X = X.to(dtype=dtype, device=device)
    y = y.to(dtype=dtype, device=device)
    
    X_aug = _prepend_intercept(X)
    D = X_aug.shape[1]

    #prior = GaussianPrior(D, prior_std, intercept_prior_std, dtype, device)
    if prior_dist == "Gauss":
        prior = GaussianPrior(
            D, prior_std=prior_std,
            intercept_prior_std=intercept_prior_std,
            dtype=dtype, device=device,
        )
    elif prior_dist == "StudT":
        prior = StudentTPrior(
            D, prior_scale=prior_std, df=3.0,
            intercept_prior_std=intercept_prior_std,
            dtype=dtype, device=device,
        )
    else:
        raise ValueError(f"Unknown prior_dist: {prior_dist}")
    likelihood = BernoulliLikelihood(X_aug, y)
    model = BayesianModel(prior, likelihood)

    # Reference
    x_ref, Sigma_inv = find_reference_glm(model.energy, D, diagonal_only=diagonal_only, n_steps=adam_steps, lr=adam_lr)

    return TorchTarget(
        name=f"linear_regression_D{D-1}",
        D=D,
        grad_target=model.grad_energy,
        x_ref=x_ref,
        Sigma_inv=Sigma_inv,
        meta={"model": model, "X_aug": X_aug, "y": y, "has_intercept": True},
    )
    
