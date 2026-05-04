import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class Prior(nn.Module):
    """
    A prior over a flat parameter vector.
    Must provide:
        log_prob(beta: Tensor) -> Tensor (scalar)
        precision_diag() -> Tensor [D] (for use as Sigma_inv when applicable)
    """
    def log_prob(self, beta: Tensor) -> Tensor:
        raise NotImplementedError

    def precision_diag(self) -> Tensor:
        raise NotImplementedError


class GaussianPrior(Prior):
    def __init__(self, D: int, prior_std: float = 1.0,
                 intercept_prior_std: Optional[float] = None,
                 dtype=torch.float64, device="cpu"):
        super().__init__()
        prec = torch.full((D,), 1.0 / prior_std ** 2, dtype=dtype, device=device)
        if intercept_prior_std is not None:
            prec[0] = 1.0 / intercept_prior_std ** 2
        self.register_buffer("_precision", prec)

    def log_prob(self, beta: Tensor) -> Tensor:
        return -0.5 * (self._precision * beta ** 2).sum()

    def precision_diag(self) -> Tensor:
        return self._precision


class StudentTPrior(Prior):
    """Student-t prior: heavy-tailed slab, lets large signals escape shrinkage.

    Marginal: β_i ~ t_ν(0, scale).  ν=1 is Cauchy (matches horseshoe tails).
    The intercept gets a wide Gaussian (Student-t with finite-variance ν is fine
    too; Gaussian keeps it simple and matches your existing convention).
    """
    def __init__(self, D, prior_scale=1.0, df=3.0,
                 intercept_prior_std=None,
                 dtype=torch.float64, device="cpu"):
        super().__init__()
        self.D = D
        self.df = float(df)
        self.scale = float(prior_scale)
        self.intercept_prior_std = intercept_prior_std
        var = (self.df / max(self.df - 2.0, 1e-3)) * self.scale ** 2 if df > 2.0 \
              else self.scale ** 2
        prec = torch.full((D,), 1.0 / var, dtype=dtype, device=device)
        if intercept_prior_std is not None:
            prec[0] = 1.0 / intercept_prior_std ** 2
        self.register_buffer("_precision", prec)

    def log_prob(self, beta: Tensor) -> Tensor:
        slopes = beta[1:] if self.intercept_prior_std is not None else beta
        z2 = (slopes / self.scale) ** 2
        slab_lp = -0.5 * (self.df + 1.0) * torch.log1p(z2 / self.df).sum()
        if self.intercept_prior_std is not None:
            intercept = beta[0]
            slab_lp = slab_lp - 0.5 * (intercept / self.intercept_prior_std) ** 2
        return slab_lp

    def precision_diag(self) -> Tensor:
        return self._precision
    
