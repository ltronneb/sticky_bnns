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
