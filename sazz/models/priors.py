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


class ModuleGaussianPrior(Prior):
    """Gaussian prior aligned with a ParamSpec's flatten order.

    Accepts either a 1-D precision vector (diagonal prior) or a 2-D precision
    matrix (dense prior, e.g. Zellner's g-prior where the slab covariance is
    g * sigma^2 * (X'X)^{-1}).

    Parameters
    ----------
    prec : Tensor
        Shape [D] for a diagonal prior, or [D, D] for a dense prior.
    """

    def __init__(self, prec: Tensor):
        super().__init__()
        if prec.dim() not in (1, 2):
            raise ValueError(f"prec must be 1-D or 2-D, got shape {prec.shape}")
        self._dense = prec.dim() == 2
        self.register_buffer("_precision", prec)

    def log_prob(self, beta: Tensor) -> Tensor:
        if self._dense:
            return -0.5 * (beta @ self._precision @ beta)
        return -0.5 * (self._precision * beta ** 2).sum()

    def precision_diag(self) -> Tensor:
        if self._dense:
            return self._precision.diagonal()
        return self._precision


class ModuleStudentTPrior(Prior):
    """Student-t prior aligned with a ParamSpec's flatten order.

    Slope coordinates get t_ν(0, scale); bias coordinates get a wide Gaussian,
    mirroring ModuleGaussianPrior's per-coordinate precision vector.

    Parameters
    ----------
    bias_mask : BoolTensor [D]
        True for coordinates that are biases (Gaussian treatment).
        Build from a ParamSpec via:
            torch.tensor([is_b for sz, is_b in zip(spec.numels, spec.is_bias)
                          for _ in range(sz)])
    bias_std : float
        Prior std for bias coordinates.
    scale : float
        Scale parameter of the Student-t for slope coordinates.
    df : float
        Degrees of freedom of the Student-t.
    """
    def __init__(self, bias_mask: Tensor, bias_std: float = 10.0,
                 scale: float = 1.0, df: float = 3.0):
        super().__init__()
        self.df = float(df)
        self.scale = float(scale)
        self.bias_std = float(bias_std)
        self.register_buffer("_bias_mask", bias_mask.bool())

        var = (df / max(df - 2.0, 1e-3)) * scale ** 2 if df > 2.0 else scale ** 2
        prec = torch.where(bias_mask, 1.0 / bias_std ** 2, 1.0 / var)
        self.register_buffer("_precision", prec)

    def log_prob(self, beta: Tensor) -> Tensor:
        slopes = beta[~self._bias_mask]
        z2 = (slopes / self.scale) ** 2
        lp = -0.5 * (self.df + 1.0) * torch.log1p(z2 / self.df).sum()
        biases = beta[self._bias_mask]
        lp = lp - 0.5 * (biases / self.bias_std).pow(2).sum()
        return lp

    def precision_diag(self) -> Tensor:
        return self._precision
    
