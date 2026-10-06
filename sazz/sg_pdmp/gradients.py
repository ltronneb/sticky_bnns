"""Stochastic gradient of a BNN energy with control variates (Fearnhead et al.,
2024, eq. 2), on the targets of sazz.paper_ready.models.bnn.

The energy is U(x) = P(x) - sum_j l_j(x), with P the prior part (the Gaussian
prior, plus the HalfNormal noise prior and its log-Jacobian when the noise is
learned) and l_j the log likelihood of data point j. For a minibatch B of n
points drawn uniformly with replacement and a centre x_hat,

    g(x) = grad P(x) - (N / n) sum_{j in B} [grad l_j(x) - grad l_j(x_hat)] - sum_j grad l_j(x_hat)

is unbiased for grad U(x). With batch_size = 1 this is the paper's estimator.
Without a centre the bracket keeps only grad l_j(x) and the last sum is dropped.

The per-point gradients at x_hat are kept in an [N, D] table when it fits in
cache_bytes, so a call costs n per-point gradients. Otherwise they are
recomputed on the batch, and a call costs 2n.
"""

from typing import Optional

import torch
from torch import Tensor


class ControlVariateGradient:
    def __init__(self, bm, batch_size: int, x_hat: Optional[Tensor] = None,
                 cache_bytes: int = 2 ** 30, chunk: int = 256):
        self.bm, self.n = bm, batch_size
        self.N = bm.X.shape[0]
        self.X, self.y = bm.X, bm.y
        self.prec = bm.prior_precision
        self.learns_noise = bm.learns_noise
        self.s0 = bm.prior_sigma_scale

        def ll_data(beta: Tensor, Xb: Tensor, yb: Tensor) -> Tensor:
            ll = bm.log_lik.single(beta, Xb, yb)
            return ll - self._noise_prior(beta) if self.learns_noise else ll

        def grad_ll(beta: Tensor, Xb: Tensor, yb: Tensor) -> Tensor:
            # plain autograd, about half the overhead of torch.func.grad per call
            with torch.enable_grad():
                b = beta.detach().requires_grad_(True)
                return torch.autograd.grad(ll_data(b, Xb, yb), b)[0]

        self._grad_ll = grad_ll
        self.data_grads = 0          # per-point likelihood gradients evaluated so far
        self.x_hat = self.table = self.full_hat = None
        if x_hat is not None:
            self.x_hat = x_hat.to(dtype=bm.X.dtype, device=bm.device)
            itemsize = torch.finfo(bm.X.dtype).bits // 8
            if self.N * bm.D * itemsize <= cache_bytes:
                per_point = torch.func.vmap(torch.func.grad(ll_data), in_dims=(None, 0, 0))
                self.table = torch.cat([per_point(self.x_hat, self.X[i:i + chunk].unsqueeze(1),
                                                  self.y[i:i + chunk].unsqueeze(1))
                                        for i in range(0, self.N, chunk)])
                self.full_hat = self.table.sum(0)
            else:
                self.full_hat = sum(self._grad_ll(self.x_hat, self.X[i:i + chunk], self.y[i:i + chunk])
                                    for i in range(0, self.N, chunk))
            self.data_grads += self.N
        self.cost_per_call = batch_size * (2 if self.x_hat is not None and self.table is None else 1)

    def _noise_prior(self, beta: Tensor) -> Tensor:
        """log HalfNormal(s0) density of sigma plus the log-Jacobian of log_sigma."""
        return -0.5 * (beta[-1].exp() / self.s0) ** 2 + beta[-1]

    def grad_prior(self, x: Tensor) -> Tensor:
        g = self.prec * x
        if self.learns_noise:
            g = g.clone()
            g[-1] = g[-1] + (2 * x[-1]).exp() / self.s0 ** 2 - 1.0
        return g

    def __call__(self, x: Tensor) -> Tensor:
        idx = torch.randint(0, self.N, (self.n,), device=self.X.device)
        Xb, yb = self.X[idx], self.y[idx]
        diff = self._grad_ll(x, Xb, yb)
        if self.x_hat is not None:
            diff = diff - (self.table[idx].sum(0) if self.table is not None
                           else self._grad_ll(self.x_hat, Xb, yb))
        g = self.grad_prior(x) - (self.N / self.n) * diff
        self.data_grads += self.cost_per_call
        return g if self.x_hat is None else g - self.full_hat

    def iterations_for(self, grad_budget: float) -> int:
        """Iterations that spend grad_budget full-data gradients in total,
        including the pass at x_hat already made."""
        return max(1, int((grad_budget * self.N - self.data_grads) // self.cost_per_call))
