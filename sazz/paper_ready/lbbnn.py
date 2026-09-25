"""Latent binary Bayesian neural network (LBBNN) by variational inference,
after Hubin & Storvik, "Variational Inference for Bayesian Neural Networks
under Model and Parameter Uncertainty", adapted to Gaussian regression and
arbitrary depth.

Every weight gets a spike-and-slab variational posterior, an inclusion
indicator with probability alpha = sigmoid(lambda) (concrete-relaxed during
training) times a mean-field Gaussian slab. The priors are a Gaussian slab with the
same fan-in scaled std as the other methods, a Beta-Binomial on the indicators
and a HalfNormal on a learned noise std.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LBBNNConfig:
    layer_sizes: list
    activation: str = "tanh"
    noise_std: Optional[float] = None
    prior_sigma_scale: float = 1.0
    prior_std_weight: float = 3.0
    prior_std_bias: float = 3.0
    temper: float = 0.5
    prior_pa: tuple = (1.0, 1.1)
    prior_pb: tuple = (1.0, 1.1)
    learn_model_prior: bool = True
    epochs: int = 5_000
    batch_size: int = 100
    lr: float = 1e-2


def _normal_log_prob(x, mu, sigma):
    return -0.5 * math.log(2 * math.pi) - torch.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)


class SpikeSlabLinear(nn.Module):
    def __init__(self, n_in: int, n_out: int, cfg: LBBNNConfig):
        super().__init__()
        u = lambda shape, lo, hi: nn.Parameter(torch.empty(shape).uniform_(lo, hi))
        self.w_mu, self.w_rho = u((n_out, n_in), -0.2, 0.2), u((n_out, n_in), -5.0, -4.0)
        self.lam = u((n_out, n_in), 0.0, 1.0)
        self.b_mu, self.b_rho = u(n_out, -0.2, 0.2), u(n_out, -5.0, -4.0)
        pa, pb = torch.empty(1).uniform_(*cfg.prior_pa), torch.empty(1).uniform_(*cfg.prior_pb)
        if cfg.learn_model_prior:
            self.pa, self.pb = nn.Parameter(pa), nn.Parameter(pb)
        else:
            self.register_buffer("pa", pa)
            self.register_buffer("pb", pb)
        self.slab_std = cfg.prior_std_weight / math.sqrt(n_in)
        self.bias_std, self.temper = cfg.prior_std_bias, cfg.temper

    @property
    def alpha(self) -> Tensor:
        return torch.sigmoid(self.lam)

    def forward(self, x: Tensor):
        """Sampled output and the layer's log p - log q."""
        a, w_sd, b_sd = self.alpha, F.softplus(self.w_rho), F.softplus(self.b_rho)
        g = torch.distributions.RelaxedBernoulli(probs=a, temperature=self.temper).rsample()
        w = g * (self.w_mu + w_sd * torch.randn_like(w_sd))
        b = self.b_mu + b_sd * torch.randn_like(b_sd)
        log_slab = -0.5 * math.log(2 * math.pi) - math.log(self.slab_std) - w ** 2 / (2 * self.slab_std ** 2)
        log_p = ((g * log_slab + (1 - g) + 1e-8).sum()
                 + (_normal_log_prob(b, 0.0, torch.tensor(self.bias_std)) + 1e-8).sum()
                 + self._beta_binomial(g))
        log_q = (torch.log(g * _normal_log_prob(w, self.w_mu, w_sd).exp() + (1 - g) + 1e-8).sum()
                 + (g * torch.log(a + 1e-8) + (1 - g) * torch.log(1 - a + 1e-8)).sum()
                 + _normal_log_prob(b, self.b_mu, b_sd).sum())
        return F.linear(x, w, b), log_p - log_q

    def _beta_binomial(self, g: Tensor) -> Tensor:
        pa, pb, lg = self.pa, self.pb, torch.lgamma
        return (lg(g + pa) + lg(1 + pb - g) + lg(pa + pb) - lg(pa + g) - lg(2 - g)
                - lg(1 + pa + pb) - lg(pa) - lg(pb)).sum()


class LBBNN(nn.Module):
    def __init__(self, cfg: LBBNNConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(SpikeSlabLinear(a, b, cfg)
                                    for a, b in zip(cfg.layer_sizes[:-1], cfg.layer_sizes[1:]))
        self.act = torch.tanh if cfg.activation == "tanh" else F.relu
        self.learns_noise = cfg.noise_std is None
        if self.learns_noise:
            self.log_sigma = nn.Parameter(torch.zeros(()))

    def negative_elbo(self, X: Tensor, y: Tensor, n_batches: int) -> Tensor:
        h, kl = X, 0.0
        for i, layer in enumerate(self.layers):
            h, lpq = layer(h)
            kl = kl - lpq
            if i < len(self.layers) - 1:
                h = self.act(h)
        sigma = self.log_sigma.exp() if self.learns_noise else torch.tensor(self.cfg.noise_std)
        nll = -_normal_log_prob(y, h.squeeze(-1), sigma).sum()
        if self.learns_noise:
            s = self.cfg.prior_sigma_scale
            nll = nll - (0.5 * math.log(2 / math.pi) - math.log(s)
                         - 0.5 * (sigma / s) ** 2 + self.log_sigma)
        return nll + kl / n_batches

    @torch.no_grad()
    def sample(self, n: int) -> Tensor:
        """Posterior draws with hard inclusion masks, flattened as
        [W0, b0, W1, b1, ...] (+ log_sigma)."""
        out = []
        for _ in range(n):
            flat = []
            for L in self.layers:
                w = L.w_mu + F.softplus(L.w_rho) * torch.randn_like(L.w_mu)
                w = w * (torch.rand_like(L.alpha) < L.alpha)
                flat += [w.flatten(), L.b_mu + F.softplus(L.b_rho) * torch.randn_like(L.b_mu)]
            if self.learns_noise:
                flat.append(self.log_sigma.reshape(1))
            out.append(torch.cat(flat))
        return torch.stack(out)


def run_lbbnn(data: dict, cfg: LBBNNConfig, seed: int, n_draws: int = 4000,
              device="cpu", dtype=torch.float64):
    """Fit by Adam on the negative ELBO with minibatches. Returns (draws,
    elapsed_sec, full-batch-equivalent gradient evaluations, inclusion probs)."""
    torch.manual_seed(seed)
    X = data["X_train"].to(dtype=dtype, device=device)
    y = data["y_train"].to(dtype=dtype, device=device)
    model = LBBNN(cfg).to(dtype=dtype, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    N = X.shape[0]
    bs = min(cfg.batch_size, N)
    n_batches = math.ceil(N / bs)
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(N, device=device)
        for k in range(n_batches):
            idx = perm[k * bs:(k + 1) * bs]
            opt.zero_grad()
            model.negative_elbo(X[idx], y[idx], n_batches).backward()
            opt.step()
        if epoch % 1000 == 0:
            alpha = torch.cat([L.alpha.flatten() for L in model.layers]).mean()
            print(f"  LBBNN epoch {epoch}/{cfg.epochs}  mean alpha {float(alpha):.3f}")
    elapsed = time.perf_counter() - t0
    evals = round(cfg.epochs * n_batches * bs / N)
    return (model.sample(n_draws).cpu(), elapsed, evals,
            [L.alpha.detach().cpu() for L in model.layers])
