"""
Variational and prior distributions for LBBNN (Latent Binary Bayesian Neural
Networks), ported from Hubin & Storvik, "Variational Inference for Bayesian
Neural Networks under Model and Parameter Uncertainty" --

    https://github.com/aliaksah/Variational-Inference-for-Bayesian-Neural-Networks-under-Model-and-Parameter-Uncertainty

The reference implementation is a set of standalone per-dataset scripts
(MNIST/LBBNN-GP-MF-MNIST.py and siblings) with everything at module scope:
DEVICE, TEMPER/TEMPER_PRIOR, BATCH_SIZE and the architecture are globals read
directly from inside the class bodies. These ports keep the math verbatim and
change only the plumbing:

  - `DEVICE` reads become `self.mu.device` / the passed tensor's device, so a
    layer works wherever its parameters live (the reference hardcodes
    `torch.cuda.set_device(0)` at import).
  - `TEMPER_PRIOR` becomes a constructor arg on `Bernoulli` (`temperature`),
    since the right relaxation temperature is target-dependent -- the
    reference's 0.001 was tuned for 784-400-600-10 on MNIST and is very
    unlikely to transfer to e.g. a 1-100-1 toy regression.
  - `exact` (sample a hard Bernoulli / round gamma before scoring, instead of
    the concrete relaxation) stays a per-instance flag, as in the reference.
    It is what the median-probability-model evaluation path wants.

Numerical note: the `+ 1e-8` guards inside the logs are the reference's and
are kept -- they matter, since alpha is driven to the 0/1 boundary during
training and `log(0)` there would poison the ELBO.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


class Gaussian:
    """Mean-field Gaussian variational posterior over one parameter block,
    parameterized by (mu, rho) with sigma = softplus(rho)."""

    def __init__(self, mu: Tensor, rho: Tensor):
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self) -> Tensor:
        return torch.log1p(torch.exp(self.rho))

    def rsample(self) -> Tensor:
        epsilon = self.normal.sample(self.rho.size()).to(
            dtype=self.rho.dtype, device=self.rho.device
        )
        return self.mu + self.sigma * epsilon

    def log_prob(self, input: Tensor) -> Tensor:
        return (
            -math.log(math.sqrt(2 * math.pi))
            - torch.log(self.sigma)
            - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)
        ).sum()

    def log_prob_iid(self, input: Tensor) -> Tensor:
        """Elementwise (un-summed) counterpart of log_prob, used by
        full_log_prob's spike-and-slab mixture."""
        return (
            -math.log(math.sqrt(2 * math.pi))
            - torch.log(self.sigma)
            - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)
        )

    def full_log_prob(self, input: Tensor, gamma: Tensor) -> Tensor:
        """log q(w) for the spike-and-slab variational posterior: a gamma-
        weighted mixture of the Gaussian slab and a point mass at zero
        (whose density contributes the `(1 - gamma)` term)."""
        return torch.log(
            gamma * torch.exp(self.log_prob_iid(input)) + (1 - gamma) + 1e-8
        ).sum()


class Bernoulli:
    """Variational posterior over the binary inclusion indicators, relaxed to
    a concrete/RelaxedBernoulli so the ELBO stays reparameterizable.

    `temperature` is the reference's TEMPER_PRIOR global. Setting `exact=True`
    swaps the relaxation for a hard Bernoulli draw (and rounds before scoring
    in log_prob) -- used for median-probability-model style evaluation, not
    for training (it has no usable pathwise gradient)."""

    def __init__(self, alpha: Tensor, temperature: float = 0.001, exact: bool = False):
        self.alpha = alpha
        self.temperature = temperature
        self.exact = exact

    def rsample(self) -> Tensor:
        if self.exact:
            return torch.distributions.Bernoulli(self.alpha).sample()
        return torch.distributions.RelaxedBernoulli(
            probs=self.alpha, temperature=self.temperature
        ).rsample()

    def log_prob(self, input: Tensor) -> Tensor:
        if self.exact:
            gamma = torch.round(input.detach())
        else:
            gamma = input
        return (
            gamma * torch.log(self.alpha + 1e-8)
            + (1 - gamma) * torch.log(1 - self.alpha + 1e-8)
        ).sum()


class GaussGamma:
    """The reference's learnable Normal-Gamma prior on the weights: a
    Gamma(a, b) prior on the precision tau, with the weight scored under
    N(0, 1/tau) only where gamma selects it in.

    Kept verbatim from the reference (including the slightly unusual
    `- tau * input**2` term sitting outside the gamma weighting, and the
    `+ (1 - gamma) + 1e-8` slack). Available via
    LBBNNConfig(weight_prior="gauss_gamma"), but NOT the default here -- see
    GaussianPrior below for why."""

    def __init__(self, a: Tensor, b: Tensor):
        self.a = a
        self.b = b
        self.exact = False

    def log_prob(self, input: Tensor, gamma: Tensor) -> Tensor:
        tau = torch.distributions.Gamma(self.a, self.b).rsample()
        g = torch.round(gamma.detach()) if self.exact else gamma
        two_pi = torch.tensor(2 * math.pi, dtype=input.dtype, device=input.device)
        return (
            g
            * (
                self.a * torch.log(self.b)
                + (self.a - 0.5) * tau
                - self.b * tau
                - torch.lgamma(self.a)
                - 0.5 * torch.log(two_pi)
            )
            - tau * torch.pow(input, 2)
            + (1 - g)
            + 1e-8
        ).sum()


class GaussianPrior:
    """Fixed-scale Gaussian slab prior, scored only where gamma selects the
    weight in.

    This is NOT in the reference implementation -- it is the default here so
    that an LBBNN run and a PDMP/NUTS/SVI run on the same dataset share the
    *same* prior. Everything else in this repo puts a fixed fan-in-scaled
    Gaussian on the weights (build_fan_in_prior_precision: sigma =
    prior_std_weight / sqrt(fan_in)), so using the reference's learnable
    Normal-Gamma by default would confound "LBBNN vs sticky sampler" with
    "Normal-Gamma prior vs fan-in Gaussian prior". Pass
    weight_prior="gauss_gamma" to get the paper's version instead.

    `sigma` is a scalar (a python float, already fan-in scaled by the caller).
    The `(1 - gamma)` term mirrors GaussGamma's, keeping the spike branch's
    contribution on the same footing as the reference."""

    def __init__(self, sigma: float):
        self.sigma = sigma
        self.exact = False

    def log_prob(self, input: Tensor, gamma: Tensor) -> Tensor:
        g = torch.round(gamma.detach()) if self.exact else gamma
        log_slab = (
            -math.log(math.sqrt(2 * math.pi))
            - math.log(self.sigma)
            - (input ** 2) / (2 * self.sigma ** 2)
        )
        return (g * log_slab + (1 - g) + 1e-8).sum()


class BetaBinomial:
    """Beta-Binomial prior on the inclusion indicators, giving the model-space
    prior. Ported verbatim (the lgamma expression is the reference's
    n=1 Beta-Binomial pmf written out in full, including the
    `lgamma(ones_like(input))` term that is identically zero)."""

    def __init__(self, pa: Tensor, pb: Tensor):
        self.pa = pa
        self.pb = pb
        self.exact = False

    def log_prob(self, input: Tensor, pa: Tensor, pb: Tensor) -> Tensor:
        gamma = torch.round(input.detach()) if self.exact else input
        ones = torch.ones_like(input)
        return (
            torch.lgamma(ones)
            + torch.lgamma(gamma + ones * self.pa)
            + torch.lgamma(ones * (1 + self.pb) - gamma)
            + torch.lgamma(ones * (self.pa + self.pb))
            - torch.lgamma(ones * self.pa + gamma)
            - torch.lgamma(ones * 2 - gamma)
            - torch.lgamma(ones * (1 + self.pa + self.pb))
            - torch.lgamma(ones * self.pa)
            - torch.lgamma(ones * self.pb)
        ).sum()

    def rsample(self) -> Tensor:
        probs = torch.distributions.Beta(self.pa, self.pb).rsample()
        return torch.distributions.RelaxedBernoulli(
            probs=probs, temperature=0.001
        ).rsample()
