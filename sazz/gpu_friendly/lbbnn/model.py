"""
LBBNN model: a spike-and-slab ("latent binary") Bayesian neural network
trained by variational inference, after Hubin & Storvik.

Ported from the reference implementation's BayesianLinear/BayesianNetwork
(MNIST/LBBNN-GP-MF-MNIST.py), with three substantive generalizations -- each
needed to run this against the benchmarks in this repo:

 1. ARBITRARY DEPTH. The reference hardcodes `self.l1/l2/l3` and repeats them
    literally in forward(), log_prior(), log_variational_posterior() and
    sample_elbo() (the inclusion probabilities are recomputed by name in five
    places). Here the layers live in an nn.ModuleList driven by `layer_sizes`,
    so the same class serves a 1-100-1 toy and a 784-256-256-10 FFN.

 2. REGRESSION. The reference is classification-only (log_softmax +
    F.nll_loss). Its benchmarks are MNIST/FMNIST/PHONEME; the toy and UCI
    cells in this repo are Gaussian-likelihood regression with a learned
    noise scale. `likelihood="gaussian"` adds a `log_sigma` parameter with
    the same HalfNormal(prior_sigma_scale) prior BayesianModule.build puts on
    it, so LBBNN and the PDMP samplers target the same model. Pass a float
    `noise_std` to fix it instead (what the toys do -- they know their true
    standardized noise).

 3. MINIBATCH-SHAPE INDEPENDENCE. The reference's sample_elbo allocates
    `torch.zeros(samples, BATCH_SIZE, CLASSES)` from a module-level global,
    so a final partial batch silently breaks it. Here the likelihood is
    accumulated as a scalar per MC sample, so any batch size works.

The ELBO itself is unchanged: for each of `samples` Monte Carlo draws, sample
the inclusion mask gamma per layer (concrete-relaxed), sample the weights,
score log p(w, gamma) - log q(w, gamma), and add the negative log likelihood;
the KL term is divided by the number of minibatches so the per-batch loss sums
to the full-dataset ELBO over an epoch.

The parameterization is the reference's: `lambdal` is the logit of the
inclusion probability alpha = sigmoid(lambdal), and the forward pass uses
`weight = gamma * w` so an excluded weight contributes exactly zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sazz.gpu_friendly.lbbnn.distributions import (
    BetaBinomial, Bernoulli, GaussGamma, Gaussian, GaussianPrior,
)


@dataclass
class LBBNNConfig:
    """Hyper-parameters for an LBBNN fit.

    Defaults follow the reference implementation where the reference has an
    opinion (temper/temper_prior 0.001, the uniform_(...) inits, lr 1e-3),
    and follow this repo's conventions where it does not (the Gaussian weight
    prior, fan-in scaling -- see distributions.GaussianPrior for why)."""

    layer_sizes: list[int]
    activation: str = "tanh"

    likelihood: Literal["gaussian", "categorical"] = "gaussian"
    # Gaussian-likelihood only. None learns log_sigma under a
    # HalfNormal(prior_sigma_scale) prior, matching BayesianModule.build.
    noise_std: Optional[float] = None
    prior_sigma_scale: float = 1.0

    # Weight prior: "gaussian" (fixed fan-in-scaled slab, this repo's
    # convention, comparable with the PDMP/NUTS/SVI runs) or "gauss_gamma"
    # (the reference's learnable Normal-Gamma).
    weight_prior: Literal["gaussian", "gauss_gamma"] = "gaussian"
    prior_std_weight: float = 3.0
    prior_std_bias: float = 3.0
    fan_in_scaling: bool = True

    # Concrete relaxation temperatures. The reference uses 0.001 for both on
    # MNIST-scale problems (250 epochs over 60k points = 150k steps). At the
    # much smaller step counts a toy/UCI fit runs, 0.001 leaves alpha pinned
    # near its init -- measured on a 1-50-1 sine target, 400 epochs: alpha
    # stayed in [0.5, 0.73] with zero weights driven to either boundary. 0.5
    # relaxes enough for the inclusion gradient to actually move alpha while
    # still resembling a binary mask, so it is the default here. This is the
    # single most target-sensitive knob in this config -- re-check
    # `mean_alpha` in the training log whenever you move to a new target.
    temper: float = 0.5
    temper_prior: float = 0.5

    # Beta-Binomial model-space prior init (reference: uniform_(1, 1.1)).
    prior_pa_init: tuple[float, float] = (1.0, 1.1)
    prior_pb_init: tuple[float, float] = (1.0, 1.1)
    # Whether the Beta-Binomial hyper-parameters are learned. The reference
    # makes them nn.Parameters; keeping them fixed is a more conservative
    # default for small regression targets where they can run away.
    learn_model_prior: bool = True

    # Variational init (reference: mu ~ U(-0.2, 0.2), rho ~ U(-5, -4),
    # lambdal ~ U(0, 1) i.e. alpha starts near 0.5-0.73).
    mu_init: tuple[float, float] = (-0.2, 0.2)
    rho_init: tuple[float, float] = (-5.0, -4.0)
    lambdal_init: tuple[float, float] = (0.0, 1.0)

    # Optimization. The reference's 250 epochs / lr 1e-3 are for MNIST, where
    # an epoch is 600 minibatch steps; a 200-point toy epoch is 1-4 steps, so
    # matching the reference's epoch count would give ~1000x fewer updates.
    # These defaults are sized for small full-batch-ish targets: on the 1-50-1
    # sine check, 5000 epochs at lr 1e-2 drove mean_alpha 0.61 -> 0.29 with
    # 49% of weights at alpha < 0.1, while 400 epochs at 1e-3 moved it not at
    # all. Large-N targets (MNIST FFN) should override back toward the
    # reference's values, where the step count comes from the data.
    epochs: int = 5_000
    batch_size: int = 100
    lr: float = 1e-2
    mc_samples: int = 1  # reference SAMPLES
    # Reference COND_OPT: mask weight_mu gradients by the sampled gamma, so
    # excluded weights don't drift on their means.
    cond_opt: bool = False


def _get_activation(name: str):
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return F.relu
    raise ValueError(f"Unsupported activation: {name!r}")


class BayesianLinear(nn.Module):
    """One spike-and-slab linear layer: Gaussian variational posterior on the
    weights, relaxed-Bernoulli on the inclusion indicators, Gaussian (or
    Normal-Gamma) slab prior and Beta-Binomial model prior.

    `prior_sigma_w` is the already-fan-in-scaled slab std for this layer, so
    the layer itself needs no knowledge of the scaling convention."""

    def __init__(self, in_features: int, out_features: int, cfg: LBBNNConfig,
                 prior_sigma_w: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cfg = cfg

        # --- weight variational parameters -------------------------------
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(*cfg.mu_init)
        )
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(*cfg.rho_init)
        )

        # --- inclusion (model) variational parameters --------------------
        # alpha = sigmoid(lambdal); stored as a logit so it stays in (0, 1)
        # without a constraint.
        self.lambdal = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(*cfg.lambdal_init)
        )

        # --- bias variational parameters ---------------------------------
        # Biases are never subject to model uncertainty (gamma == 1), matching
        # the reference, which scores them with `torch.ones_like(bias)`.
        self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(*cfg.mu_init))
        self.bias_rho = nn.Parameter(torch.empty(out_features).uniform_(*cfg.rho_init))

        # --- priors -------------------------------------------------------
        if cfg.weight_prior == "gauss_gamma":
            self.weight_a = nn.Parameter(torch.empty(1).uniform_(1.0, 1.1))
            self.weight_b = nn.Parameter(torch.empty(1).uniform_(1.0, 1.1))
            self.bias_a = nn.Parameter(torch.empty(out_features).uniform_(1.0, 1.1))
            self.bias_b = nn.Parameter(torch.empty(out_features).uniform_(1.0, 1.1))
        self.prior_sigma_w = prior_sigma_w

        pa = torch.empty(1).uniform_(*cfg.prior_pa_init)
        pb = torch.empty(1).uniform_(*cfg.prior_pb_init)
        if cfg.learn_model_prior:
            self.pa = nn.Parameter(pa)
            self.pb = nn.Parameter(pb)
        else:
            self.register_buffer("pa", pa)
            self.register_buffer("pb", pb)

        # Per-forward scalars, read back by the network's log_prior() /
        # log_variational_posterior() (the reference's protocol).
        self.log_prior = torch.zeros(())
        self.log_variational_posterior = torch.zeros(())
        self.gammas: Optional[Tensor] = None

    # -- distribution views over the current parameters --------------------
    # Rebuilt per call rather than cached: the reference caches these at
    # __init__ and relies on the Gaussian holding references to the live
    # nn.Parameters, which works but silently breaks under any
    # reparameterization (e.g. torch.func). Rebuilding is cheap.

    @property
    def weight(self) -> Gaussian:
        return Gaussian(self.weight_mu, self.weight_rho)

    @property
    def bias(self) -> Gaussian:
        return Gaussian(self.bias_mu, self.bias_rho)

    @property
    def alpha(self) -> Tensor:
        return torch.sigmoid(self.lambdal)

    def _weight_prior(self):
        if self.cfg.weight_prior == "gauss_gamma":
            return GaussGamma(self.weight_a, self.weight_b)
        return GaussianPrior(self.prior_sigma_w)

    def _bias_prior(self):
        if self.cfg.weight_prior == "gauss_gamma":
            return GaussGamma(self.bias_a, self.bias_b)
        return GaussianPrior(self.cfg.prior_std_bias)

    def sample_gamma(self) -> Tensor:
        """Draw a relaxed inclusion mask from the current alpha."""
        return Bernoulli(self.alpha, temperature=self.cfg.temper_prior).rsample()

    def forward(self, input: Tensor, cgamma: Optional[Tensor] = None,
                sample: bool = False, medimean: bool = False,
                calculate_log_probs: bool = False) -> Tensor:
        """
        Three modes, as in the reference:
          - training or `sample`: sample both gamma (passed in as `cgamma`,
            so the caller controls the MC draw) and the weights.
          - `medimean`: use the weight means under the *given* mask -- the
            median-probability-model path when cgamma is a hard alpha > 0.5.
          - otherwise: the joint posterior mean, alpha * mu (no sampling).
        """
        if self.training or sample:
            assert cgamma is not None, "cgamma required when sampling"
            self.gammas = cgamma
            weight = cgamma * self.weight.rsample()
            bias = self.bias.rsample()
        elif medimean:
            assert cgamma is not None, "cgamma required for medimean"
            weight = cgamma * self.weight_mu
            bias = self.bias_mu
        else:
            weight = self.alpha * self.weight_mu
            bias = self.bias_mu

        if self.training or calculate_log_probs:
            self.log_prior = (
                self._weight_prior().log_prob(weight, cgamma)
                + self._bias_prior().log_prob(bias, torch.ones_like(bias))
                + BetaBinomial(self.pa, self.pb).log_prob(cgamma, pa=self.pa, pb=self.pb)
            )
            self.log_variational_posterior = (
                self.weight.full_log_prob(input=weight, gamma=cgamma)
                + Bernoulli(self.alpha, temperature=self.cfg.temper_prior).log_prob(cgamma)
                + self.bias.log_prob(bias)
            )

        return F.linear(input, weight, bias)


class LBBNN(nn.Module):
    """Arbitrary-depth latent-binary BNN.

    Output convention differs by likelihood, and deliberately differs from the
    reference for classification: the reference applies
    `log_softmax(relu(l3(...)))`, i.e. a ReLU on the final logits, which
    clamps every negative logit to zero and is almost certainly an oversight
    rather than a modeling choice. Here the activation is applied only between
    layers, as in this repo's FFN. `legacy_output_relu=True` restores the
    reference's exact behavior for a like-for-like reproduction."""

    def __init__(self, cfg: LBBNNConfig, legacy_output_relu: bool = False):
        super().__init__()
        self.cfg = cfg
        self.legacy_output_relu = legacy_output_relu
        self.activation = _get_activation(cfg.activation)

        sizes = cfg.layer_sizes
        self.layers = nn.ModuleList([
            BayesianLinear(
                n_in, n_out, cfg,
                prior_sigma_w=(
                    cfg.prior_std_weight / math.sqrt(n_in)
                    if cfg.fan_in_scaling else cfg.prior_std_weight
                ),
            )
            for n_in, n_out in zip(sizes[:-1], sizes[1:])
        ])

        self.learns_noise = (cfg.likelihood == "gaussian" and cfg.noise_std is None)
        if self.learns_noise:
            # Point estimate of log_sigma (the reference has no analogue --
            # it is classification-only). Kept as a plain parameter rather
            # than a variational site so the added KL term stays zero and the
            # comparison against the samplers' learned-noise coordinate is
            # about the weights, not about how noise is treated.
            self.log_sigma = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor, gammas: Optional[list[Tensor]] = None,
                sample: bool = False, medimean: bool = False) -> Tensor:
        h = x
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            g = gammas[i] if gammas is not None else None
            h = layer(h, g, sample=sample, medimean=medimean)
            if i < n - 1:
                h = self.activation(h)
        if self.cfg.likelihood == "categorical":
            if self.legacy_output_relu:
                h = F.relu(h)
            return F.log_softmax(h, dim=1)
        return h

    def log_prior(self) -> Tensor:
        return sum(layer.log_prior for layer in self.layers)

    def log_variational_posterior(self) -> Tensor:
        return sum(layer.log_variational_posterior for layer in self.layers)

    def sample_gammas(self) -> list[Tensor]:
        return [layer.sample_gamma() for layer in self.layers]

    def _negative_log_likelihood(self, output: Tensor, target: Tensor) -> Tensor:
        """Summed (not averaged) NLL over the batch, matching the reference's
        `reduction='sum'` so the KL/NUM_BATCHES scaling is correct."""
        if self.cfg.likelihood == "categorical":
            return F.nll_loss(output, target, reduction="sum")

        pred = output.squeeze(-1)
        if self.learns_noise:
            sigma = torch.exp(self.log_sigma)
        else:
            sigma = torch.tensor(
                self.cfg.noise_std, dtype=output.dtype, device=output.device
            )
        nll = (
            0.5 * math.log(2 * math.pi)
            + torch.log(sigma)
            + 0.5 * ((target - pred) / sigma) ** 2
        ).sum()
        if self.learns_noise:
            # HalfNormal(prior_sigma_scale) prior on sigma, plus the
            # log|d sigma / d log_sigma| Jacobian -- the same term
            # BayesianModule.build adds for its log_sigma coordinate, so the
            # two targets agree.
            scale = self.cfg.prior_sigma_scale
            log_prior_sigma = (
                0.5 * math.log(2.0 / math.pi)
                - math.log(scale)
                - 0.5 * (sigma / scale) ** 2
                + self.log_sigma
            )
            nll = nll - log_prior_sigma
        return nll

    def sample_elbo(self, input: Tensor, target: Tensor, num_batches: int,
                    samples: Optional[int] = None):
        """Monte Carlo ELBO for one minibatch.

        Returns (loss, log_prior, log_variational_posterior, nll), where
        `loss = nll + (log_q - log_p) / num_batches` -- the reference's
        scaling, so summing over an epoch's batches gives the full-data
        negative ELBO."""
        samples = samples or self.cfg.mc_samples

        log_priors = []
        log_variational_posteriors = []
        nlls = []
        for _ in range(samples):
            gammas = self.sample_gammas()
            output = self.forward(input, gammas=gammas, sample=True)
            log_priors.append(self.log_prior())
            log_variational_posteriors.append(self.log_variational_posterior())
            nlls.append(self._negative_log_likelihood(output, target))

        log_prior = torch.stack(log_priors).mean()
        log_variational_posterior = torch.stack(log_variational_posteriors).mean()
        negative_log_likelihood = torch.stack(nlls).mean()

        loss = negative_log_likelihood + (
            log_variational_posterior - log_prior
        ) / num_batches
        return loss, log_prior, log_variational_posterior, negative_log_likelihood

    # -- posterior access ---------------------------------------------------

    @torch.no_grad()
    def inclusion_probabilities(self) -> list[Tensor]:
        """Per-layer alpha, the marginal posterior inclusion probabilities --
        the LBBNN analogue of a sticky sampler's per-coordinate freeze
        fraction."""
        return [layer.alpha.detach().clone() for layer in self.layers]

    @torch.no_grad()
    def sample_posterior_flat(self, n: int, generator=None,
                              hard_gamma: bool = True) -> Tensor:
        """Draw `n` samples from the variational posterior and flatten them
        to this repo's convention: [W0, b0, W1, b1, ...] with each W flattened
        row-major in (n_out, n_in), matching FFN.named_parameters() order (and
        therefore BayesianModule's beta layout and run_nuts/run_svi's output).

        A learned log_sigma is appended as the trailing coordinate, exactly as
        BayesianModule.build does when it learns noise.

        `hard_gamma` draws a genuine binary mask (a draw from the model space)
        rather than a concrete relaxation, so the returned samples carry exact
        zeros and downstream sparsity measurements are meaningful."""
        device = self.layers[0].weight_mu.device
        dtype = self.layers[0].weight_mu.dtype

        out = []
        for _ in range(n):
            flat = []
            for layer in self.layers:
                eps_w = torch.randn(
                    layer.weight_mu.shape, generator=generator,
                    dtype=dtype, device=device,
                )
                w = layer.weight_mu + layer.weight.sigma * eps_w
                if hard_gamma:
                    u = torch.rand(
                        layer.alpha.shape, generator=generator,
                        dtype=dtype, device=device,
                    )
                    gamma = (u < layer.alpha).to(dtype)
                else:
                    gamma = layer.alpha
                w = gamma * w

                eps_b = torch.randn(
                    layer.bias_mu.shape, generator=generator,
                    dtype=dtype, device=device,
                )
                b = layer.bias_mu + layer.bias.sigma * eps_b

                flat.append(w.reshape(-1))
                flat.append(b.reshape(-1))
            if self.learns_noise:
                flat.append(self.log_sigma.reshape(1))
            out.append(torch.cat(flat))
        return torch.stack(out)
