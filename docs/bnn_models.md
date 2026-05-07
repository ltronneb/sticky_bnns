# BNN Models: Mathematical Reference

This document covers the feedforward Bayesian neural network used in the UCI regression benchmarks.
The entry point is [`sazz/scripts/bnns/uci_bnn.py`](../sazz/scripts/bnns/uci_bnn.py).

---

## 1. Architecture

A fully connected feedforward network (FFN) with $L$ layers:

$$
h^{(0)} = \mathbf{x} \in \mathbb{R}^{d_0}
$$
$$
h^{(\ell)} = \phi\!\left(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\right), \quad \ell = 1, \ldots, L-1
$$
$$
\hat{y} = W^{(L)} h^{(L-1)} + b^{(L)} \in \mathbb{R}
$$

where $\phi$ is the elementwise activation function. Implemented in `FFN` ([`sazz/models/neural_networks.py`](../sazz/models/neural_networks.py)).

**Defaults:**

| Parameter | Default | Note |
|-----------|---------|------|
| `layer_sizes` | `[p, 50, 1]` | architecture; $p$ is dataset-specific |
| `activation` | `"relu"` | also supports `"tanh"` |

For Boston housing $p = 13$, giving $D = 13 \cdot 50 + 50 + 50 \cdot 1 + 1 = 751$ parameters.

The parameter vector $\beta \in \mathbb{R}^D$ is the concatenation produced by `ParamSpec.from_module` ([`sazz/utils/bnn_utils.py`](../sazz/utils/bnn_utils.py)), in `nn.Module.named_parameters()` order:

$$
\beta = \bigl(\underbrace{\mathrm{vec}(W^{(1)})^\top}_{d_1 d_0},\; b^{(1)\top}_{d_1},\; \ldots,\; \mathrm{vec}(W^{(L)})^\top_{d_L d_{L-1}},\; b^{(L)\top}_{d_L}\bigr)
$$

Within each weight matrix, rows are stacked (row-major, matching PyTorch's `view(-1)`).

---

## 2. Joint Model

The Bayesian model ([`sazz/models/model.py`](../sazz/models/model.py)) factorises as

$$
\log p(\beta \mid \mathbf{y}, X) = \log p(\beta) + \log p(\mathbf{y} \mid X, \beta)
$$

with energy $U(\beta) = -\log p(\beta \mid \mathbf{y}, X)$ targeted by the PDMP samplers via `model.grad_energy`.

---

## 3. Prior

### 3.1 Standard Gaussian Prior

Diagonal Gaussian prior with per-coordinate scale set by `build_prior_precision` ([`sazz/utils/bnn_utils.py`](../sazz/utils/bnn_utils.py)):

$$
\beta_k \sim \mathcal{N}\!\left(0,\; \sigma_k^2\right)
$$

The scale $\sigma_k$ depends on whether coordinate $k$ is a weight or a bias:

**Weights** (with fan-in scaling, `fan_in_scaling = True`):
$$
\sigma_k = \frac{\sigma_w}{\sqrt{d_{\ell-1}}}
$$

where $d_{\ell-1}$ is the fan-in of the layer containing $\beta_k$ (number of incoming connections). This is an isotropic variant of the He initialisation scale and keeps the prior variance roughly constant in terms of the pre-activation signal.

**Weights** (without fan-in scaling, `fan_in_scaling = False`):
$$
\sigma_k = \sigma_w
$$

**Biases:**
$$
\sigma_k = \sigma_b
$$

The resulting precision vector $\lambda_k = \sigma_k^{-2}$ is stored as a 1-D tensor in `ModuleGaussianPrior` ([`sazz/models/priors.py`](../sazz/models/priors.py)):

$$
\log p(\beta) = -\frac{1}{2} \sum_k \lambda_k \beta_k^2
$$

**Defaults (`BNNConfig`):**

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $\sigma_w$ | `prior_std_weight` | $1.0$ |
| $\sigma_b$ | `prior_std_bias` | $1.0$ |
| fan-in scaling | `fan_in_scaling` | `True` |

So a weight in layer 1 of Boston (fan-in = 13) has prior std $1/\sqrt{13} \approx 0.28$; a weight in layer 2 (fan-in = 50) has prior std $1/\sqrt{50} \approx 0.14$.

### 3.2 Spike-and-Slab Prior (Sticky variants)

The sticky sampler implements a spike-and-slab prior by combining the Gaussian slab density above with a point mass at zero. The spike is not placed in the prior object — instead, coordinates frozen to exactly zero by the sampler carry the spike probability. The prior object encodes only the **slab**:

$$
\beta_k \mid \gamma_k = 1 \sim \mathcal{N}(0, \sigma_k^2), \qquad \beta_k \mid \gamma_k = 0 = 0
$$

The slab std $\sigma_k$ uses the same fan-in convention as §3.1.

The prior inclusion probability $w$ appears only in the kappa calibration (§6). Biases are always in the slab ($\gamma_k = 1$ always).

**Defaults:** `prior_inclusion_weight = [0.7, 0.7]` per layer (i.e. $w = 0.7$ for both hidden and output layers; see `configs_for` in [`sazz/scripts/bnns/uci_bnn.py:92`](../sazz/scripts/bnns/uci_bnn.py#L92)).

### 3.3 Horseshoe Prior (NUTS-HS)

The NUTS horseshoe baseline uses a non-centred parameterisation ([`sazz/scripts/bnns/uci_bnn.py:376`](../sazz/scripts/bnns/uci_bnn.py#L376)):

$$
\tau_\ell \sim \text{HalfCauchy}(\sigma_w), \quad \lambda_{ij}^{(\ell)} \sim \text{HalfCauchy}(1)
$$
$$
\tilde{W}_{ij}^{(\ell)} \sim \mathcal{N}(0, 1), \quad W_{ij}^{(\ell)} = \tau_\ell \cdot \lambda_{ij}^{(\ell)} \cdot \tilde{W}_{ij}^{(\ell)}
$$
$$
b_j^{(\ell)} \sim \mathcal{N}(0, \sigma_b^2)
$$

One global scale $\tau_\ell$ per layer, one local scale $\lambda_{ij}$ per weight. Biases are kept Gaussian for comparability. `target_accept = 0.95`.

---

## 4. Likelihood

### 4.1 Fixed Noise (standard)

$$
y_i \mid \mathbf{x}_i, \beta \sim \mathcal{N}\!\left(\hat{y}_i(\beta),\; \sigma_\varepsilon^2\right)
$$

$$
\log p(\mathbf{y} \mid X, \beta) = -\frac{N}{2}\log(2\pi\sigma_\varepsilon^2) - \frac{1}{2\sigma_\varepsilon^2}\sum_{i=1}^N\left(y_i - \hat{y}_i(\beta)\right)^2
$$

Implemented by `ModuleGaussianLikelihood` ([`sazz/models/likelihoods.py`](../sazz/models/likelihoods.py)).

**Defaults:**

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $\sigma_\varepsilon$ | `noise_std` | $0.3$ (boston/energy), $0.01$ (naval) |

Note: data is standardised — $y \leftarrow (y - \bar{y}_\text{train})/s_y$ — so `noise_std` is in units of **standardised** $y$.

### 4.2 Learned Noise (`--learned-noise`)

The noise scale becomes a free parameter. The full sampled vector is extended by one coordinate:

$$
\beta_\text{ext} = (\beta,\; \log\sigma_\varepsilon) \in \mathbb{R}^{D+1}
$$

Implemented by `ModuleGaussianLikelihoodLearnedNoise` ([`sazz/models/likelihoods.py`](../sazz/models/likelihoods.py)).

**Likelihood:**
$$
\log p(\mathbf{y} \mid X, \beta, \log\sigma_\varepsilon) = \sum_{i=1}^N \log\mathcal{N}(y_i;\; \hat{y}_i(\beta),\; e^{2\log\sigma_\varepsilon})
$$

**Prior on $\sigma_\varepsilon$:** Half-Normal with scale `prior_sigma_scale`:

$$
\sigma_\varepsilon \sim \text{HalfNormal}(\sigma_s), \quad \sigma_s = 1.0
$$

On the log scale (with the Jacobian $d\sigma / d\log\sigma = \sigma$), this becomes:

$$
\log p(\log\sigma_\varepsilon) = -\frac{1}{2}\left(\frac{e^{\log\sigma_\varepsilon}}{\sigma_s}\right)^2 + \log\sigma_\varepsilon
$$

This prior is encoded inside the likelihood; the prior precision for the $\log\sigma_\varepsilon$ coordinate is set to $0$ in the `ModuleGaussianPrior` precision vector ([`sazz/scripts/bnns/uci_bnn.py:204`](../sazz/scripts/bnns/uci_bnn.py#L204)).

**Reference point correction:** After MAP optimisation the Hessian of the $\log\sigma_\varepsilon$ coordinate is augmented by the prior curvature ([`sazz/scripts/bnns/uci_bnn.py:217`](../sazz/scripts/bnns/uci_bnn.py#L217)):

$$
\Sigma^{-1}_{D+1, D+1} \mathrel{+}= \frac{2\hat\sigma_\varepsilon^2}{\sigma_s^2}
$$

where $\hat\sigma_\varepsilon = e^{\widehat{\log\sigma_\varepsilon}}$ is the MAP noise scale.

**File naming:** Learned-noise runs are saved with suffix `_ln` (e.g., `sticky_boomerang_ln.pt`). Fixed-noise runs have no suffix.

---

## 5. Data Preprocessing

All data is split 90/10 train/test (per split seed `BASE_SEED + split_id = 42 + split_id`).
Standardisation uses **training set statistics only**:

$$
\tilde{\mathbf{x}}_i = \frac{\mathbf{x}_i - \bar{\mathbf{x}}_\text{train}}{s_\text{train}}, \qquad
\tilde{y}_i = \frac{y_i - \bar{y}_\text{train}}{s_{y,\text{train}}}
$$

Zero-variance features (if any) are kept with $s = 1$. The `y_std` ($s_{y,\text{train}}$) is saved in every `.pt` file for back-transforming predictions to the original scale.

**Splits:** 5 splits by default (`split_id` $\in \{0, 1, 2, 3, 4\}$, `--splits`).

---

## 6. Kappa Calibration (Sticky Samplers)

The kappa parameter controls the spike-and-slab stickiness: larger $\kappa$ means faster thawing (less sticky). It is calibrated from the spike-and-slab prior via `make_kappa_from_inclusion` ([`sazz/utils/bnn_utils.py`](../sazz/utils/bnn_utils.py)):

For weight coordinate $k$ with slab std $\sigma_k$ and prior inclusion probability $w$:

$$
\kappa_k = \frac{w}{1-w} \cdot \frac{1}{\sigma_k \sqrt{2\pi}}
$$

This is the ratio of the spike density at 0 (a point mass with weight $1-w$) to the slab density at 0 ($\mathcal{N}(0; 0, \sigma_k^2) = ({\sigma_k\sqrt{2\pi}})^{-1}$ weighted by $w$), representing the odds for the spike at the origin.

For bias coordinates: $\kappa_k = 10^6$ (never frozen), although this should never materialize as biases have the property `can_freeze=False`.

**Defaults (from `configs_for`):**

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $w$ | `prior_inclusion_weight` | $[0.7, 0.7]$ (per layer) |

Using Boston layer 1 as an example: $\sigma_k = 1/\sqrt{13} \approx 0.277$, $w = 0.7$:

$$
\kappa = \frac{0.7}{0.3} \cdot \frac{1}{0.277 \cdot \sqrt{2\pi}} \approx \frac{2.33}{0.694} \approx 3.36
$$

---

## 7. Reference Point Construction

**Source:** `find_reference_bnn` in [`sazz/utils/warmup.py`](../sazz/utils/warmup.py)

Called with `reference="laplace_diag"` and `n_steps=cfg.adam_steps = 5000`.

**Step 1 — MAP via Adam:**
$$
\hat\beta = \arg\min_\beta U(\beta) = \arg\min_\beta \bigl[-\log p(\beta) - \log p(\mathbf{y}\mid X,\beta)\bigr]
$$
Uses default lr $= 0.01$. The MAP will serve as the starting point for all samplers, unless otherwise specified.

**Step 2 — Diagonal Laplace precision:**
$$
\Sigma^{-1}_{ii} = \lambda_i + \hat{F}_{ii}
$$

where $\lambda_i$ is the prior precision (§3.1) and $\hat{F}_{ii}$ is the diagonal empirical Fisher:

$$
\hat{F}_{ii} = \frac{1}{N} \sum_{n=1}^N \left(\frac{\partial \log p(y_n \mid \mathbf{x}_n, \hat\beta)}{\partial \beta_i}\right)^2
$$

computed on a mini-batch of `n_fisher_batch = 64` points.

This gives a **prior + likelihood curvature** precision that is always positive definite. It is used to scale Boomerang dynamics.

**Step 3 — Boomerang warmup:** Three rounds of `warmup` with 2000 skeleton points each. After each round, the active coordinates (frozen $\leq 90\%$ of time) are identified and a **full covariance block** is computed from the pilot samples. The diagonal Laplace precision is then replaced by this data-derived precision for active coordinates, regularised with $10^{-6} I$.

The tuned refresh rate uses:
$$
\lambda_r = \max\!\left(\frac{\rho}{1-\rho}\,\hat\lambda,\; \frac{1}{\pi}\right), \qquad \rho = 0.7812
$$

where $\hat\lambda$ is the observed bounce rate in the pilot run and $\rho$ is the BPS-optimal refresh fraction.

---

## 8. PDMP Sampler Settings

All four PDMP samplers use PLI thinning (`thinning="pli"`). The following are module-level constants in [`sazz/scripts/bnns/uci_bnn.py`](../sazz/scripts/bnns/uci_bnn.py):

| Symbol | Constant | Value |
|--------|----------|-------|
| — | `N_SKELETON` | $100{,}000$ |
| — | `N_RESAMPLE` | $50{,}000$ |
| $t_\text{burn}$ | `BURNIN_FRAC` | $0.2$ |
| $\lambda_r$ (initial) | — | $1.0$ (boomerang); tuned before run |
| $t_\text{max}$ | `T_MAX_ZZ` | $0.1$ |
| $\gamma$ | `GAMMA_ZZ` | $0.01$ |

Samples are thinned to `N_SAVE = NUTS_DRAWS * NUTS_CHAINS = 8000` before saving.

### Boomerang path interpolation

Trajectory between skeleton events $(x_k, v_k, t_k)$:

$$
x(t) = x_\text{ref} + (x_k - x_\text{ref})\cos(t - t_k) + v_k\sin(t - t_k)
$$

For **sticky Boomerang**, coordinates where $|x_k^{(j)}| < 10^{-12}$ and $|v_k^{(j)}| < 10^{-12}$ are held frozen at exactly $0$ across the interval.

### ZigZag path interpolation

$$
x(t) = x_k + (t - t_k) v_k, \quad v_k \in \{-1, 0, +1\}^D
$$

For **sticky ZigZag**, coordinates with $|v_k^{(j)}| < 10^{-12}$ and $|x_k^{(j)}| < 10^{-12}$ are held at $0$.

Both resampling functions live in [`sazz/utils/sampling.py`](../sazz/utils/sampling.py).

---

## 9. NUTS Baseline

Via NumPyro. Flatten convention matches ParamSpec: for each layer $\ell$, weights $W^{(\ell)}$ are stored row-major then bias $b^{(\ell)}$, so the flat vector is `[W0, b0, W1, b1, ...]` — identical to ParamSpec output.

| Constant | Value |
|----------|-------|
| `NUTS_DRAWS` | $2{,}000$ per chain |
| `NUTS_WARMUP` | $1{,}000$ |
| `NUTS_CHAINS` | $4$ |
| Total samples | $8{,}000$ |
| `target_accept_prob` | $0.9$ (Gaussian prior), $0.95$ (horseshoe) |

With `learned_noise=True`, NUTS samples `sigma ~ HalfNormal(1.0)` as a named variable; it is appended as the last column of the flat array.

---

## 10. Coordinate Ordering Summary

| Convention | Where |
|-----------|-------|
| `[W0, b0, W1, b1, ...]` row-major | ParamSpec, PDMP samples, NUTS samples |
| No rotation applied | Unlike GLM scripts; BNN convention is uniform |

There is **no** bias-first rotation for BNNs — both PDMPs and NUTS produce samples in the same `[weights, bias]` per-layer order, matching what `FFN` evaluates via `functional_call`.

---

## 11. Evaluation

Downstream evaluation is done in the notebooks. Key quantities:

**Predictive mean and variance** (test set, standardised scale):
$$
\bar{y}^* = \frac{1}{S} \sum_{s=1}^S \hat{y}(\mathbf{x}^*; \beta^{(s)})
$$
$$
\mathbb{V}[\hat{y}^*]_\text{epistemic} = \frac{1}{S}\sum_s \hat{y}(\mathbf{x}^*;\beta^{(s)})^2 - \bar{y}^{*2}
$$
$$
\mathbb{V}[\hat{y}^*]_\text{total} = \mathbb{V}[\hat{y}^*]_\text{epistemic} + \sigma_\varepsilon^2
$$

For learned noise, $\sigma_\varepsilon^2$ is replaced by $\frac{1}{S}\sum_s e^{2\log\sigma^{(s)}}$.

**Test RMSE** (original scale, via `y_std`):
$$
\text{RMSE} = s_y \cdot \sqrt{\frac{1}{N_\text{te}} \sum_i (\bar{y}_i^* - \tilde{y}_i)^2}
$$

**Test NLL** (Gaussian predictive):
$$
\text{NLL} = -\frac{1}{N_\text{te}} \sum_i \log \mathcal{N}(y_i^*;\, \bar{y}_i^*,\, \hat\sigma^2_i)
$$

**CRPS** (Gaussian closed form):
$$
\text{CRPS}(\hat\sigma, \mu, y) = \hat\sigma\!\left[\frac{1}{\sqrt{\pi}} - 2\phi\!\left(\frac{y-\mu}{\hat\sigma}\right) - \frac{y-\mu}{\hat\sigma}\!\left(2\Phi\!\left(\frac{y-\mu}{\hat\sigma}\right) - 1\right)\right]
$$
