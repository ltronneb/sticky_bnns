# GLM Models: Mathematical Reference

This document covers the three GLM benchmark scripts and the Clyde et al. (2010) reproduction.
All scripts share the same sampler infrastructure in [`sazz/utils/glm_utils.py`](../sazz/utils/glm_utils.py).

---

## 1. Shared Model Structure

Every GLM is a Bayesian model with joint log-posterior

$$
\log p(\beta \mid \mathbf{y}) = \log p(\beta) + \log p(\mathbf{y} \mid \beta)
$$

and energy (negative log-posterior)

$$
U(\beta) = -\log p(\beta) - \log p(\mathbf{y} \mid \beta).
$$

The PDMP samplers target $\pi(\beta) \propto e^{-U(\beta)}$ via `model.grad_energy` ([`sazz/models/model.py`](../sazz/models/model.py)).

The parameter vector $\beta \in \mathbb{R}^D$ is flattened by `ParamSpec` ([`sazz/utils/bnn_utils.py`](../sazz/utils/bnn_utils.py)) in `nn.Linear` order: **weights first, bias last** — i.e. $\beta = (\beta_1, \ldots, \beta_p, \beta_0)^\top$ where $\beta_0$ is the intercept. After resampling, the convention is rotated to $(\beta_0, \beta_1, \ldots, \beta_p)$ to match the augmented design matrix $\tilde{X} = [\mathbf{1} \mid X]$.

---

## 2. Linear Regression

**Script:** [`sazz/scripts/linear_regression.py`](../sazz/scripts/linear_regression.py)

### 2.1 Data Model

$$
y_i = \beta_0 + \mathbf{x}_i^\top \boldsymbol{\beta} + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma_\varepsilon^2)
$$

Features are standardised: $\mathbf{x}_i \leftarrow (\mathbf{x}_i - \bar{\mathbf{x}}) / \text{std}(\mathbf{x})$ before model fitting.

### 2.2 Prior

Diagonal Gaussian prior with separate scales for weights and the intercept:

$$
p(\beta_j) = \mathcal{N}(0,\, \sigma_w^2), \quad j = 1,\ldots,p
$$
$$
p(\beta_0) = \mathcal{N}(0,\, \sigma_b^2)
$$

The precision vector $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_p, \lambda_0)$ stored by `ModuleGaussianPrior` ([`sazz/models/priors.py`](../sazz/models/priors.py)) has entries

$$
\lambda_j = \frac{1}{\sigma_w^2}, \qquad \lambda_0 = \frac{1}{\sigma_b^2}.
$$

**Defaults (`Config`):**

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $\sigma_w$ | `prior_std` | $1.0$ |
| $\sigma_b$ | `bias_prior_std` | $10.0$ |

The prior log-probability is $\log p(\beta) = -\tfrac{1}{2} \boldsymbol{\beta}^\top \Lambda \boldsymbol{\beta}$ (up to a constant).

### 2.3 Likelihood

$$
\log p(\mathbf{y} \mid \beta) = -\frac{N}{2}\log(2\pi\sigma_\varepsilon^2) - \frac{1}{2\sigma_\varepsilon^2} \sum_{i=1}^N (y_i - \beta_0 - \mathbf{x}_i^\top\boldsymbol{\beta})^2
$$

Implemented by `ModuleGaussianLikelihood` ([`sazz/models/likelihoods.py`](../sazz/models/likelihoods.py)) with **fixed** noise.

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $\sigma_\varepsilon$ | `lik_noise_std` | $0.5$ |

### 2.4 Closed-Form Posterior

Because the prior and likelihood are both Gaussian, the posterior is exactly Gaussian:

$$
p(\beta \mid \mathbf{y}) = \mathcal{N}\!\left(\mu_*, \Sigma_*\right), \quad \Sigma_*^{-1} = \Lambda + \frac{1}{\sigma_\varepsilon^2}\tilde{X}^\top\tilde{X}, \quad \mu_* = \Sigma_* \frac{1}{\sigma_\varepsilon^2}\tilde{X}^\top\mathbf{y}
$$

NUTS is run against this same model as a gold-standard reference.

### 2.5 Sampler Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `n_skel` | $10{,}000$ | Skeleton (event) count for PDMP |
| `n_resample` | $5{,}000$ | Samples drawn from skeleton path |
| `burnin_frac` | $0.5$ | Fraction of path discarded as burn-in |
| `refresh_rate` $\lambda_r$ | $1.0$ | Boomerang Poisson refresh rate |
| `kappa_null` | $0.4$ | Kappa for weight coordinates (sticky only) |
| `kappa_int` | $10^6$ | Kappa for intercept (never frozen) |
| `t_max_zz` | $0.1$ | ZigZag upper bound on event-time draws |
| `gamma_zz` | $0.01$ | ZigZag linear expansion coefficient |
| `thinning` | `"pli"` | PLI or Brent root-finder for event times |

---

## 3. Logistic Regression

**Script:** [`sazz/scripts/logistic_regression.py`](../sazz/scripts/logistic_regression.py)

### 3.1 Data Model

$$
y_i \mid \mathbf{x}_i, \beta \sim \text{Bernoulli}\!\left(\sigma(\beta_0 + \mathbf{x}_i^\top\boldsymbol{\beta})\right), \quad \sigma(z) = \frac{1}{1+e^{-z}}
$$

### 3.2 Prior

Same diagonal Gaussian as linear regression.

**Defaults (`Config`):**

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $\sigma_w$ | `prior_std` | $1.0$ |
| $\sigma_b$ | `bias_prior_std` | $10.0$ |

### 3.3 Likelihood

$$
\log p(\mathbf{y} \mid \beta) = \sum_{i=1}^N \left[ y_i \log \sigma(f_i) + (1-y_i)\log(1-\sigma(f_i)) \right], \quad f_i = \beta_0 + \mathbf{x}_i^\top\boldsymbol{\beta}
$$

Implemented by `ModuleBernoulliLikelihood` ([`sazz/models/likelihoods.py`](../sazz/models/likelihoods.py)); logits are passed directly to `torch.distributions.Bernoulli`.

No analytic posterior — NUTS and PDMPs all target the true unnormalised posterior.

### 3.4 Reference Point

Because the posterior is not Gaussian, `find_reference_glm` ([`sazz/utils/warmup.py`](../sazz/utils/warmup.py)) finds the MAP via Adam ($1000$ steps, $\text{lr}=0.01$) and forms the **full Hessian** (not diagonal; `diagonal_only=False`) as $\Sigma^{-1}$:

$$
\Sigma^{-1} = \nabla^2 U(\hat\beta_\text{MAP})
$$

This is the Laplace approximation precision matrix, used to scale Boomerang dynamics.

### 3.5 Defaults

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $N$ | `N` | $300$ |
| $p$ | `D` | $8$ |
| $K$ | `n_signals` | $3$ |
| signal magnitude | `signal_scale` | $1.5$ |
| intercept | `intercept_true` | $0.5$ |

Sampler hyperparameters identical to linear regression.

---

## 4. Sparse Linear Regression

**Script:** [`sazz/scripts/sparse_regression.py`](../sazz/scripts/sparse_regression.py)

### 4.1 Data Model

Same as linear regression (§2.1) but with $D = 200$ features and only $K = 5$ nonzero coefficients drawn as $\beta_j \sim \text{signal\_scale} \cdot \mathcal{N}(0,1)$.

### 4.2 Prior Choice

Two options, selected by `--prior`:

**Gaussian (`"Gauss"`):** same diagonal Gaussian as §2.2.

**Student-t (`"StudT"`):** heavy-tailed shrinkage prior via `ModuleStudentTPrior` ([`sazz/models/priors.py`](../sazz/models/priors.py)):

$$
p(\beta_j) = t_\nu(0,\, s^2), \quad j = 1, \ldots, p
$$
$$
p(\beta_0) = \mathcal{N}(0,\, \sigma_b^2)
$$

Log-probability for weight coordinates:

$$
\log p(\beta_j) = -\frac{\nu+1}{2} \log\!\left(1 + \frac{\beta_j^2}{\nu s^2}\right)
$$

The `precision_diag()` used for the reference metric is the marginal variance of $t_\nu(0,s^2)$:

$$
\text{Var}[t_\nu(0,s^2)] = \frac{\nu}{\nu-2} s^2 \quad (\nu > 2), \qquad \lambda_j = \left(\frac{\nu}{\max(\nu-2, 10^{-3})} s^2\right)^{-1}
$$

**Defaults:**

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $\nu$ | `df` (hardcoded) | $3.0$ |
| $s$ | `prior_std` | $1.0$ |
| $\sigma_b$ | `bias_prior_std` | $10.0$ |

### 4.3 Kappa Calibration

Sticky samplers set kappa to $0.4 \approx \tfrac{1}{\sqrt{2\pi}}$ for all weight coordinates and $10^6$ for the intercept (same constants as linear regression). At $D=200$ the sticky samplers are expected to freeze most null coordinates; the primary metrics are therefore `p0_nulls` (fraction of time null coords are at zero) and $F_1$ score from 95% credible intervals.

---

## 5. Clyde Benchmark (g-Prior)

**Script:** [`sazz/scripts/clyde_benchmark.py`](../sazz/scripts/clyde_benchmark.py)

Reproduces Table 2 of Clyde, Ghosh & Littman (2010). Only sticky variants are run; the reference is the exact posterior computed by enumeration over all $2^p$ models.

### 5.1 Data Model

$$
y_i = \alpha + \mathbf{x}_i^\top\boldsymbol{\beta} + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

$X$ is centred column-wise. Column 9 (0-indexed: 8) is correlated with column 2 (index 1):

$$
x_{i,8} = \rho\, x_{i,1} + \sqrt{1-\rho^2}\, z_i, \quad z_i \sim \mathcal{N}(0,1)
$$

True coefficients (paper Table 2):
$$
\boldsymbol{\beta}_\text{true} = (-0.48,\; 8.72,\; -1.76,\; -1.87,\; 0,\; 0,\; 0,\; 0,\; 4.00,\; 0,\; 0,\; 0,\; 0,\; 0,\; 0)
$$

**Defaults:**

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $n$ | `n` | $100$ |
| $p$ | `p` | $15$ |
| $\sigma$ | `sigma` | $1.0$ |
| $g$ | `g` | $100$ ($= n$) |
| $\alpha$ | `alpha_true` | $2.0$ |
| $\rho$ | `rho` | $0.99$ |

### 5.2 Zellner g-Prior (Slab)

The Zellner g-prior places a data-dependent Gaussian on the active coefficients:

$$
\boldsymbol{\beta}_\gamma \mid \gamma \sim \mathcal{N}\!\left(\mathbf{0},\; g\sigma^2 (X_\gamma^\top X_\gamma)^{-1}\right)
$$

For the PDMP sampler we work with the **marginalised-over-$\gamma$** slab (all $p$ weights simultaneously). The slab precision matrix over the full weight vector is:

$$
\Omega_w = \frac{X_c^\top X_c}{g \sigma^2} \in \mathbb{R}^{p \times p}
$$

where $X_c = X - \bar{X}$ (centred). The full $D \times D$ precision matrix (with $D = p + 1$) is

$$
\Omega = \begin{pmatrix} \Omega_w & 0 \\ 0 & \sigma_b^{-2} \end{pmatrix}
$$

stored as a dense tensor by `ModuleGaussianPrior` ([`sazz/models/priors.py`](../sazz/models/priors.py)), which then computes

$$
\log p(\beta) = -\tfrac{1}{2}\, \beta^\top \Omega\, \beta.
$$

The intercept $\alpha$ is given an independent wide Gaussian $\mathcal{N}(0, \sigma_b^2)$ with $\sigma_b = 10$.

### 5.3 Spike-and-Slab via Kappa

The sticky sampler implements the **spike** through the frozen state: coordinate $j$ is at exactly $0$ (frozen) with effective prior probability proportional to the spike weight. The **kappa** parameter converts the prior inclusion probability $w$ and slab scale $\tau$ into a freeze/thaw rate.

For the g-prior, the marginal slab std is

$$
\tau = \sqrt{\frac{g}{n}\,\sigma^2}
$$

(the diagonal entry of $g\sigma^2(X^\top X)^{-1}$ averaged over coordinates). The kappa for each weight coordinate is then

$$
\kappa_j = \frac{w}{1-w} \cdot \frac{1}{\tau\sqrt{2\pi}}
$$

where $w = 0.5$ is the prior inclusion probability. The intercept always has $\kappa_\alpha = 10^6$ (never frozen).

**Defaults:**

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $w$ | `w_prior` | $0.1$ |
| $\sigma_b$ | `bias_prior_std` | $10.0$ |

### 5.4 Exact Posterior (Reference)

The marginal likelihood for model $\gamma$ (subset of active variables) under the g-prior is

$$
\log p(\mathbf{y} \mid \gamma) = \frac{n - p_\gamma - 1}{2} \log(1+g) - \frac{n-1}{2} \log\!\left(1 + g(1 - R^2_\gamma)\right)
$$

where $R^2_\gamma$ is the OLS coefficient of determination on centred data using only $X_\gamma$, and $p_\gamma = |\gamma|$.

Posterior model weights: $P(\gamma \mid \mathbf{y}) \propto e^{\log p(\mathbf{y}\mid\gamma)} P(\gamma)$ (uniform model prior).

Marginal inclusion probabilities:

$$
\pi_j = \sum_{\gamma : j \in \gamma} P(\gamma \mid \mathbf{y})
$$

BMA fitted values (centred):

$$
\hat\mu_\text{BMA} = \bar{y} + \sum_\gamma P(\gamma\mid\mathbf{y}) \cdot \frac{g}{1+g}\, X_\gamma \hat\beta_\gamma^\text{OLS}
$$

Implemented in `exact_g_prior_posterior` ([`sazz/scripts/clyde_benchmark.py`](../sazz/scripts/clyde_benchmark.py)).

### 5.5 Evaluation Metrics

$$
\text{RMSE}(\pi) \times 10^2 = 100\sqrt{\frac{1}{p}\sum_{j=1}^p (\hat\pi_j - \pi_j)^2}
$$

$$
\text{RMSE}(\mu) \times 10^3 = 1000\sqrt{\frac{1}{n}\sum_{i=1}^n (\hat\mu_i - \mu_i^\text{BMA})^2}
$$

averaged over `n_sim = 100` simulated datasets.

---

## 6. Reference Construction (`find_reference_glm`)

**Source:** [`sazz/utils/warmup.py`](../sazz/utils/warmup.py)

For all GLM scripts, the sampler reference is built as follows:

1. **MAP estimate** $\hat\beta$: Adam optimisation of $U(\beta)$, 1000 steps, lr $= 0.01$.

2. **Precision matrix** $\Sigma^{-1}$:
   - `diagonal_only=True` (linear/sparse regression): diagonal entries $[\nabla^2 U(\hat\beta)]_{ii}$.
   - `diagonal_only=False` (logistic regression, Clyde): full Hessian $\nabla^2 U(\hat\beta)$, symmetrised and jittered by $10^{-8} I$.

3. Entries are clamped to $[10^{-6},\; 10^8]$.

The Boomerang sampler uses $(\hat\beta, \Sigma^{-1})$ as its reference for the quadratic term; the ZigZag sampler uses them only for the initial position.

---

## 7. NUTS Baselines

### Gaussian baseline
NUTS is run via PyMC with priors matching the PDMP model exactly. `target_accept = 0.9`, 4 chains.

### Horseshoe baseline (`--nuts-hs`)

Non-centred horseshoe prior replaces the Gaussian slab:

$$
\tau \sim \text{HalfCauchy}(\beta = \sigma_w), \quad \lambda_j \sim \text{HalfCauchy}(\beta=1)
$$
$$
\tilde\beta_j \sim \mathcal{N}(0,1), \quad \beta_j = \tau\,\lambda_j\,\tilde\beta_j
$$

`target_accept = 0.95` (higher because of funnel geometry). Color `"C4"`, marker `"P"`.

---

## 8. Coordinate Convention Summary

| Convention | Where |
|-----------|-------|
| `[weight..., bias]` | ParamSpec flatten order; `nn.Linear` storage; sampler internal |
| `[bias, weight...]` | After `run_pdmps` rotation; NUTS samples; `compute_metrics` |
| Rotation in code | `samples = np.concatenate([samples[:, -1:], samples[:, :-1]], axis=1)` in [`sazz/utils/glm_utils.py:72`](../sazz/utils/glm_utils.py#L72) |

The Boomerang reference `x_ref` and `Sigma_inv` remain in `[weight..., bias]` order internally; the ZigZag uses the rotated `x_ref_rot` as `x0` (see [`clyde_benchmark.py`](../sazz/scripts/clyde_benchmark.py)).
