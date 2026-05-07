# PDMP Sampler Hyperparameters — Choices and Defaults

This document explains the general design rationale behind the key PDMP hyperparameters used across all scripts in this repo. For model-specific tables of default values, see [glm_models.md](glm_models.md) and [bnn_models.md](bnn_models.md).

---

## 1. Kappa (Stickiness)

### What it controls

The kappa parameter $\kappa$ governs how quickly a frozen coordinate can **thaw**. A coordinate frozen at $0$ thaws at rate $\kappa \cdot |v|$ (Boomerang) or $\kappa \cdot 1$ (ZigZag). Larger $\kappa$ means less sticky — the process spends less time at zero.

### Calibration from a spike-and-slab prior

The natural calibration sets $\kappa$ so that the long-run fraction of time spent at zero matches the prior spike weight $1 - w$. For a coordinate $j$ with slab $\mathcal{N}(0, \sigma_j^2)$ and inclusion probability $w$:

$$
\kappa_j = \frac{w}{1 - w} \cdot \frac{1}{\sigma_j\,\sqrt{2\pi}}
$$

**Derivation.** At stationarity the process freezes at rate proportional to the spike weight and thaws at rate $\kappa$ times the slab density at $0$, which is $\mathcal{N}(0;\,0,\sigma_j^2) = ({\sigma_j\sqrt{2\pi}})^{-1}$. Setting the freeze and thaw odds equal to the spike-to-slab odds gives the formula above.

Implemented in `make_kappa_from_inclusion` ([`sazz/utils/bnn_utils.py`](../sazz/utils/bnn_utils.py)).

### GLM default: `kappa_null = 0.4`

For GLM scripts, the prior std is $\sigma_w = 1$ and the inclusion probability is $w = 0.5$ (equal spike/slab odds). Plugging in:

$$
\kappa = \frac{0.5}{0.5} \cdot \frac{1}{1 \cdot \sqrt{2\pi}} = \frac{1}{\sqrt{2\pi}} \approx 0.399
$$

So **kappa = 0.4 is exactly** $1/\sqrt{2\pi}$, the $\mathcal{N}(0,1)$ density at $0$.

### BNN default: calibrated per layer

BNN weights use **fan-in scaling**: $\sigma_w^{(l)} = \sigma_0 / \sqrt{d_l}$ where $d_l$ is the number of inputs to layer $l$ and $\sigma_0$ is `prior_std_weight`. This keeps the prior variance of pre-activations independent of width. The effective kappa for layer $l$ is:

$$
\kappa^{(l)} = \frac{w}{1-w} \cdot \frac{\sqrt{d_l}}{\sigma_0\,\sqrt{2\pi}}
$$

so **wider layers get a larger kappa** (less sticky), which counteracts the smaller slab std. For the Boston UCI dataset (13 inputs, first layer) with $w = 0.7$ and $\sigma_0 = 1$:

$$
\kappa \approx \frac{0.7}{0.3} \cdot \frac{\sqrt{13}}{\sqrt{2\pi}} \approx 3.36
$$

### Intercept / bias kappa: $10^6$

Biases never have a spike at zero from a modelling standpoint — there is no reason to enforce $\beta_0 = 0$. The value $\kappa = 10^6$ is effectively infinite: the coordinate thaws instantaneously and never stays frozen. In practice, bias coordinates also carry `can_freeze=False` so they bypass the sticky mechanism entirely.

---

## 2. Refreshment Rate ($\lambda_r$)

### What it controls

The Boomerang sampler adds **Poisson refreshments** at rate $\lambda_r$: at each refresh event the velocity is redrawn from the reference Gaussian $\mathcal{N}(0, \Sigma)$. This breaks the periodicity of the deterministic orbits and ensures ergodicity.

### Geometric floor: $\lambda_r \geq 1/\pi$

The Boomerang orbit has period $2\pi$ in the reference-measure geometry. If $\lambda_r$ is too small, the trajectory can complete a near-half-orbit between refreshments, meaning two successive bounces at roughly opposite points on the orbit can approximately cancel out and the sampler degenerates toward cycling around the reference. The floor $1/\pi$ ensures at least one expected refresh per half-orbit.

### Adaptive tuning: BPS-optimal ratio

The default initial value $\lambda_r = 1.0$ is a safe starting point for most problems. In the BNN scripts, `tune_refresh_rate` ([`sazz/utils/warmup.py`](../sazz/utils/warmup.py)) adapts it via a pilot run:

$$
\lambda_r = \max\!\left(\frac{\rho}{1-\rho}\,\hat\lambda_\text{bounce},\;\frac{1}{\pi}\right), \qquad \rho = 0.7812
$$

where $\hat\lambda_\text{bounce}$ is the empirical bounce rate from the pilot. The ratio $\rho = 0.7812$ is the BPS-optimal refresh fraction from Bouchard-Côté et al. (2018), borrowed as a heuristic: it targets $\approx 78\%$ of events being refreshments. This is more conservative than strictly necessary but yields stable mixing across a range of problems.

### ZigZag: no refreshment rate

The ZigZag sampler does not use a refreshment rate. Instead it has a **uniform excess switching rate** $\gamma$ (see §4) added to every coordinate for irreducibility.

---

## 3. Reference Measure ($x_\text{ref}$, $\Sigma^{-1}$)

### Role

The Boomerang sampler orbits around a **reference point** $x_\text{ref}$ with **reference precision** $\Sigma^{-1}$. The reference measure $\mathcal{N}(x_\text{ref}, \Sigma)$ acts as a proposal: the closer it is to the posterior, the fewer bounces are needed to correct for the discrepancy, and the longer the trajectory segments between events.

### GLM: full Laplace approximation

For GLMs, the posterior is either exactly Gaussian (linear regression) or well-approximated by one (logistic regression). `find_reference_glm` uses the **full Hessian** of $-\log p(\beta \mid y)$ at the MAP:

$$
\Sigma^{-1} = \nabla^2 U(\hat\beta_\text{MAP})
$$

This is the exact posterior precision for linear regression, and the Laplace precision for logistic regression.

### BNN: diagonal Laplace + pilot refinement

For BNNs the Hessian is too large to invert exactly. `find_reference_bnn` uses a **diagonal Laplace** approximation:

$$
\Sigma^{-1}_{ii} = \lambda_i + \hat{F}_{ii}
$$

where $\lambda_i$ is the prior precision and $\hat F_{ii}$ is the diagonal empirical Fisher. This is always positive definite (prior precision regularises it) and cheap to compute.

The reference is then **refined** for the Boomerang sampler only: after the diagonal initialisation, for three rounds of pilot sampling, the block covariance is recomputed from pilot samples. This will then into the reference $\Sigma^{-1}_{ii}$.

### Reference as geometry

A well-chosen reference acts as a **preconditioning**: $\Sigma^{-1}$ maps the posterior to something close to a standard Gaussian, so the sampler operates in a well-conditioned space. A poor reference (e.g., using only the prior) means large curvature differences across coordinates and slower mixing.

---

## 4. ZigZag-specific Parameters

### `t_max` (upper bound on event-time draws)

The ZigZag uses PLI (piecewise linear interpolation) or Brent root-finding to sample event times. `t_max` caps the search interval for both methods. It acts as a **horizon** per iteration: if no bounce is found within $[0, t_\text{max}]$ the particle travels freely for $t_\text{max}$ before the next attempt. The default $0.1$ works well when the posterior is preconditioned by the reference; it may need adjustment for very flat or very sharp posteriors.

### `gamma` (excess switching rate)

Every ZigZag coordinate has a minimum switching rate of $\gamma > 0$ added uniformly. This **linear excess** prevents the sampler from getting stuck if the gradient is exactly zero along some direction, ensuring irreducibility. The default $\gamma = 0.01$ is small enough to be negligible for typical posteriors but nonzero enough to guarantee correctness.

---

## 5. Skeleton Size and Burn-in

### `n_skel`

The number of **skeleton events** (bounces, refreshments, freeze/thaw) collected. This is the primary cost driver: each event requires a gradient evaluation. Larger `n_skel` gives better-mixed chains at the cost of compute. Typical values:

| Setting | `n_skel` |
|---------|----------|
| GLM scripts | $10{,}000$ |
| BNN UCI | $100{,}000$ |

### `burnin_frac`

Fraction of the skeleton path discarded as burn-in. The default is $0.5$ for GLM scripts and $0.2$ for BNN scripts. The BNN scripts use a lower fraction because the warmup procedure (MAP + pilot tuning) already places the chain near the posterior before `n_skel` events are collected.

### `n_resample`

The skeleton path is a continuous piecewise trajectory; `n_resample` independent samples are drawn from it via PLI after burn-in. The resample count should be large enough to represent the posterior well but is decoupled from compute cost (resampling is cheap).

---

## 6. Summary Table

| Parameter | Default (GLM) | Default (BNN) | Rationale |
|-----------|--------------|---------------|-----------|
| `kappa_null` | $0.4 = 1/\sqrt{2\pi}$ | per layer (fan-in) | Spike-and-slab calibration at $\sigma=1$, $w=0.5$ |
| `kappa_int` / bias | $10^6$ | $10^6$ | No spike on intercept/bias |
| `refresh_rate` $\lambda_r$ | $1.0$ | tuned (floor $1/\pi$) | BPS-optimal ratio; floor prevents orbit cancellation |
| reference | full Laplace | diag Laplace + pilot | Best available Gaussian approximation to posterior |
| `t_max_zz` | $0.1$ | $0.1$ | Horizon cap in preconditioned space |
| `gamma_zz` | $0.01$ | $0.01$ | Irreducibility floor for ZigZag |
| `burnin_frac` | $0.5$ | $0.2$ | Lower for BNN because warmup pre-heats the chain |
