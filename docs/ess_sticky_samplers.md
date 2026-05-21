# ESS for sticky samplers

## Traditional ESS

For a Markov chain $(\beta^{(1)}, \dots, \beta^{(S)})$ targeting a continuous distribution $\pi$, the effective sample size is defined as

$$
\mathrm{ESS} = \frac{S}{1 + 2\sum_{k=1}^{\infty} \rho_k},
$$

where $\rho_k = \mathrm{Corr}(\beta^{(t)}, \beta^{(t+k)})$ is the lag-$k$ autocorrelation. Intuitively, ESS counts how many i.i.d. draws from $\pi$ would yield the same Monte Carlo variance as the $S$ correlated draws. In practice the sum is truncated using the Geyer initial positive-sequence (IPS) rule.

## The difficulty for sticky samplers

Sticky samplers target a *mixed measure*

$$
\pi(d\beta) = p_0 \,\delta_0(d\beta) + (1 - p_0)\, \pi_{\mathrm{slab}}(\beta)\, d\beta,
$$

where $\delta_0$ is a point mass at zero and $\pi_{\mathrm{slab}}$ is an absolutely continuous slab component. A well-functioning sticky sampler will park at $\beta = 0$ for extended consecutive steps whenever the posterior favours the spike. This produces long runs of identical values, which standard autocorrelation estimators interpret as severe mixing failure — even though the behaviour is correct.

Applying IPS-ESS directly to such a chain is therefore misleading in both directions:

- **For null coordinates** (true $\beta = 0$): the chain is mostly zero, so $\rho_k \approx 1$ for many lags and ESS $\approx 0$, yet the sampler may be exploring the spike-slab boundary perfectly well.
- **For signal coordinates** (true $\beta \neq 0$): the chain occasionally visits zero even for a signal, inflating the apparent variance and distorting $\rho_k$.

## What we report instead

We decompose the problem into two separately estimable quantities:

1. **Indicator ESS.** Apply IPS-ESS to the binary sequence $\mathbf{1}[\beta^{(t)} \neq 0]$. This measures how efficiently the chain explores spike vs. slab assignment and is well-defined under standard autocorrelation theory.

2. **Active (slab) ESS.** Extract the subsequence of non-zero draws and apply IPS-ESS to that. This measures mixing within the slab conditional $\pi_{\mathrm{slab}}$, ignoring the time spent at zero.

Neither quantity alone is a complete efficiency summary. The indicator ESS captures selection mixing; the active ESS captures slab mixing; neither accounts for the interplay between the two.

## Open question

A more principled single-number summary might be based on *excursion counts*: each contiguous block of non-zero samples constitutes one slab excursion, and the number of distinct excursions upper-bounds the number of independent slab draws. This connects to regeneration-based ESS theory (Mykland, Tierney & Yu, 1995), where returns to a recurrent atom yield exact i.i.d. refreshes. For sticky samplers the zero set is a natural candidate atom. Working this out carefully for PDMP sticky samplers remains an open problem.
