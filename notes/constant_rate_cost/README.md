# Grid vs single segment, gradient evaluations per event

Simulation in `sim.py` (`python notes/constant_rate_cost/sim.py`, about 4 s).

## 1. Cost of one window

Every rate evaluation costs one gradient. A window first builds its bound, then proposes times from it. Each proposal costs one evaluation of the true rate. A rejection continues in the same window with the same bound, an accepted proposal costs one more gradient for the bounce and ends the window, and a window with no accepted proposal ends at its right end.

| window | grid, $K$ segments | single segment |
|---|---|---|
| $r$ rejections, then an accept | $K + 3 + r$ | $4 + r$, or $3 + r$ if cached |
| $r$ rejections, no accept | $K + 1 + r$ | $2 + r$, or $1 + r$ if cached |

The grid builds the bound from $K+1$ nodes. Single segment evaluates the two ends, and when the previous window ended without an event, its right end is the new left end (cached), so only 1 evaluation is needed.

## 2. Expected cost per event, constant rate

Let the rate be a constant $\lambda$ and the window have length $t_{\max}$. The bounds are then exact up to the inflation $\varepsilon = 0.01$, so proposals arrive at rate $(1+\varepsilon)\lambda$ and each is accepted with probability $1/(1+\varepsilon)$. Accepted proposals form a Poisson process of rate $\lambda$ and rejections an independent one of rate $\varepsilon\lambda$. With $\tau \sim \mathrm{Exp}(\lambda)$ the time of the first accepted proposal,

$$
P(\text{accept}) = P(\tau < t_{\max}) = 1 - e^{-\lambda t_{\max}}, \qquad P(\text{empty}) = e^{-\lambda t_{\max}} .
$$

Write $q = e^{-\lambda t_{\max}}$. The expected cost of a window is the cost with an accept times $P(\text{accept})$ plus the cost without times $P(\text{empty})$.

**Grid.**

$$
\mathbb{E}[G_{\mathrm{grid}}] = (K + 3)(1-q) + (K + 1)\,q + \mathbb{E}[r] = K + 1 + 2(1-q) + \mathbb{E}[r].
$$

**Single segment.** The previous window was empty with probability $q$. The expected number of events in this window is then $q(1\cdot q + 3\cdot(1-q))$, since the probability that this window is empty is also $q$. If the previous window was non-empty, the expected number of events in this window is $(1-q)(2\cdot q + 4(1-q))$.
$$
\mathbb{E}[G_{\mathrm{ss}}] = q(1\cdot q + 3\cdot(1-q)) + (1-q)(2\cdot q + 4(1-q)) + \mathbb{E}[r] \\
= 4-3q + \mathbb{E}[r]
$$

A window holds at most one event, and it holds one with probability $1-q$, so the cost per event is

$$
C = \frac{\mathbb{E}[G]}{1-q}.
$$

## 3. Expected number of rejections

Rejections arrive at rate $\varepsilon\lambda$ along the distance travelled. With an accept the window is travelled up to $\tau$, without one up to $t_{\max}$, so

$$
\mathbb{E}[r] = \varepsilon\lambda\,\mathbb{E}[\tau \mid \tau < t_{\max}]\,(1-q) + \varepsilon\lambda\,t_{\max}\,q .
$$

For the truncated exponential

$$
\mathbb{E}[\tau \mid \tau < t_{\max}] = \frac{1}{\lambda} - \frac{t_{\max}\,q}{1-q},
$$

and the $t_{\max} q$ terms cancel,

$$
\mathbb{E}[r] = \varepsilon\,(1-q) = \varepsilon\,\big(1 - e^{-\lambda t_{\max}}\big).
$$

## 4. Cost per event

Inserting $\mathbb{E}[r]$ into section 2,

$$
C_{\mathrm{grid}} = \frac{K+1}{1-e^{-\lambda t_{\max}}} + 2 + \varepsilon ,
\qquad
C_{\mathrm{ss}} = \frac{2-q}{1-q} + 2 + \varepsilon = \frac{1}{1-e^{-\lambda t_{\max}}} + 3 + \varepsilon .
$$

For long windows, $\lambda t_{\max} \to \infty$, these approach $K + 3 + \varepsilon$ and $4 + \varepsilon$. For short windows both grow like $1/(\lambda t_{\max})$, the grid $K+1$ times faster.

## 5. The adapted window, balanced rule for both bounds

Both bounds adapt $t_{\max}$ with the balanced rule. After every window

$$
t_{\max} \leftarrow t_{\max}\,\alpha^{\mathbb{1}\{\text{empty}\} - r}, \qquad \alpha = 1.01 .
$$

The expected change of $\log t_{\max}$ per window is $\log\alpha\,\big(P(\text{empty}) - \mathbb{E}[r]\big)$, so the window settles where $P(\text{empty}) = \mathbb{E}[r]$. With a constant rate both bounds are exact up to $\varepsilon$, so $\mathbb{E}[r] = \varepsilon(1-q)$ for both (section 3), and both settle at the same window,

$$
q^\ast = \varepsilon\,(1-q^\ast) \;\Rightarrow\; q^\ast = \frac{\varepsilon}{1+\varepsilon} \approx 0.0099, \qquad \lambda t_{\max}^\ast = \log\frac{1+\varepsilon}{\varepsilon} \approx 4.6 .
$$

Inserting $1/(1-q^\ast) = 1+\varepsilon$ into section 4,

$$
C_{\mathrm{ss}}^\ast = (1+\varepsilon) + 3 + \varepsilon = 4 + 2\varepsilon \approx 4.02,
\qquad
C_{\mathrm{grid}}^\ast = (K+1)(1+\varepsilon) + 2 + \varepsilon \approx K + 3 ,
$$

$$
\frac{C_{\mathrm{grid}}^\ast}{C_{\mathrm{ss}}^\ast} = \frac{(K+1)(1+\varepsilon) + 2 + \varepsilon}{4 + 2\varepsilon} \approx \frac{K+3}{4}.
$$

Under the same rule the gain comes from the bound alone, $K+1$ nodes against $2$ and the cached end. The rule decides only the window, and the window changes the costs through $1/(1-q^\ast)$, which is close to 1 for any sensible rule.

In v1 the number of segments follows the window, $K = \min\big(\max(\lceil t_{\max}/\Delta \rceil, 2), K_{\max}\big)$ with grid spacing $\Delta$, so a longer adapted window also means a larger $K$.

## 6. Simulation check

The simulation draws the proposals of each window and counts gradients exactly as in section 1, with $\lambda = 1$.

| $\lambda t_{\max}$ | single segment, sim | formula | grid $K=4$, sim | formula |
|---|---|---|---|---|
| 0.5 | 5.53 | 5.55 | 14.72 | 14.72 |
| 1 | 4.59 | 4.59 | 9.96 | 9.92 |
| 3 | 4.06 | 4.06 | 7.27 | 7.27 |
| 8 | 4.01 | 4.01 | 7.01 | 7.01 |

The adaptive runs start at $\lambda t_{\max} = 0.1$ and use the same rule for both bounds, first the **balanced** rule of section 5, then for comparison the **alg4** rule of the v1 code ($t_{\max}$ divided by $\alpha_- = 1.04$ after a window with any rejection, otherwise multiplied by $\alpha_+ = 1.01$ after an empty window, which settles at $q^\ast \approx 0.038$ and $\lambda t_{\max}^\ast \approx 3.3$).

| rule | single segment $C$ | grid $C$, $K = 1, 2, 4, 7, 10, 22$ | gain grid / single segment |
|---|---|---|---|
| balanced | 4.02 (formula 4.02) | 4.03, 5.05, 7.07, 10.11, 13.14, 25.28 | 1.00, 1.25, 1.76, 2.51, 3.27, 6.29 |
| alg4 | 4.05 (formula 4.05) | 4.09, 5.14, 7.23, 10.36, 13.47, 26.00 | 1.01, 1.27, 1.78, 2.56, 3.32, 6.42 |

$(K+3)/4$ gives $1.00, 1.25, 1.75, 2.50, 3.25, 6.25$. The two rules give nearly the same gain, as section 5 predicts. The empty shares come out slightly above $q^\ast$ because of the start-up transient.

![cost](constant_rate_cost.png)

## 7. Freeze and thaw windows (sticky samplers)

A freeze or thaw ends a window cut short at the hitting or thawing time, usually with no proposal in it and no bounce. Single segment pays the build, $2$. The grid pays $K+1$ with $K = 2$ at the minimum, $3$. So per freeze or thaw

$$
\frac{C_{\mathrm{grid}}}{C_{\mathrm{ss}}} \approx \frac{3}{2}.
$$

With a share $\varphi$ of freeze and thaw events,

$$
\frac{C_{\mathrm{grid}}}{C_{\mathrm{ss}}} \approx \frac{(1-\varphi)(K+3) + 3\varphi}{4(1-\varphi) + 2\varphi}.
$$

## 8. To check against the data (Status step 13)

The observed small network gains are $1.84$ (ZigZag) and $1.66$ (Boomerang), which fit $K \approx 3$ to $4$ in $(K+3)/4$. Sticky Boomerang, where freezes and thaws are 86 to 97 % of events, gives $1.55$ to $1.59$ for $\varphi = 0.9$ and $K = 4$ to $5$, against the observed $1.58$.

Two things the model leaves out, both visible in the rate plot of `uci_skeletons.ipynb` (section 8).

* The ZigZag bound bounds every coordinate separately and sums, so it sits above the rate even when the rate is flat. Along the small network's paths the rate fills about 92 % of the single-segment bound, an effective $\varepsilon \approx 0.09$ rather than $0.01$. Then $q^\ast \approx 0.08$ and $\lambda t_{\max}^\ast \approx 2.5$, closer to the $2.8$ to $3.4$ seen in the plot than the $4.6$ above.
* The sticky rates curve inside a window, so the bounds are looser than the constant-rate case.

Steps

1. Read the real $K$ and $\texttt{rate\_evals}$ per window from the v1 diagnostics (sparsity ablation `*_skeleton.pt` in `results/paper`) and plug them in.
2. Check the floors of 4 per bounce and 2 per freeze or thaw against the v2 event mix in the skeleton notebook.
3. Redo section 5 with the effective $\varepsilon$ from the rate plot.



It follows from asking when t_max stops drifting.

1. What one window does to t_max. The balanced rule multiplies t_max by α when the window is empty and divides by α for each rejection:

$$
\log t_{\max} ;\leftarrow; \log t_{\max} + \log\alpha,\big(\mathbb{1}{\text{empty}} - r\big).
$$

2. The average change per window. Taking expectations with $P(\text{empty}) = q$ and, from section 3, $\mathbb{E}[r] = \varepsilon(1-q)$:

$$
\mathbb{E}\big[\Delta \log t_{\max}\big] = \log\alpha,\big(q - \varepsilon(1-q)\big).
$$

3. The sign of the drift. Since $q = e^{-\lambda t_{\max}}$ falls as $t_{\max}$ grows:

If $t_{\max}$ is too small, windows are often empty, $q$ is close to 1, the drift is positive, and $t_{\max}$ grows.
If $t_{\max}$ is too large, windows almost always contain an event, $q$ is close to 0, and the drift is about $-\varepsilon\log\alpha < 0$, so $t_{\max}$ shrinks.
So $t_{\max}$ is pushed towards the point where the drift is zero, and that point is stable.

4. Setting the drift to zero:

$$
q = \varepsilon(1-q) \quad\Longleftrightarrow\quad q^\ast = \frac{\varepsilon}{1+\varepsilon},
\qquad
\lambda t_{\max}^\ast = -\log q^\ast = \log\frac{1+\varepsilon}{\varepsilon} \approx \log 101 \approx 4.6 .
$$

In words, the rule settles where an empty window is as frequent as a rejection. Each of them pushes $t_{\max}$ by the same factor $\alpha$, in opposite directions.

Two caveats:

It's an equilibrium in the average, not a fixed value. $t_{\max}$ keeps fluctuating around $t_{\max}^\ast$ by factors of α. The simulation confirms it: it settles at an empty share of 0.0117 against the predicted 0.0099, with the small excess coming from the start-up transient, and at λt_max ≈ 4.4 to 4.5 against 4.6.
The same argument gives the grid rule's equilibrium, with different weights. There, each window with any rejection shrinks $t_{\max}$ by $\alpha_- = 1.04$ and each empty one grows it by $\alpha_+ = 1.01$. The drift is zero where $q \log\alpha_+ = P(\text{any rejection}) \log\alpha_-$, which gives $q^\ast \approx \rho\varepsilon/(1+\rho\varepsilon)$ with $\rho = \log\alpha_-/\log\alpha_+ \approx 3.94$.