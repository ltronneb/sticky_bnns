"""Gradients per event under a constant rate, single segment (v2) vs grid (v1).

Only the bookkeeping of the two thinning schemes is simulated, no PDMP. The rate
is lambda = 1 (everything depends on lambda * h only), the bound is the exact
rate times 1 + eps, and a proposal is accepted with probability 1 / (1 + eps).

  single segment  window build costs 2, or 1 when the left end is cached (the
                  previous window was empty), plus 1 per proposal and 1 for the
                  bounce after an accepted proposal.
  grid            window build costs K + 1, or K when the left end is cached,
                  plus 1 per proposal and 1 for the bounce. (The real grid runs
                  and Algorithm 4 of Andral and Kamatani never cache.)

Both bounds can adapt t_max with either rule.
  balanced        t_max is multiplied by alpha^(empty - rejections) (v2).
  alg4            t_max is divided by alpha_minus after a window with any
                  rejection, else multiplied by alpha_plus after an empty
                  window (v1, labelled Algorithm 4 in the code).

    python notes/constant_rate_cost/sim.py
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPS = 0.01                                      # bound inflation, both versions
ALPHA = 1.01                                    # balanced rule (v2)
ALPHA_PLUS, ALPHA_MINUS = 1.01, 1.04            # alg4 rule (v1)
rng = np.random.default_rng(0)


def window(h: float):
    """One thinning window of length h. Returns (accepted, proposals, rejections)."""
    t, props, rej = 0.0, 0, 0
    while True:
        t += rng.exponential(1.0 / (1.0 + EPS))
        if t >= h:
            return False, props, rej
        props += 1
        if rng.random() < 1.0 / (1.0 + EPS):
            return True, props, rej
        rej += 1


def adapt(h: float, acc: bool, rej: int, rule: str) -> float:
    if rule == "balanced":
        return h * ALPHA ** (int(not acc) - rej)
    if rule == "alg4":
        return h / ALPHA_MINUS if rej else (h * ALPHA_PLUS if not acc else h)
    return h                                    # rule None, fixed window


def single_segment(n_events: int, h: float, rule=None):
    grads = events = windows = empties = 0
    cached = False
    while events < n_events:
        acc, props, rej = window(h)
        grads += (1 if cached else 2) + props + acc          # acc = the bounce
        windows += 1
        events += acc
        empties += not acc
        cached = not acc
        h = adapt(h, acc, rej, rule)
    return grads / events, empties / windows, h


def grid(n_events: int, h: float, K: int, rule=None):
    grads = events = windows = empties = 0
    cached = False
    while events < n_events:
        acc, props, rej = window(h)
        grads += (K if cached else K + 1) + props + acc
        windows += 1
        events += acc
        empties += not acc
        cached = not acc
        h = adapt(h, acc, rej, rule)
    return grads / events, empties / windows, h


# ------------------------------------------------------------- formulas
def c_ss(lh):
    return 3 + EPS + 1 / (1 - np.exp(-lh))


def c_grid(lh, K):
    return K / (1 - np.exp(-lh)) + 3 + EPS


R = math.log(ALPHA_MINUS) / math.log(ALPHA_PLUS)
Q_STAR = {"balanced": EPS / (1 + EPS),          # empty-window share at equilibrium,
          "alg4": R * EPS / (1 + R * EPS)}      # the same for both bounds (alg4 for small eps)


def main():
    out = Path(__file__).parent
    n = 50_000
    lh = np.array([0.25, 0.5, 1, 2, 3, 5, 8])
    KS = [1, 2, 4, 8, 16, 32]

    print("1. fixed window, simulated vs formula (gradients per event)")
    print(f"{'lambda h':>9}{'SS sim':>9}{'SS formula':>12}" + "".join(f"{f'K={K} sim':>10}{'formula':>9}" for K in KS))
    sim_ss = [single_segment(n, x)[0] for x in lh]
    sim_g = {K: [grid(n, x, K)[0] for x in lh] for K in KS}
    for i, x in enumerate(lh):
        row = f"{x:>9.2f}{sim_ss[i]:>9.3f}{c_ss(x):>12.3f}"
        row += "".join(f"{sim_g[K][i]:>10.3f}{c_grid(x, K):>9.3f}" for K in KS)
        print(row)

    eq = {}
    for rule in ("balanced", "alg4"):
        q = Q_STAR[rule]
        print(f"\n2. adaptive t_max, rule = {rule}, the same rule for both bounds (start at lambda h = 0.1)")
        c, qs, h = single_segment(4 * n, 0.1, rule)
        print(f"  single segment  C = {c:.3f} (formula {c_ss(-math.log(q)):.3f})   empty share {qs:.4f} "
              f"(formula {q:.4f})   lambda h at end {h:.2f} (formula {-math.log(q):.2f})")
        eq[rule] = {"ss": c}
        for K in KS:
            c_g, q_g, _ = grid(4 * n, 0.1, K, rule)
            eq[rule][K] = c_g
            print(f"  grid K = {K:<3}    C = {c_g:.3f} (formula {c_grid(-math.log(q), K):.3f})   empty share {q_g:.4f} "
                  f"(formula {q:.4f})   ratio grid / SS {c_g / c:.2f} (about (K + 3) / 4 = {(K + 3) / 4:.2f})")

    # figure
    # one panel, sized to sit next to the Boston bar chart in the paper
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    xs = np.linspace(0.15, 9, 300)
    K_PLOT = [K for K in KS if K > 1]           # grid with K = 1 is the single segment itself
    blues = plt.cm.Blues(np.linspace(0.45, 0.95, len(K_PLOT)))
    ax.plot(xs, c_ss(xs), color="0.1", lw=2, label="Single segment")
    ax.plot(lh, sim_ss, "o", color="0.1", ms=4)
    for K, col in zip(K_PLOT, blues):
        ax.plot(xs, c_grid(xs, K), color=col, lw=1.6, label=f"Grid, $K = {K}$")
        ax.plot(lh, sim_g[K], "o", color=col, ms=4)
    ax.axvline(-math.log(Q_STAR["balanced"]), color="0.4", ls=":", lw=1)
    ax.text(-math.log(Q_STAR["balanced"]) + 0.12, 110, "Adapted window", fontsize=7.5, color="0.3")
    ax.set(yscale="log", xlabel=r"window length $\lambda t_{\max}$", ylabel="Gradients per event", xlim=(0, 9))
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"constant_rate_cost.{ext}", dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {out / 'constant_rate_cost.pdf'}")


if __name__ == "__main__":
    main()
