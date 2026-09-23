"""
Aggregate sticky_exactness_linreg.py across seeds: is the PDMP-vs-Gibbs gap
NOISE or BIAS?

One seed cannot answer that. A single coordinate sitting 0.05 below the exact
inclusion probability is either (a) finite-sample mixing on a hard coordinate,
or (b) a systematic defect in the freeze/thaw balance. The discriminator is the
SIGN of the deviation across independent seeds: noise flips sign, bias does not.

For each sampler this reports, over all coordinates and seeds, the mean SIGNED
deviation from Gibbs with a standard error. Under exactness the mean signed
deviation is 0, so |mean| / SE is a z-score -- |z| under ~2 is consistent with
exactness, a large |z| with one sign is evidence of bias.

Usage:
    python -m sazz.gpu_friendly.scripts.sticky_exactness_summary
    python -m sazz.gpu_friendly.scripts.sticky_exactness_summary --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

OUT = Path("results/paper/sticky_exactness")
ZERO_TOL = 1e-10
KEYS = [("sticky_zigzag", "Sticky ZigZag"), ("sticky_boomerang", "Sticky Boomerang")]


def incl_prob(draws: torch.Tensor) -> torch.Tensor:
    return (draws.abs() > ZERO_TOL).to(torch.float64).mean(dim=0)

def ess_batch_means(x: np.ndarray) -> float:
    """ESS of a 0/1 trace via non-overlapping batch means."""
    n = x.size
    b = max(1, int(n ** 0.5))
    nb = n // b
    if nb < 2:
        return float(n)
    bm = x[: nb * b].reshape(nb, b).mean(axis=1)
    var_bm = bm.var(ddof=1)
    if var_bm <= 0:
        return float(n)
    return float(n * x.var(ddof=1) / (b * var_bm))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--D", type=int, default=20)
    ap.add_argument("--w", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    dev = {k: [] for k, _ in KEYS}      # signed per-coordinate deviations
    pmap = {k: [] for k, _ in KEYS}     # sampler inclusion probs (for the figure)
    pref_all = []                        # exact inclusion probs (for the figure)
    # Per-coordinate posterior mean and sd, per seed, for the coefficient
    # panel. "gibbs" is included here (unlike dev/pmap) because that panel
    # shows the exact posterior as a series in its own right, not as a
    # reference to deviate from.
    post_mean = {k: [] for k, _ in KEYS}
    post_sd = {k: [] for k, _ in KEYS}
    post_mean["gibbs"] = []
    post_sd["gibbs"] = []
    beta_true = None
    spars = {k: [] for k, _ in KEYS}
    spars["gibbs"] = []
    c_theory = {k: [] for k, _ in KEYS}
    bv = {k: 0 for k, _ in KEYS}
    hard = []                            # (seed, coord, p_ref, p_zz, p_bm)
    seeds_found = []
    n_gibbs = None

    for s in args.seeds:
        p = args.out / f"linreg_D{args.D}_w{args.w}_seed{s}.pt"
        if not p.exists():
            print(f"  (missing seed {s}: {p})")
            continue
        r = torch.load(p, weights_only=False)
        seeds_found.append(s)
        p_ref = incl_prob(r["gibbs"]["draws"])
        pref_all.append(p_ref.numpy())
        n_gibbs = r["gibbs"]["draws"].shape[0]
        spars["gibbs"].append(float((r["gibbs"]["draws"].abs() <= ZERO_TOL).to(torch.float64).mean()))
        # beta_true is generated from D/n_signals/signal_scale alone (no seed
        # dependence -- only X, y and the noise differ per seed), so every
        # seed carries the same vector and coordinates are comparable across
        # seeds. Assert rather than assume, since pooling in the coefficient
        # panel is only meaningful if that holds.
        bt = r["beta_true"].numpy()
        if beta_true is None:
            beta_true = bt
        elif not np.allclose(beta_true, bt):
            raise ValueError(
                f"beta_true differs between seeds (first vs seed {s}); the "
                f"coefficient panel pools coordinates across seeds and that "
                f"is only valid when the truth is shared."
            )
        post_mean["gibbs"].append(r["gibbs"]["draws"].mean(dim=0).numpy())
        post_sd["gibbs"].append(r["gibbs"]["draws"].std(dim=0).numpy())
                
        g_ind = (r["gibbs"]["draws"].abs() > ZERO_TOL).numpy().astype(float)   # [n, D]
        ess_g = np.array([ess_batch_means(g_ind[:, i]) for i in range(args.D)])
        for k, _ in KEYS:
            d = r[k]["draws"]
            dev[k].append((incl_prob(d) - p_ref).numpy())
            pmap[k].append(incl_prob(d).numpy())
            post_mean[k].append(d.mean(dim=0).numpy())
            post_sd[k].append(d.std(dim=0).numpy())
            spars[k].append(float((d.abs() <= ZERO_TOL).to(torch.float64).mean()))
            bv[k] += int(r[k].get("bv", 0))
            k_ind = (r[k]["draws"].abs() > ZERO_TOL).numpy().astype(float)
            ess_k = np.array([ess_batch_means(k_ind[:, i]) for i in range(args.D)])
            pr = p_ref.numpy()
            c_theory[k].append(np.sqrt(pr * (1 - pr) * (1.0 / ess_g + 1.0 / ess_k)))
        # Coordinates where the exact posterior is genuinely uncertain are the
        # ones that discriminate; near 0/1 every sampler agrees trivially.
        pz, pb = incl_prob(r["sticky_zigzag"]["draws"]), incl_prob(r["sticky_boomerang"]["draws"])
        for i in range(args.D):
            if 0.05 < float(p_ref[i]) < 0.95:
                hard.append((s, i, float(p_ref[i]), float(pz[i]), float(pb[i])))

    if not seeds_found:
        print("No results found -- run sticky_exactness_linreg.py first.")
        return

    print(f"Seeds: {seeds_found}   (D={args.D}, w={args.w})\n")

    print("=== Signed deviation from Gibbs, pooled over all coordinates x seeds ===")
    print("(exactness => mean signed deviation 0; |z| <~ 2 is consistent with it)\n")
    print(f"{'sampler':<20}{'mean signed':>13}{'SE':>9}{'z':>8}{'mean |dev|':>12}{'max |dev|':>11}")
    print("-" * 73)
    for k, lab in KEYS:
        a = np.concatenate(dev[k])
        mean, se = a.mean(), a.std(ddof=1) / np.sqrt(a.size)
        print(f"{lab:<20}{mean:>13.5f}{se:>9.5f}{mean/se:>8.2f}"
              f"{np.abs(a).mean():>12.5f}{np.abs(a).max():>11.5f}")

    print(f"\n=== Hard coordinates only (0.05 < P_exact < 0.95) ===")
    if hard:
        hz = np.array([h[3] - h[2] for h in hard])
        hb = np.array([h[4] - h[2] for h in hard])
        print(f"n = {len(hard)} coordinate-seed pairs\n")
        print(f"{'sampler':<20}{'mean signed':>13}{'SE':>9}{'z':>8}{'n neg':>8}{'n pos':>8}")
        print("-" * 66)
        for lab, a in [("Sticky ZigZag", hz), ("Sticky Boomerang", hb)]:
            se = a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else float("nan")
            z = a.mean() / se if se and se == se else float("nan")
            print(f"{lab:<20}{a.mean():>13.5f}{se:>9.5f}{z:>8.2f}"
                  f"{int((a < 0).sum()):>8}{int((a > 0).sum()):>8}")
        print(f"\n{'seed':<6}{'coord':>6}{'P_exact':>10}{'stickyZZ':>10}{'stickyBoom':>12}"
              f"{'dZZ':>9}{'dBoom':>9}")
        for s, i, pr, pz, pb in sorted(hard):
            print(f"{s:<6}{i:>6}{pr:>10.3f}{pz:>10.3f}{pb:>12.3f}{pz-pr:>9.3f}{pb-pr:>9.3f}")
    else:
        print("  (none -- every coordinate was unambiguous)")

    print(f"\n=== Sparsity (fraction of draws exactly zero) ===")
    print(f"{'Gibbs (exact)':<20}{np.mean(spars['gibbs']):>10.4f} +- {np.std(spars['gibbs'], ddof=1):.4f}")
    for k, lab in KEYS:
        print(f"{lab:<20}{np.mean(spars[k]):>10.4f} +- {np.std(spars[k], ddof=1):.4f}"
              f"   bound violations (all seeds): {bv[k]}")

    make_figure(pref_all, dev, c_theory, post_mean, post_sd, beta_true, n_gibbs, seeds_found,
                args.out / f"exactness_summary_D{args.D}_w{args.w}.png")


def make_figure(pref_all, dev, c_theory, post_mean, post_sd, beta_true, n_gibbs: int,
                 seeds, path: Path) -> None:
    """Single-column paper figure: sticky PDMPs vs the exact posterior.

    Left  -- posterior coefficients. Per coordinate, the posterior mean with
             a +-1 posterior-sd bar, one series per sampler, against the true
             beta. This checks the SECOND exactness claim (E[beta_i | y], the
             continuous parameter) that the inclusion-probability panels do
             not touch, and it is the panel that shows what the posterior
             actually looks like rather than only how far apart two estimates
             of it are.

             It replaces an earlier calibration panel (exact p on x, sampler p
             on y). That panel was a rotation of the right one -- same data,
             same x, y differing only by subtracting the diagonal -- and at
             its scale the deviations it was meant to show were ~1% of the
             axis height, thinner than the diagonal drawn over them. Two
             panels of one argument, one of them unreadable.

    Right -- signed deviation against exact probability. This is the panel
             that separates noise from bias: unbiased scatter sits symmetric
             about zero with no trend in p, whereas a defective freeze/thaw
             balance would tilt or shift systematically.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_ref = np.concatenate(pref_all)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    style = {"sticky_zigzag": ("C2", "o", "Sticky ZigZag"),
             "sticky_boomerang": ("C0", "^", "Sticky Boomerang")}

    # -- left: posterior coefficients --
    # Averaged over seeds: beta_true is shared (asserted in main), so the
    # per-coordinate posterior is the same object in every seed and averaging
    # is a variance reduction, not a mix of different quantities. The bar is
    # the mean WITHIN-seed posterior sd (the actual posterior width), not the
    # across-seed spread of the means (~0.06 here, a different and much
    # smaller quantity that would misleadingly suggest a tight posterior).
    ax = axes[0]
    D = len(beta_true)
    # Show every signal coordinate plus a few null ones. All D=20 coordinates
    # would spend three quarters of the axis on nulls that sit flat at zero,
    # squeezing the part that carries the argument into the left fifth. The
    # nulls still have to appear -- "the samplers agree the rest is zero" is
    # part of the claim -- but a handful of them makes that point as well as
    # fifteen, and the remainder is summarised in the axis label.
    n_sig = int(np.count_nonzero(beta_true))
    n_show = min(D, n_sig + 4)
    idx = np.arange(n_show)
    series = [("gibbs", "C4", "D", "Gibbs")]
    series += [(k, style[k][0], style[k][1], style[k][2]) for k, _ in KEYS]
    # Small horizontal offsets so three overlapping series stay legible.
    offsets = np.linspace(-0.26, 0.26, len(series))
    for (k, col, mk, lab), dx in zip(series, offsets):
        m = np.mean(post_mean[k], axis=0)[:n_show]
        sd = np.mean(post_sd[k], axis=0)[:n_show]
        ax.errorbar(idx + dx, m, yerr=sd, fmt=mk, color=col, ms=3.4,
                    lw=0, elinewidth=1.0, capsize=0, alpha=0.9,
                    label=lab, zorder=3)
    # Truth drawn last so it reads on top of the estimates.
    ax.scatter(idx, beta_true[:n_show], marker="*", s=15, color="C1",
               linewidths=1.6, zorder=4, label=r"$\beta_i$")
    ax.axhline(0.0, color="0.75", lw=0.7, zorder=1)
    # Separate the signals from the nulls so the reader sees which is which
    # without counting against beta_true.
    ax.axvline(n_sig - 0.5, color="0.8", lw=0.8, ls=":", zorder=1)
    ax.set_xlabel(f"coordinate $i$", fontsize=12)
    ax.set_ylabel(r"Posterior mean", fontsize=12)
    ax.set_xlim(-0.7, n_show - 0.3)
    ax.set_xticks(idx[::2])
    ax.tick_params(axis="both", labelsize=10)
    lo = min(float(np.min(beta_true[:n_show])), -0.95)
    hi = float(np.max(beta_true[:n_show]))
    ax.set_ylim(lo - 0.18, hi + 0.42)
    # Two columns keeps the legend inside the headroom above the tallest
    # signal instead of needing a whole extra unit of empty axis.
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9,
              handletextpad=0.35, borderpad=0.3, labelspacing=0.2,
              ncol=2, columnspacing=0.8)

    # Coordinate 3 is a genuine posterior effect that looks like a sampler
    # error: its true value is small enough that the spike-and-slab posterior
    # shrinks it most of the way to zero, so the red truth marker sits well
    # away from three estimates that agree with each other exactly. Without a
    # word here the natural reading is "the samplers missed one", which is the
    # opposite of this figure's point -- all three agree, and they agree
    # because the exact posterior really does shrink it.
    shrunk = None
    m_gibbs = np.mean(post_mean["gibbs"], axis=0)
    for i in range(n_sig):
        bt_i, m_i = float(beta_true[i]), float(m_gibbs[i])
        if abs(bt_i) > 1e-12 and abs(m_i) < 0.5 * abs(bt_i):
            shrunk = i if shrunk is None else shrunk
    # if shrunk is not None:
    #     ax.annotate(
    #         "shrunk by\nthe posterior",
    #         xy=(shrunk + 0.12, float(beta_true[shrunk])),
    #         xytext=(shrunk + 1.25, lo + 0.10),
    #         fontsize=6.4, color="0.35", va="bottom", ha="left",
    #         arrowprops=dict(arrowstyle="-", color="0.55", lw=0.7,
    #                         shrinkA=1.5, shrinkB=2.5),
    #         zorder=5,
    #     )
    ax.set_title("Posterior coefficients", fontsize=13)

    # -- right: signed deviation, the noise-vs-bias panel --
    ax = axes[1]
    ax.axhline(0.0, color="0.35", lw=0.9, zorder=2)
    a_all = np.concatenate([np.concatenate(dev[k]) for k, _ in KEYS])
    p_all = np.tile(p_ref, len(KEYS))
    shape = np.sqrt(np.clip(p_all * (1.0 - p_all), 0.0, None))
    hard_m = (p_all > 0.05) & (p_all < 0.95)
    ct_all = np.concatenate([np.concatenate(c_theory[k]) for k, _ in KEYS])
    c_band = float(np.median(ct_all[hard_m] / shape[hard_m]))
    z = a_all[hard_m] / ct_all[hard_m]
    c_fit = float(np.sqrt(np.mean((a_all[hard_m] / shape[hard_m]) ** 2)))
    print(f"\n=== Deviation vs predicted Monte Carlo scale ===")
    print(f"  fitted coefficient     {c_fit:.5f}")
    print(f"  predicted (median ESS) {c_band:.5f}")
    print(f"  ratio                  {c_fit / c_band:.2f}")
    print(f"  sd of standardized deviations {z.std(ddof=1):.2f}  (1.0 = as predicted)")
    grid = np.linspace(0.0, 1.0, 201)
    band = 2.0 * c_band * np.sqrt(grid * (1.0 - grid))
    ax.fill_between(grid, -band, band, color="0.75", alpha=0.55, lw=0, zorder=1)#,
                    #label=r"$\pm2$SD (binomial)")
    for k, (col, mk, lab) in style.items():
        a = np.concatenate(dev[k])
        ax.scatter(p_ref, a, s=20, color=col, marker=mk, alpha=0.75,
                   linewidths=0, zorder=3, label=lab)
    ax.set_xlabel(r"Gibbs $P(\gamma_i{=}1\mid y)$", fontsize=12)
    ax.set_ylabel(r"Sticky PDMP deviation", fontsize=12)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.5, handletextpad=0.4)
    ax.set_title("Signed deviation", fontsize=13)

    for a in axes:
        #a.tick_params(labelsize=7.5)
        #a.xaxis.label.set_size(8); a.yaxis.label.set_size(8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    fig.tight_layout(pad=0.4, w_pad=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nfigure -> {path}  (and .pdf)")


if __name__ == "__main__":
    main()
