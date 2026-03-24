"""
mcmc_plots.py
=============
Plotting utilities for MCMC sample diagnostics.

Translated from the R notebook "What to do with Curved Distributions"
by Filippo Pagani. Licensed under GPL-3.

Main function
-------------
    better_pairs(samples, resol=1.0, labels=None)

Quick start
-----------
    import numpy as np
    import matplotlib.pyplot as plt
    from mcmc_plots import better_pairs

    # Your MCMC samples — shape (n_samples, n_dims)
    samples = np.random.randn(50_000, 3)
    fig, axes = better_pairs(samples, resol=0.7, labels=["x", "y", "z"])
    plt.show()

Notes on the contours
---------------------
Each contour in the lower-triangle panels encloses the stated fraction of the
total probability mass (highest-density region, computed from the 2-D histogram).
This is analogous to conf_contour2() / myBetterPairs() from the original R notebook.
Contour lines are *not* smoothed with a kernel, which makes it easier to see when
the distribution is genuinely multimodal or highly skewed.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# ── Default confidence levels ──────────────────────────────────────────────────
DEFAULT_CLEVELS = [0.9999, 0.999, 0.99] + list(
    np.round(np.arange(0.9, 0.05, -0.1), 10)
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _spectral_colors(n: int) -> list:
    """n colors from the reversed Spectral colormap (warm→cool for low→high density)."""
    cmap = plt.cm.get_cmap("Spectral_r", n)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _diagonal_hist(ax, x: np.ndarray, color: str = "#08306b") -> None:
    """
    1-D histogram panel (diagonal of pairs plot).
    Uses 4× the number of bins that numpy's 'auto' heuristic suggests,
    to expose fine structure that coarser binning would hide.
    """
    _, bins_auto = np.histogram(x, bins="auto")
    n_bins = max(10, (len(bins_auto) - 1) * 4)
    counts, bins = np.histogram(x, bins=n_bins)
    y = counts / counts.max()
    ax.bar(bins[:-1], y, width=np.diff(bins), color=color, align="edge", alpha=0.85)
    ax.set_ylim(0, 1.5)


def _hist2d(
    x: np.ndarray, y: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2-D histogram. Returns (bin_centers_x, bin_centers_y, counts)."""
    counts, xedges, yedges = np.histogram2d(x, y, bins=n_bins)
    xx = 0.5 * (xedges[:-1] + xedges[1:])
    yy = 0.5 * (yedges[:-1] + yedges[1:])
    return xx, yy, counts


def _draw_contours(
    ax,
    xx: np.ndarray,
    yy: np.ndarray,
    counts: np.ndarray,
    clevels: list,
    colors: list,
    draw_labels: bool = False,
) -> None:
    """
    Draw highest-density-region contours on a 2-D histogram.

    For each alpha in clevels, the contour encloses the *smallest* region that
    contains at least alpha × 100 % of the total probability mass. This is
    computed by sorting bins in decreasing density order, taking the cumulative
    sum, and finding the density threshold at which the cumulative sum crosses
    alpha — exactly as conf_contour2() does in the original R code.
    """
    flat = counts.flatten()
    zdens = np.sort(flat)[::-1]           # descending density
    czdens = np.cumsum(zdens)
    czdens = czdens / czdens[-1]          # normalise to [0, 1]

    for i, alpha in enumerate(clevels):
        idx = np.where(czdens <= alpha)[0]
        if len(idx) == 0:
            continue
        crit_val = zdens[idx[-1]]
        if crit_val <= 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                cs = ax.contour(
                    xx,
                    yy,
                    counts.T,          # transpose: contour expects Z[row, col] = Z[y, x]
                    levels=[crit_val],
                    colors=[colors[i]],
                    linewidths=1.2,
                )
                if draw_labels:
                    ax.clabel(cs, fmt=f"{alpha:.3g}", fontsize=6)
            except Exception:
                pass


# ── Public API ─────────────────────────────────────────────────────────────────

def better_pairs(
    samples,
    resol: float = 1.0,
    labels: list | None = None,
    clevels: list | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    draw_labels: bool = False,
):
    """
    Pairs plot for MCMC samples with 2-D histogram contours.

    * **Diagonal** — 1-D histograms with ~4× the default bin count, to expose
      fine differences in large samples that coarse binning would hide.
    * **Lower triangle** — 2-D histograms with highest-density-region contours.
      Each contour encloses the stated fraction of total probability mass.
      Contours are *not* KDE-smoothed, which makes over-smoothing obvious.
    * **Upper triangle** — blank (as in the original R notebook).

    Python translation of ``myBetterPairs()`` from the R notebook
    *"What to do with Curved Distributions"* by Filippo Pagani.

    Parameters
    ----------
    samples : array-like, shape (n_samples, n_dims)
        One row per sample, one column per variable.
    resol : float, optional
        Bin resolution for 2-D histograms. Smaller → finer (more bins per axis).
        Typical range: 0.3–3.0. Default ``1.0`` gives ~100 bins per axis.
        If contours look too jagged, increase ``resol`` or ``n_samples``.
        If they look over-smoothed, decrease ``resol``.
    labels : list of str, optional
        Variable names for axis labels. Default: ``['x1', 'x2', ...]``.
    clevels : list of float, optional
        Confidence levels. Default:
        ``[0.9999, 0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]``.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches. Default: ``(4*d, 4*d)``.
    title : str, optional
        Overall figure title.
    draw_labels : bool, optional
        If ``True``, label contour lines with their probability values.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of matplotlib.axes.Axes, shape (d, d)

    Examples
    --------
    >>> import numpy as np
    >>> from mcmc_plots import better_pairs
    >>> rng = np.random.default_rng(0)
    >>> samples = rng.standard_normal((50_000, 3))
    >>> fig, axes = better_pairs(samples, resol=0.7, labels=["x", "y", "z"])
    >>> import matplotlib.pyplot as plt
    >>> plt.show()
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    n, d = samples.shape

    if labels is None:
        labels = [f"x{i + 1}" for i in range(d)]
    if clevels is None:
        clevels = DEFAULT_CLEVELS
    if figsize is None:
        figsize = (4 * d, 4 * d)

    n_bins = max(20, int(100 / resol))
    colors = _spectral_colors(len(clevels))

    fig, axes = plt.subplots(d, d, figsize=figsize)
    if d == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                _diagonal_hist(ax, samples[:, i])
                ax.set_xlabel(labels[i], fontsize=9)
                ax.set_yticks([])
            elif i > j:
                xx, yy, counts = _hist2d(samples[:, j], samples[:, i], n_bins)
                _draw_contours(ax, xx, yy, counts, clevels, colors, draw_labels)
                if i == d - 1:
                    ax.set_xlabel(labels[j], fontsize=9)
                if j == 0:
                    ax.set_ylabel(labels[i], fontsize=9)
            else:
                ax.set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=colors[i], lw=2, label=f"{clevels[i]:.4g}")
        for i in range(len(clevels))
    ]
    fig.legend(
        handles=legend_handles,
        title="Probability\nContained",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
        bbox_to_anchor=(1.0, 1.0),
        framealpha=0.85,
    )

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    plt.tight_layout()
    return fig, axes
