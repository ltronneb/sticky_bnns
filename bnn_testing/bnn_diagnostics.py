"""
Sticky Boomerang BNN Diagnostics
================================
Comprehensive visualisation for Bayesian Neural Network inference
with the Sticky Boomerang sampler.

Produces three figure groups:
  1. Network structure & sparsity  (weight heatmaps, PIPs, connectivity graph)
  2. Sampling diagnostics          (traces, ACF, model size, energy)
  3. Predictive performance        (decision boundary / predictions, uncertainty)

Usage
-----
    from bnn_diagnostics import plot_bnn_diagnostics
    figs = plot_bnn_diagnostics(sampler, X, y, layer_sizes, shapes, slices,
                                weight_mask, X_test=None, y_test=None)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe
from sazz.samplers.boomerang_sampler.utils import resample_sticky_pdmp_path


# ================================================================
# Helper: unpack flat theta → list of (W, b)
# ================================================================
def _unpack(theta, shapes, slices):
    params = []
    for (w_shape, _), (w_sl, b_sl) in zip(shapes, slices):
        W = theta[w_sl].reshape(w_shape)
        b = theta[b_sl]
        params.append((W, b))
    return params


def _forward_np(theta, X, shapes, slices):
    """Numpy forward pass (tanh hidden, sigmoid output)."""
    params = _unpack(theta, shapes, slices)
    h = X
    for W, b in params[:-1]:
        h = np.tanh(h @ W + b)
    W_last, b_last = params[-1]
    logits = (h @ W_last + b_last).ravel()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs


# ================================================================
# 1. Network Structure & Sparsity
# ================================================================
def plot_network_structure(sampler, layer_sizes, shapes, slices, weight_mask):
    """
    Fig 1: Weight heatmaps, posterior inclusion probabilities,
           and network connectivity diagram.
    """
    pos = sampler.Position[:sampler.iteration]
    times = sampler.Time[:sampler.iteration]
    dt = np.diff(times, prepend=0)
    D = sampler.D

    # Time-weighted posterior mean and PIP
    total_time = dt.sum()
    post_mean = (pos * dt[:, None]).sum(axis=0) / total_time
    frozen_frac = np.zeros(D)
    for d in range(D):
        at_zero = np.abs(pos[:, d]) < 1e-12
        frozen_frac[d] = dt[at_zero].sum() / total_time
    pip = 1.0 - frozen_frac

    n_layers = len(layer_sizes) - 1

    fig = plt.figure(figsize=(6 * n_layers + 4, 10), constrained_layout=True)
    gs = GridSpec(2, n_layers + 1, figure=fig, width_ratios=[1] * n_layers + [1.2])

    # ── Row 0: Weight heatmaps (posterior mean) ──
    for l in range(n_layers):
        ax = fig.add_subplot(gs[0, l])
        w_sl = slices[l][0]
        w_shape = shapes[l][0]
        W_mean = post_mean[w_sl].reshape(w_shape)
        vmax = np.max(np.abs(W_mean)) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(W_mean.T, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xlabel(f"Input (layer {l})")
        ax.set_ylabel(f"Output (layer {l+1})")
        ax.set_title(f"W{l+1} posterior mean")
        plt.colorbar(im, ax=ax, shrink=0.7)

    # ── Row 1: PIP heatmaps per layer ──
    for l in range(n_layers):
        ax = fig.add_subplot(gs[1, l])
        w_sl = slices[l][0]
        w_shape = shapes[l][0]
        W_pip = pip[w_sl].reshape(w_shape)
        im = ax.imshow(W_pip.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xlabel(f"Input (layer {l})")
        ax.set_ylabel(f"Output (layer {l+1})")
        ax.set_title(f"W{l+1} inclusion P(w≠0)")
        plt.colorbar(im, ax=ax, shrink=0.7)

    # ── Network connectivity diagram ──
    ax_net = fig.add_subplot(gs[:, -1])
    _draw_network(ax_net, layer_sizes, post_mean, pip, shapes, slices)
    ax_net.set_title("Network connectivity\n(line = PIP, colour = sign)")

    fig.suptitle("Network Structure & Sparsity", fontsize=15, fontweight="bold")
    return fig


def _draw_network(ax, layer_sizes, post_mean, pip, shapes, slices):
    """Draw a neural network diagram with edges coloured by weight sign
    and thickness by PIP."""
    ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
    max_nodes = max(layer_sizes)
    ax.set_ylim(-0.5, max_nodes - 0.5)
    ax.axis("off")

    # Node positions
    positions = []
    for l, size in enumerate(layer_sizes):
        y_offset = (max_nodes - size) / 2.0
        layer_pos = [(l, y_offset + j) for j in range(size)]
        positions.append(layer_pos)

    # Draw edges
    for l in range(len(layer_sizes) - 1):
        w_sl = slices[l][0]
        w_shape = shapes[l][0]
        W_mean = post_mean[w_sl].reshape(w_shape)
        W_pip = pip[w_sl].reshape(w_shape)

        for i in range(w_shape[0]):
            for j in range(w_shape[1]):
                p = W_pip[i, j]
                if p < 0.05:
                    continue  # skip near-zero PIP connections
                color = "steelblue" if W_mean[i, j] >= 0 else "crimson"
                alpha = float(np.clip(p, 0.1, 1.0))
                lw = float(np.clip(p * 3, 0.3, 3.0))
                x0, y0 = positions[l][i]
                x1, y1 = positions[l + 1][j]
                ax.plot([x0, x1], [y0, y1], c=color, alpha=alpha,
                        lw=lw, zorder=1)

    # Draw nodes
    for l, layer_pos in enumerate(positions):
        for (x, y) in layer_pos:
            circle = plt.Circle((x, y), 0.15, fc="white", ec="black",
                                lw=1.5, zorder=3)
            ax.add_patch(circle)

    # Layer labels
    labels = ["Input"] + [f"Hidden {l}" for l in range(1, len(layer_sizes) - 1)] + ["Output"]
    for l, label in enumerate(labels):
        ax.text(l, -0.4, label, ha="center", fontsize=8, style="italic")


# ================================================================
# 2. Sampling Diagnostics
# ================================================================
def plot_sampling_diagnostics(sampler, weight_mask):
    """
    Fig 2: Trace plots, ACF, model size, energy trace,
           weight vs bias frozen fractions.
    """
    pos = sampler.Position[:sampler.iteration]
    times = sampler.Time[:sampler.iteration]
    dt = np.diff(times, prepend=0)
    D = sampler.D

    # Time-weighted frozen fraction
    total_time = dt.sum()
    frozen_frac = np.zeros(D)
    for d in range(D):
        at_zero = np.abs(pos[:, d]) < 1e-12
        frozen_frac[d] = dt[at_zero].sum() / total_time
    pip = 1.0 - frozen_frac

    # Resample path
    t, x = resample_sticky_pdmp_path(sampler, n_samples=min(5000, sampler.iteration * 2))

    # Model size = number of active weights (not biases) over time
    active_weights = np.sum(np.abs(x[:, weight_mask]) >= 1e-10, axis=1)
    n_weights = weight_mask.sum()

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    # ── Trace of 6 representative weights ──
    ax1 = fig.add_subplot(gs[0, 0])
    weight_idx = np.where(weight_mask)[0]
    # pick weights with varying PIP
    pip_weights = pip[weight_idx]
    sorted_by_pip = weight_idx[np.argsort(pip_weights)]
    n_show = min(6, len(sorted_by_pip))
    picks = sorted_by_pip[np.linspace(0, len(sorted_by_pip) - 1, n_show, dtype=int)]
    cmap = plt.cm.viridis
    step = max(1, len(t) // 3000)
    for k, idx in enumerate(picks):
        color = cmap(k / max(n_show - 1, 1))
        ax1.plot(t[::step], x[::step, idx], lw=0.4, c=color, alpha=0.8,
                 label=f"w[{idx}] PIP={pip[idx]:.2f}")
    ax1.set_xlabel("t")
    ax1.set_ylabel("weight value")
    ax1.set_title("Weight traces (sorted by PIP)")
    ax1.legend(fontsize=6, ncol=2, loc="upper right")

    # ── ACF of selected weights ──
    ax2 = fig.add_subplot(gs[0, 1])
    max_lag = min(300, len(x) // 2)
    for k, idx in enumerate(picks):
        color = cmap(k / max(n_show - 1, 1))
        series = x[:, idx] - x[:, idx].mean()
        acf = np.correlate(series, series, mode="full")
        acf = acf[len(acf) // 2:]
        if acf[0] > 0:
            acf /= acf[0]
        ax2.plot(range(max_lag), acf[:max_lag], lw=0.7, c=color, alpha=0.8)
    ax2.axhline(0, c="grey", lw=0.5, ls="--")
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("ACF")
    ax2.set_title("Autocorrelation")

    # ── Active weights over time ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t[::step], active_weights[::step], lw=0.4, c="teal")
    ax3.axhline(n_weights, c="grey", lw=0.5, ls=":", label=f"total weights = {n_weights}")
    ax3.set_xlabel("t")
    ax3.set_ylabel("# active weights")
    ax3.set_title("Network sparsity over time")
    ax3.legend(fontsize=8)

    # ── PIP histogram for weights ──
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(pip[weight_mask], bins=30, color="steelblue", edgecolor="k",
             alpha=0.7, label="weights")
    ax4.hist(pip[~weight_mask], bins=10, color="coral", edgecolor="k",
             alpha=0.7, label="biases")
    ax4.set_xlabel("Posterior Inclusion Probability")
    ax4.set_ylabel("Count")
    ax4.set_title("PIP distribution")
    ax4.legend(fontsize=8)

    # ── Energy trace ──
    ax5 = fig.add_subplot(gs[1, 1])
    # compute energy at skeleton points (subsample for speed)
    skel_step = max(1, sampler.iteration // 500)
    skel_idx = range(0, sampler.iteration, skel_step)
    if hasattr(sampler, 'E') and callable(sampler.E):
        energies = [sampler.E(pos[i]) for i in skel_idx]
        ax5.plot(times[list(skel_idx)], energies, lw=0.5, c="purple")
        ax5.set_xlabel("t")
        ax5.set_ylabel("E(θ)")
        ax5.set_title("Energy trace (neg log posterior)")
    else:
        ax5.text(0.5, 0.5, "E(θ) not available", ha="center", va="center",
                 transform=ax5.transAxes)
        ax5.set_title("Energy trace")

    # ── Frozen fraction: weights vs biases ──
    ax6 = fig.add_subplot(gs[1, 2])
    w_froz = frozen_frac[weight_mask]
    b_froz = frozen_frac[~weight_mask]
    bp = ax6.boxplot([w_froz, b_froz], labels=["Weights", "Biases"],
                     patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("coral")
    bp["boxes"][1].set_alpha(0.6)
    ax6.set_ylabel("Fraction of time frozen at 0")
    ax6.set_title("Frozen time: weights vs biases")

    fig.suptitle("Sampling Diagnostics", fontsize=15, fontweight="bold")
    return fig


# ================================================================
# 3. Predictive Performance
# ================================================================
def plot_predictive(sampler, X_train, y_train, shapes, slices,
                    X_test=None, y_test=None, n_posterior_samples=200,
                    layer_sizes=None):
    """
    Fig 3: Posterior predictive.
    - If d_in == 2: decision boundary with uncertainty
    - General d_in: calibration, predicted probability histogram,
                    accuracy over posterior samples
    """
    pos = sampler.Position[:sampler.iteration]
    times = sampler.Time[:sampler.iteration]
    d_in = X_train.shape[1]

    # Draw posterior samples (time-spaced)
    sample_idx = np.linspace(
        sampler.iteration // 5,  # skip burn-in
        sampler.iteration - 1,
        min(n_posterior_samples, sampler.iteration // 2),
        dtype=int,
    )
    theta_samples = pos[sample_idx]

    if X_test is None:
        X_test = X_train
        y_test = y_train

    # ── Predict with each posterior sample ──
    all_probs = np.array([
        _forward_np(theta, X_test, shapes, slices)
        for theta in theta_samples
    ])  # (n_samples, n_test)

    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)
    pred_labels = (mean_probs > 0.5).astype(float)
    accuracy = np.mean(pred_labels == y_test) if y_test is not None else None

    if d_in == 2:
        return _plot_predictive_2d(
            X_train, y_train, X_test, y_test,
            theta_samples, shapes, slices,
            mean_probs, std_probs, accuracy,
        )
    else:
        return _plot_predictive_general(
            X_test, y_test, all_probs,
            mean_probs, std_probs, accuracy,
            theta_samples, shapes, slices
        )


def _plot_predictive_2d(X_train, y_train, X_test, y_test,
                         theta_samples, shapes, slices,
                         mean_probs, std_probs, accuracy):
    """Decision boundary + uncertainty for 2D input."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    # Grid
    pad = 0.5
    x0_range = np.linspace(X_train[:, 0].min() - pad, X_train[:, 0].max() + pad, 150)
    x1_range = np.linspace(X_train[:, 1].min() - pad, X_train[:, 1].max() + pad, 150)
    G0, G1 = np.meshgrid(x0_range, x1_range)
    X_grid = np.column_stack([G0.ravel(), G1.ravel()])

    # Predict on grid with all posterior samples
    grid_probs = np.array([
        _forward_np(theta, X_grid, shapes, slices)
        for theta in theta_samples
    ])
    grid_mean = grid_probs.mean(axis=0).reshape(G0.shape)
    grid_std = grid_probs.std(axis=0).reshape(G0.shape)

    # 1) Mean decision boundary
    ax = axes[0]
    cf = ax.contourf(G0, G1, grid_mean, levels=20, cmap="RdBu_r", alpha=0.7)
    ax.contour(G0, G1, grid_mean, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
               c="steelblue", s=15, alpha=0.6, edgecolors="k", lw=0.3, label="y=0")
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
               c="crimson", s=15, alpha=0.6, edgecolors="k", lw=0.3, label="y=1")
    ax.set_title("Mean posterior P(y=1|x)")
    ax.legend(fontsize=7)
    plt.colorbar(cf, ax=ax, shrink=0.8)

    # 2) Uncertainty (std of predicted probability)
    ax = axes[1]
    cf = ax.contourf(G0, G1, grid_std, levels=20, cmap="magma_r", alpha=0.8)
    ax.contour(G0, G1, grid_mean, levels=[0.5], colors="w", linewidths=1, linestyles="--")
    ax.scatter(X_train[:, 0], X_train[:, 1], c="white", s=5, alpha=0.3)
    ax.set_title("Predictive uncertainty (std)")
    plt.colorbar(cf, ax=ax, shrink=0.8)

    # 3) Sample decision boundaries
    ax = axes[2]
    n_show = min(30, len(theta_samples))
    for i in range(n_show):
        gi = grid_probs[i].reshape(G0.shape)
        ax.contour(G0, G1, gi, levels=[0.5], colors="steelblue",
                   linewidths=0.3, alpha=0.4)
    ax.contour(G0, G1, grid_mean, levels=[0.5], colors="k", linewidths=2)
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
               c="steelblue", s=15, alpha=0.5, edgecolors="k", lw=0.3)
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
               c="crimson", s=15, alpha=0.5, edgecolors="k", lw=0.3)
    title = "Posterior decision boundaries"
    if accuracy is not None:
        title += f"\nAccuracy: {accuracy:.1%}"
    ax.set_title(title)

    fig.suptitle("Predictive Performance (2D)", fontsize=15, fontweight="bold")
    return fig


def _plot_predictive_general(X_test, y_test, all_probs,
                              mean_probs, std_probs, accuracy,
                              theta_samples, shapes, slices):
    """General predictive diagnostics for d_in > 2."""
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    # 1) Predicted probability histogram
    ax1 = fig.add_subplot(gs[0, 0])
    if y_test is not None:
        ax1.hist(mean_probs[y_test == 0], bins=30, alpha=0.6, color="steelblue",
                 density=True, label="true y=0")
        ax1.hist(mean_probs[y_test == 1], bins=30, alpha=0.6, color="crimson",
                 density=True, label="true y=1")
        ax1.legend(fontsize=8)
    else:
        ax1.hist(mean_probs, bins=30, alpha=0.6, color="steelblue", density=True)
    ax1.set_xlabel("Mean predicted P(y=1)")
    ax1.set_ylabel("Density")
    ax1.set_title("Posterior predictive distribution")

    # 2) Uncertainty vs correctness
    ax2 = fig.add_subplot(gs[0, 1])
    if y_test is not None:
        correct = ((mean_probs > 0.5) == y_test)
        ax2.scatter(mean_probs[correct], std_probs[correct],
                    s=8, c="seagreen", alpha=0.5, label="correct")
        ax2.scatter(mean_probs[~correct], std_probs[~correct],
                    s=15, c="crimson", alpha=0.7, label="incorrect", marker="x")
        ax2.legend(fontsize=8)
    else:
        ax2.scatter(mean_probs, std_probs, s=8, c="steelblue", alpha=0.5)
    ax2.set_xlabel("Mean predicted P(y=1)")
    ax2.set_ylabel("Predictive std")
    ax2.set_title("Uncertainty vs prediction")

    # 3) Calibration curve
    ax3 = fig.add_subplot(gs[0, 2])
    if y_test is not None:
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accs = []
        bin_counts = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (mean_probs >= lo) & (mean_probs < hi)
            if mask.sum() > 0:
                bin_centers.append((lo + hi) / 2)
                bin_accs.append(y_test[mask].mean())
                bin_counts.append(mask.sum())
        ax3.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
        ax3.bar(bin_centers, bin_accs, width=1 / n_bins * 0.8,
                color="steelblue", alpha=0.6, edgecolor="k")
        ax3.set_xlabel("Predicted probability")
        ax3.set_ylabel("Observed frequency")
        ax3.set_title("Calibration")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Need y_test", ha="center", va="center",
                 transform=ax3.transAxes)

    # 4) Accuracy over posterior samples (shows posterior over accuracy)
    ax4 = fig.add_subplot(gs[1, 0])
    if y_test is not None:
        accs = [(probs > 0.5).astype(float) == y_test
                for probs in all_probs]
        sample_accs = [a.mean() for a in accs]
        ax4.hist(sample_accs, bins=25, color="teal", edgecolor="k", alpha=0.7)
        ax4.axvline(np.mean(sample_accs), c="k", lw=1.5, ls="--",
                    label=f"mean = {np.mean(sample_accs):.3f}")
        ax4.set_xlabel("Accuracy")
        ax4.set_ylabel("Count")
        ax4.set_title("Posterior over accuracy")
        ax4.legend(fontsize=8)
    else:
        ax4.text(0.5, 0.5, "Need y_test", ha="center", va="center",
                 transform=ax4.transAxes)

    # 5) Sorted predictions with uncertainty bands
    ax5 = fig.add_subplot(gs[1, 1])
    sort_idx = np.argsort(mean_probs)
    n_pts = len(mean_probs)
    ax5.fill_between(range(n_pts),
                     np.clip(mean_probs[sort_idx] - 2 * std_probs[sort_idx], 0, 1),
                     np.clip(mean_probs[sort_idx] + 2 * std_probs[sort_idx], 0, 1),
                     alpha=0.3, color="steelblue", label="±2σ")
    ax5.plot(range(n_pts), mean_probs[sort_idx], lw=0.8, c="steelblue")
    if y_test is not None:
        ax5.scatter(range(n_pts), y_test[sort_idx], s=5, c="crimson",
                    alpha=0.4, label="true label", zorder=3)
    ax5.set_xlabel("Test point (sorted)")
    ax5.set_ylabel("P(y=1)")
    ax5.set_title("Sorted predictions ± uncertainty")
    ax5.legend(fontsize=8)

    # 6) Per-sample log-likelihood (posterior predictive check)
    ax6 = fig.add_subplot(gs[1, 2])
    if y_test is not None:
        sample_nlls = []
        for probs in all_probs:
            p_clip = np.clip(probs, 1e-8, 1 - 1e-8)
            nll = -np.mean(y_test * np.log(p_clip) + (1 - y_test) * np.log(1 - p_clip))
            sample_nlls.append(nll)
        ax6.hist(sample_nlls, bins=25, color="purple", edgecolor="k", alpha=0.6)
        ax6.axvline(np.mean(sample_nlls), c="k", lw=1.5, ls="--",
                    label=f"mean = {np.mean(sample_nlls):.3f}")
        ax6.set_xlabel("Mean binary cross-entropy")
        ax6.set_ylabel("Count")
        ax6.set_title("Posterior predictive log-loss")
        ax6.legend(fontsize=8)
    else:
        ax6.text(0.5, 0.5, "Need y_test", ha="center", va="center",
                 transform=ax6.transAxes)

    title = "Predictive Performance"
    if accuracy is not None:
        title += f"  (BMA accuracy: {accuracy:.1%})"
    fig.suptitle(title, fontsize=15, fontweight="bold")
    return fig


# ================================================================
# Main entry point
# ================================================================
def plot_bnn_diagnostics(sampler, X_train, y_train, layer_sizes,
                          shapes, slices, weight_mask,
                          X_test=None, y_test=None,
                          n_posterior_samples=200):
    """
    Generate all three diagnostic figure groups.

    Returns dict of figures: {"structure", "sampling", "predictive"}
    """
    fig1 = plot_network_structure(sampler, layer_sizes, shapes, slices, weight_mask)
    fig2 = plot_sampling_diagnostics(sampler, weight_mask)
    fig3 = plot_predictive(sampler, X_train, y_train, shapes, slices,
                           X_test=X_test, y_test=y_test,
                           n_posterior_samples=n_posterior_samples,
                           layer_sizes=layer_sizes)
    return {"structure": fig1, "sampling": fig2, "predictive": fig3}