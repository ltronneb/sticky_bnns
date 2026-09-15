"""
Shared helpers for the image-classification "analyze_*_paper.py" scripts
(analyze_lenet_mnist_paper.py, analyze_ffn_mnist_paper.py) and their
pixel_noise.ipynb counterparts.

Covers the parts of those scripts that are architecture-agnostic: MAP/run
checkpoint loading and cross-checking, the predictive engine wrapper,
calibration metrics, the structured-sparsity ledger, and the rotation /
pixel-noise corruption sweep (levels -> pooled accuracy/P(true)/confidence/
entropy/ECE, plus the two summary figures). Architecture-specific bits
(the module class, its layer shapes, conv-vs-FFN dead-unit definitions,
input flattening) stay in each calling script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


# ==========================================================================
# Checkpoint loading
# ==========================================================================
def load_runs(run_dir: Path, run_specs: list[tuple[str, str]]) -> dict[str, dict]:
    """Load each (label, filename) in run_specs from run_dir; skip missing
    files with a printed note. Raises if none were found."""
    runs = {}
    for label, fname in run_specs:
        p = run_dir / fname
        if not p.exists():
            print(f"[{label}] missing {p} -- skipped")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        runs[label] = ck
        n, D = ck["samples"].shape
        print(f"[{label}] {n} draws x D={D}  test_acc(ckpt)={ck['test_accuracy']:.4f}  "
              f"sparsity(ckpt)={ck['sparsity_frac']:.4f}  prune_frac={ck['prune_frac']:.4f}  "
              f"act={ck['activation']}  wall={ck['elapsed_sec']/3600:.1f}h")
    assert runs, f"no run files under {run_dir}"
    return runs


def check_map_matches_runs(runs: dict[str, dict], map_ck: dict, dtype: torch.dtype) -> torch.Tensor:
    """Assert every run's x_ref / cold_start_mask matches map_ck (the run's
    cold-start point), and return X_REF cast to dtype.

    Compares cold_start_mask against (x_ref == 0) in x_ref's NATIVE dtype,
    not a dtype-cast copy: casting e.g. float64 -> float32 can flip a
    handful of exact zeros (rounding), which desyncs an == 0 check from the
    checkpoint's own mask even when the underlying data matches exactly.
    """
    x_ref_native = map_ck["x_ref"]
    X_REF = x_ref_native.to(dtype)
    for label, ck in runs.items():
        dmax = (ck["x_ref"].to(dtype) - X_REF).abs().max().item()
        cs_ok = torch.equal(ck["cold_start_mask"].bool(), (x_ref_native == 0))
        assert dmax < 1e-5 and cs_ok, (
            f"[{label}] MAP MISMATCH: max|x_ref - MAP.x_ref|={dmax:.3e}, cold_start match={cs_ok}. "
            f"MAP_REF_PATH is not this run's cold-start point."
        )
        print(f"  [{label}] matches run x_ref (max|Δ|={dmax:.1e}) and cold_start_mask OK")
    return X_REF


# ==========================================================================
# Predictive engine
# ==========================================================================
def make_predict_probs(bm, dtype: torch.dtype, device: torch.device,
                       flatten: bool = False) -> Callable:
    """Build a predict_probs(beta, X, bs=512) -> [N,10] softmax closure over
    a BayesianModule `bm`. Set flatten=True for an FFN whose forward expects
    [N, 784] rather than [N, 1, 28, 28]."""
    param_dict_fn = bm.param_dict_fn

    @torch.no_grad()
    def predict_probs(beta: torch.Tensor, X: torch.Tensor, bs: int = 512) -> torch.Tensor:
        beta = beta.to(dtype=dtype, device=device)
        out = []
        for i in range(0, X.shape[0], bs):
            xb = X[i:i + bs]
            if flatten:
                xb = torch.flatten(xb, 1)
            xb = xb.to(dtype=dtype, device=device)
            logits = torch.func.functional_call(bm.module, param_dict_fn(beta), (xb,))
            out.append(torch.softmax(logits, dim=-1).cpu())
        return torch.cat(out)

    return predict_probs


def make_posterior_mean_probs(predict_probs: Callable) -> Callable:
    """Build posterior_mean_probs(samples, X, n_draws, seed=0) ->
    (mean_probs [N,10], stacked_draw_probs [n_draws,N,10]) over predict_probs."""

    @torch.no_grad()
    def posterior_mean_probs(samples: torch.Tensor, X: torch.Tensor,
                             n_draws: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(samples.shape[0], generator=g)[:min(n_draws, samples.shape[0])]
        dp = torch.stack([predict_probs(samples[i], X) for i in idx])
        return dp.mean(0), dp

    return posterior_mean_probs


# ==========================================================================
# Calibration metrics
# ==========================================================================
def entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(-1)


def calibration_metrics(y_true: torch.Tensor, probs: torch.Tensor, n_bins: int = 15) -> dict:
    """acc / NLL / ECE / Brier / mean predictive entropy, plus the
    per-confidence-bin (conf, acc, weight) triples for a reliability plot."""
    y_true = y_true.long()
    conf, pred = probs.max(-1)
    correct = (pred == y_true).float()
    acc = correct.mean().item()
    p_true = probs.gather(-1, y_true.unsqueeze(-1)).squeeze(-1)
    nll = -p_true.clamp_min(1e-12).log().mean().item()
    onehot = F.one_hot(y_true, probs.shape[-1]).float()
    brier = ((probs - onehot) ** 2).sum(-1).mean().item()
    ent = entropy(probs).mean().item()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece, bin_stats = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ba, bc, w = correct[m].mean().item(), conf[m].mean().item(), m.float().mean().item()
            ece += w * abs(ba - bc)
            bin_stats.append((bc, ba, w))
    return {"acc": acc, "nll": nll, "ece": ece, "brier": brier,
            "mean_entropy": ent, "bin_stats": bin_stats}


# ==========================================================================
# Structured-sparsity ledger
# ==========================================================================
def sparsity_ledger(samples: torch.Tensor, zero_tol: float = 1e-8) -> dict:
    """Per-coordinate {ever_nonzero, always_zero, always_nonzero} over draws."""
    s = samples.numpy()
    is_zero = np.abs(s) < zero_tol
    return {"ever_nonzero": ~is_zero.all(0),
            "always_zero": is_zero.all(0),
            "always_nonzero": (~is_zero).all(0)}


def layer_survival_table(ledgers: dict[str, dict], run_display: dict[str, str],
                         layers: list[str], layer_slices: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Per-(run, layer) fraction ever/always-nonzero and always-zero."""
    rows = []
    for label, L in ledgers.items():
        for lname in layers:
            a, b = layer_slices[lname]
            sl = slice(a, b)
            rows.append({
                "run": run_display[label], "layer": lname, "n": b - a,
                "frac_ever_nonzero": L["ever_nonzero"][sl].mean(),
                "frac_always_nonzero": L["always_nonzero"][sl].mean(),
                "frac_always_zero": L["always_zero"][sl].mean(),
            })
    return pd.DataFrame(rows)


def plot_layer_survival(c1_df: pd.DataFrame, ledgers: dict, run_display: dict[str, str],
                        layers: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(layers))
    width = 0.8 / max(len(ledgers), 1)
    for k, label in enumerate(ledgers):
        sub = c1_df[c1_df["run"] == run_display[label]].set_index("layer").loc[layers]
        ax.bar(x + k * width, sub["frac_always_nonzero"], width,
               label=f"{run_display[label]} -- always active")
        ax.bar(x + k * width, sub["frac_ever_nonzero"] - sub["frac_always_nonzero"], width,
               bottom=sub["frac_always_nonzero"], alpha=0.4,
               label=f"{run_display[label]} -- sometimes active")
    ax.set(xticks=x + width * (len(ledgers) - 1) / 2, ylim=(0, 1),
           ylabel="fraction of layer params", title=title)
    ax.set_xticklabels(layers, rotation=60, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def print_static_vs_dynamic_sparsity(runs: dict, ledgers: dict, run_display: dict[str, str],
                                     zero_tol: float = 1e-8) -> None:
    """Per-draw sparsity vs the fraction of coords zero in EVERY draw. A big
    gap means the same overall sparsity is realised by different
    coordinates each draw (freeze/thaw churn), not a fixed pruned subset."""
    print("\n  static vs dynamic sparsity:")
    for label, ck in runs.items():
        s = ck["samples"].numpy()
        per_draw = (np.abs(s) < zero_tol).mean(1)
        L = ledgers[label]
        print(f"    {run_display[label]:16s}: per-draw sparsity {per_draw.mean():.3f} "
              f"(min {per_draw.min():.3f}, max {per_draw.max():.3f})  |  "
              f"zero in EVERY draw {L['always_zero'].mean():.3f}  |  "
              f"never zero {L['always_nonzero'].mean():.3f}  |  "
              f"churns (neither) {1 - L['always_zero'].mean() - L['always_nonzero'].mean():.3f}")


def accuracy_vs_sparsity_table(clean_df: pd.DataFrame, X_REF: torch.Tensor, runs: dict,
                               run_display: dict[str, str], zero_tol: float = 1e-8) -> pd.DataFrame:
    rows = [{
        "model": "MAP (pruned x_ref)",
        "accuracy": clean_df.loc["MAP (pruned x_ref)", "acc"],
        "weight_sparsity": float((np.abs(X_REF.numpy()) < zero_tol).mean()),
    }]
    for label, ck in runs.items():
        s = ck["samples"].numpy()
        rows.append({
            "model": run_display[label],
            "accuracy": clean_df.loc[run_display[label], "acc"],
            "weight_sparsity": float((np.abs(s) < zero_tol).mean()),
        })
    return pd.DataFrame(rows).set_index("model")


def displacement_table(runs: dict, ledgers: dict, run_display: dict[str, str],
                       layers: list[str], layer_slices: dict[str, tuple[int, int]],
                       X_REF: torch.Tensor, prior_std: np.ndarray, D: int) -> pd.DataFrame:
    """Per-(run, layer) posterior displacement from MAP, in prior-std units,
    on the coordinates that ever move off zero."""
    rows = []
    for label, ck in runs.items():
        s = ck["samples"].numpy()
        L = ledgers[label]
        mean_disp = (np.abs(s - X_REF.numpy()) / prior_std).mean(0)
        for lname in layers:
            a, b = layer_slices[lname]
            m = np.zeros(D, dtype=bool)
            m[a:b] = True
            m &= L["ever_nonzero"]
            if m.sum() == 0:
                continue
            rows.append({
                "run": run_display[label], "layer": lname, "n_moving": int(m.sum()),
                "mean_disp_priorstd": float(mean_disp[m].mean()),
                "median_disp_priorstd": float(np.median(mean_disp[m])),
                "p95_disp_priorstd": float(np.quantile(mean_disp[m], 0.95)),
            })
    return pd.DataFrame(rows)


def plot_displacement(c4_df: pd.DataFrame, runs: dict, run_display: dict[str, str],
                      layers: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(layers))
    width = 0.8 / max(len(runs), 1)
    for k, label in enumerate(runs):
        sub = c4_df[c4_df["run"] == run_display[label]].set_index("layer").reindex(layers)
        ax.bar(x + k * width, sub["mean_disp_priorstd"], width, label=run_display[label])
    ax.set(xticks=x + width * (len(runs) - 1) / 2, ylabel="mean |draw - MAP| (prior-std)",
           title=title)
    ax.set_xticklabels(layers, rotation=60, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ==========================================================================
# Rotation / pixel-noise corruption sweep (ported from *_pixel_noise.ipynb)
# ==========================================================================
def rotate_batch(X_norm: torch.Tensor, angle_deg: float, mean: float, std: float, **_) -> torch.Tensor:
    if angle_deg == 0:
        return X_norm.clone()
    img01 = X_norm * std + mean
    img01 = TF.rotate(img01, float(angle_deg),
                      interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
    return (img01 - mean) / std


def noise_batch(X_norm: torch.Tensor, sigma: float, mean: float, std: float, seed: int = 0) -> torch.Tensor:
    """Additive Gaussian noise, sigma in de-normalised [0,1] pixel units, NO
    clamp (network inputs, not display images). Izmailov et al. (2021)
    Fig 15 convention (unclipped N(0, sigma^2 I) on [0,1] images)."""
    if sigma == 0.0:
        return X_norm.clone()
    g = torch.Generator().manual_seed(seed)
    return X_norm + (sigma / std) * torch.randn(X_norm.shape, generator=g)


def block_metrics(block: np.ndarray, c: int, n_bins: int = 15) -> dict:
    """Per-image-pool metrics for one (model, digit, level) cell. `block`
    is [n_img, n_classes] posterior-mean probs; `c` is the true class.

    acc / p_true  -- robustness and confidence-in-correct-class.
    conf          -- mean max-prob = confidence in the PREDICTED class.
                     conf > acc is overconfidence; conf ~ acc is calibrated.
    entropy       -- mean predictive entropy (nats). A Bayesian model
                     should widen (entropy up) under shift; a point
                     estimate that stays sharp while wrong does not.
    ece           -- expected calibration error over this pool.
    """
    pred = block.argmax(1)
    conf = block.max(1)
    correct = (pred == c).astype(float)
    ent = -(block * np.log(np.clip(block, 1e-12, 1.0))).sum(1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(block)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())

    return dict(
        mean=block.mean(0),
        lo=np.percentile(block, 10, axis=0),
        hi=np.percentile(block, 90, axis=0),
        acc=float(correct.mean()),
        p_true=float(block[:, c].mean()),
        conf=float(conf.mean()),
        entropy=float(ent.mean()),
        ece=float(ece),
        n_img=len(block),
    )


def run_shift_sweep(levels: list, op: Callable, classes: list[int], y_test: torch.Tensor,
                    X_test: torch.Tensor, runs: dict, predict_probs: Callable,
                    n_per_class: int, n_draws_pool: int, pool_seed: int, noise_seed: int,
                    n_classes: int = 10) -> tuple[list, dict, torch.Tensor, dict]:
    """Build the pooled shifted image set, run n_draws_pool draws per
    sampler through predict_probs, and return (levels, spans, big_X, agg)
    where agg[label][(class, level)] holds mean/lo/hi P(class) plus
    acc/p_true/conf/entropy/ece/n_img, and agg[label][("pool", level)] the
    same pooled over all classes."""
    rng = np.random.default_rng(pool_seed)
    pool_idx = {c: rng.choice(np.where(y_test.numpy() == c)[0],
                              size=min(n_per_class, int((y_test == c).sum())),
                              replace=False)
                for c in classes}

    big_X, spans, row = [], {}, 0
    for c in classes:
        Xc = X_test[pool_idx[c]]
        for li, lv in enumerate(levels):
            big_X.append(op(Xc, lv, seed=noise_seed + 1000 * li + c))
            spans[(c, lv)] = (row, row + len(Xc))
            row += len(Xc)
    big_X = torch.cat(big_X)

    agg = {label: {} for label in runs}
    for label, ck in runs.items():
        S = ck["samples"]
        g = torch.Generator().manual_seed(0)
        draw_idx = torch.randperm(S.shape[0], generator=g)[:min(n_draws_pool, S.shape[0])]
        acc = torch.zeros(big_X.shape[0], n_classes)
        for j in draw_idx:
            acc += predict_probs(S[j], big_X)
        P = (acc / len(draw_idx)).numpy()
        for c in classes:
            for lv in levels:
                a, b = spans[(c, lv)]
                agg[label][(c, lv)] = block_metrics(P[a:b], c)
        for lv in levels:
            blocks, trues = [], []
            for c in classes:
                a, b = spans[(c, lv)]
                blocks.append(P[a:b])
                trues.append(np.full(b - a, c))
            Pool = np.concatenate(blocks)
            tru = np.concatenate(trues)
            pred = Pool.argmax(1)
            cf = Pool.max(1)
            corr = (pred == tru).astype(float)
            ent = -(Pool * np.log(np.clip(Pool, 1e-12, 1.0))).sum(1)
            bins = np.linspace(0.0, 1.0, 16)
            bi = np.clip(np.digitize(cf, bins) - 1, 0, 14)
            ece = sum(((bi == k).sum() / len(Pool)) *
                      abs(corr[bi == k].mean() - cf[bi == k].mean())
                      for k in range(15) if (bi == k).any())
            agg[label][("pool", lv)] = dict(
                acc=float(corr.mean()), conf=float(cf.mean()),
                p_true=float(Pool[np.arange(len(Pool)), tru].mean()),
                entropy=float(ent.mean()), ece=float(ece), n_img=len(Pool),
            )
    return levels, spans, big_X, agg


def shift_figure(kind: str, levels: list, agg: dict, runs: dict, run_display: dict[str, str],
                 classes: list[int], xlabel: str, title: str,
                 point_labels: tuple[str, ...] = ("map", "sgd")) -> plt.Figure:
    """One figure per shift. Columns: accuracy per class, P(true class),
    accuracy-vs-confidence (pooled), predictive entropy (pooled). Rows =
    models, point estimates (in point_labels) dotted, samplers solid."""
    labels = list(runs)
    class_col = {c: plt.cm.tab10(c % 10) for c in classes}
    col_titles = ["Accuracy per class", "P(true class)",
                  "Accuracy vs confidence", "Predictive entropy"]

    nrows, ncols = len(labels), 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.2, 1.85 * nrows + 0.7),
                             sharex="col", squeeze=False)
    fig.subplots_adjust(wspace=0.28, hspace=0.30,
                        left=0.07, right=0.99, top=0.88, bottom=0.16)

    for r, label in enumerate(labels):
        is_point = label in point_labels
        ls = ":" if is_point else "-"
        lw = 1.1 if is_point else 1.5

        ax = axes[r][0]
        for c in classes:
            ax.plot(levels, [agg[label][(c, lv)]["acc"] for lv in levels],
                    marker="o", ms=2.6, lw=lw, ls=ls, color=class_col[c],
                    label=str(c) if r == 0 else None)
        ax.axhline(1.0 / max(len(classes), 10), color="0.65", lw=0.7, ls="--")
        ax.set_ylim(-0.02, 1.02)

        ax = axes[r][1]
        for c in classes:
            ax.plot(levels, [agg[label][(c, lv)]["p_true"] for lv in levels],
                    marker="o", ms=2.6, lw=lw, ls=ls, color=class_col[c])
        ax.set_ylim(-0.02, 1.02)

        ax = axes[r][2]
        pa = [agg[label][("pool", lv)]["acc"] for lv in levels]
        pc = [agg[label][("pool", lv)]["conf"] for lv in levels]
        ax.plot(levels, pa, marker="o", ms=3, lw=1.6, color="0.20", label="accuracy")
        ax.plot(levels, pc, marker="s", ms=3, lw=1.6, color="#C44E52", label="confidence")
        ax.fill_between(levels, pa, pc, color="#C44E52", alpha=0.12, lw=0)
        ax.set_ylim(-0.02, 1.02)
        if r == 0:
            ax.legend(fontsize=6.5, loc="lower left", frameon=False, handlelength=1.2)

        ax = axes[r][3]
        ax.plot(levels, [agg[label][("pool", lv)]["entropy"] for lv in levels],
                marker="o", ms=3, lw=1.6, color="#4C72B0")
        ax.axhline(np.log(max(len(classes), 10)), color="0.65", lw=0.7, ls="--")
        ax.set_ylim(-0.05, np.log(max(len(classes), 10)) * 1.08)

        for j in range(ncols):
            axes[r][j].grid(alpha=0.25)
            axes[r][j].tick_params(labelsize=6.5, length=2, pad=1)
            if r == 0:
                axes[r][j].set_title(col_titles[j], fontsize=10, pad=4)
            if r == nrows - 1:
                axes[r][j].set_xlabel(xlabel, fontsize=10)
            else:
                axes[r][j].tick_params(labelbottom=False)
        axes[r][0].set_ylabel(run_display.get(label, label), fontsize=10)

    handles = [plt.Line2D([], [], color=class_col[c], marker="o", ms=3, lw=1.4) for c in classes]
    fig.legend(handles, [str(c) for c in classes], loc="lower center",
               title="true class", fontsize=10, title_fontsize=7.5,
               ncol=len(classes), frameon=False, bbox_to_anchor=(0.5, +0.05),
               handlelength=1.3, columnspacing=1.0)
    fig.suptitle(title, fontsize=10, y=0.97)
    return fig


def bar_grid_appendix(kind: str, levels: list, agg: dict, runs: dict, run_display: dict[str, str],
                      classes: list[int], sampler_colors: dict[str, str],
                      true_green: str = "#2CA02C", n_classes: int = 10,
                      point_labels: tuple[str, ...] = ("map", "sgd")) -> plt.Figure:
    """One figure per shift. Rows = true class, columns = shift level. Bars
    = pooled posterior-mean P(class), grouped by sampler (point estimates
    excluded); whiskers = 10-90% across the image pool."""
    all_classes = np.arange(n_classes)
    labels = [l for l in runs if l not in point_labels]
    n = len(labels)
    bw = 0.8 / n
    offs = [(-0.4 + bw / 2) + k * bw for k in range(n)]
    nrows, ncols = len(classes), len(levels)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(1.55 * ncols, 1.45 * nrows),
                             squeeze=False, sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.10, hspace=0.16,
                        left=0.07, right=0.99, top=0.92, bottom=0.11)

    for r, c in enumerate(classes):
        for cc, lv in enumerate(levels):
            ax = axes[r][cc]
            ax.axvspan(c - 0.5, c + 0.5, color=true_green, alpha=0.12, zorder=0)
            ax.axvline(c, color=true_green, lw=1.0, zorder=1)
            for k, label in enumerate(labels):
                A = agg[label][(c, lv)]
                x = all_classes + offs[k]
                edge = ["none"] * n_classes
                ew = [0.0] * n_classes
                edge[c] = true_green
                ew[c] = 1.2
                ax.bar(x, A["mean"], width=bw * (0.92 if n > 1 else 1.0),
                       color=sampler_colors.get(label, f"C{k}"),
                       edgecolor=edge, linewidth=ew, zorder=2,
                       label=run_display.get(label, label) if (r == 0 and cc == 0) else None)
                ax.errorbar(x, A["mean"],
                            yerr=[np.clip(A["mean"] - A["lo"], 0, None),
                                  np.clip(A["hi"] - A["mean"], 0, None)],
                            fmt="none", ecolor="black", elinewidth=0.6, capsize=1.0, zorder=3)
            ax.set_xlim(-0.6, n_classes - 0.4)
            ax.set_ylim(0, 1)
            ax.set_xticks(all_classes)
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(labelsize=7, length=2, pad=1)
            ax.set_xticklabels(all_classes if r == nrows - 1 else [], fontsize=10)
            if r == 0:
                ax.set_title(f"{lv:g}", fontsize=12, pad=3)
            if cc == 0:
                ax.set_yticklabels([0, "", 1], fontsize=12)
                ax.set_ylabel(str(c), fontsize=12, rotation=0, labelpad=8, va="center")

    fig.text(0.5, 0.06, "Predicted class", ha="center", fontsize=12)
    fig.text(0.012, 0.5, "True class", va="center", rotation="vertical", fontsize=12)
    if n > 1:
        fig.legend(loc="lower center", fontsize=12, ncol=n, frameon=False,
                   bbox_to_anchor=(0.5, +0.0), handlelength=1.4, columnspacing=1.4)
    return fig


def paper_style_figure(levels: list, agg: dict, run_display: dict[str, str],
                       colors: dict[str, str], xlabel: str, #title: str,
                       point_labels: tuple[str, ...] = ("map", "sgd"),
                       order: list[str] | None = None, n_classes: int = 10) -> plt.Figure:
    """1x4 panel pooled over all classes (i.e. NOT split by true digit),
    styled after Izmailov et al. (2021) Figure 15: one line per model, solid
    for posterior samplers, dashed for point estimates (MAP / SGD). Columns:
    accuracy, P(true class), accuracy-vs-confidence, predictive entropy --
    the same four quantities shift_figure plots per-digit-per-model, here
    pooled into a single overlaid axis per column so models compare directly
    on one set of axes (the paper's SGD-vs-HMC framing).

    `agg` is `run_shift_sweep`'s pooled aggregate: agg[label][("pool", lv)]
    holds acc/p_true/conf/entropy. `order` optionally fixes the legend/line
    draw order (e.g. point estimates first); defaults to agg's own key order.
    """
    labels = order if order is not None else list(agg)
    labels = [l for l in labels if l in agg]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0))

    def _line(ax, key, label):
        is_point = label in point_labels
        y = [agg[label][("pool", lv)][key] for lv in levels]
        ax.plot(levels, y, marker="o", ms=4, lw=2.0 if is_point else 1.6,
                ls="--" if is_point else "-", color=colors.get(label, None),
                label=run_display.get(label, label))

    ax = axes[0]
    for label in labels:
        _line(ax, "acc", label)
    #ax.axhline(1.0 / n_classes, color="0.6", lw=1.0, ls=":", label="chance")
    ax.set(xlabel=xlabel, #ylabel="Accuracy", 
           ylim=(-0.02, 1.02), title="Accuracy")

    ax = axes[1]
    for label in labels:
        _line(ax, "p_true", label)
    ax.set(xlabel=xlabel, #ylabel="P(true class)", 
           ylim=(-0.02, 1.02), title="P(true class)")

    # ax = axes[2]
    # for label in labels:
    #     #is_point = label in point_labels
    #     #acc = [agg[label][("pool", lv)]["acc"] for lv in levels]
    #     conf = [agg[label][("pool", lv)]["conf"] for lv in levels]
    #     c = colors.get(label, None)
    #     #ax.plot(levels, acc, marker="o", ms=4, lw=2.0 if is_point else 1.6,
    #     #        ls="--" if is_point else "-", color=c,
    #     #        label=f"{run_display.get(label, label)} acc")
    #     ax.plot(levels, conf, marker="s", ms=4, lw=1.2, ls="--", color=c, alpha=0.7,
    #             label=f"{run_display.get(label, label)} conf")
    # ax.set(xlabel=xlabel, ylabel="Confidence", ylim=(-0.02, 1.02),
    #       title="Confidence")

    ax = axes[2]
    for label in labels:
        _line(ax, "entropy", label)
    #ax.axhline(np.log(n_classes), color="0.6", lw=1.0, ls=":", label="max entropy")
    ax.set(xlabel=xlabel, #ylabel="Predictive entropy",
          ylim=(-0.05, np.log(n_classes) * 1.08), title="Predictive entropy")

    axes[0].legend(fontsize=8, frameon=False)
    axes[1].legend(fontsize=8, frameon=False)
    axes[2].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    #fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig
