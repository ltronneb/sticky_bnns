"""Coverage of the posterior predictive for notebooks/coverage_and_convergence.ipynb.

For test point x_i the posterior predictive is the S-component mixture

    F_i(y) = (1/S) sum_s Phi((y - f_s(x_i)) / sigma_s),

the same predictive as the NLL and CRPS in the UCI tables. The PIT value is
u_i = F_i(y_i). The central interval at level p is [F_i^-1((1-p)/2), F_i^-1((1+p)/2)],
and y_i lies in it exactly when |u_i - 1/2| <= p/2. Coverage C(p) is the share
of test points for which that holds, and W(p) is the mean interval width on the
original y scale.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.distributions import Normal

from sazz.gpu_friendly.scripts.uci_bnn_grid import load_raw_datasets, make_split, BASE_SEED
from .uci_results_utils import PAPER_ROWS, RESULTS_ROOT, DTYPE, predict

REPORT_LEVELS = (0.5, 0.8, 0.9, 0.95)
CURVE_LEVELS = np.linspace(0.01, 0.99, 99)
CACHE_PATH = RESULTS_ROOT / "_coverage_cache.pkl"
STD_NORMAL = Normal(0.0, 1.0)


# ---------------------------------------------------------------------------
# Per run
# ---------------------------------------------------------------------------

def pit_values(f: Tensor, sigma: Tensor, y: Tensor) -> Tensor:
    """u_i = F_i(y_i).   f [S, N], sigma [S], y [N] -> [N]."""
    return STD_NORMAL.cdf((y[None, :] - f) / sigma[:, None]).mean(0)


def predictive_quantiles(f: Tensor, sigma: Tensor, probs: Tensor, n_iter: int = 60) -> Tensor:
    """F_i^-1(pi) for every pi in probs by bisection, valid since F_i is increasing.
    The bracket [min_s f_s - 10 sigma_s, max_s f_s + 10 sigma_s] holds every quantile.  -> [L, N]."""
    L, N = probs.shape[0], f.shape[1]
    lo = (f - 10 * sigma[:, None]).min(0).values.expand(L, N).clone()
    hi = (f + 10 * sigma[:, None]).max(0).values.expand(L, N).clone()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        F = STD_NORMAL.cdf((mid[None] - f[:, None, :]) / sigma[:, None, None]).mean(0)
        below = F < probs[:, None]
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def coverage(u: np.ndarray, levels) -> np.ndarray:
    """C(p) = (1/N) sum_i 1{|u_i - 1/2| <= p/2}, for every p in levels."""
    return (np.abs(u[None, :] - 0.5) <= np.asarray(levels)[:, None] / 2).mean(1)


def run_coverage(run: dict, data: dict) -> dict:
    samples = run["samples"].to(DTYPE)
    f = predict(samples[:, :-1], run["layer_sizes"], run.get("activation", "tanh"), data["X_test"])
    sigma = samples[:, -1].exp()
    p = torch.tensor(REPORT_LEVELS, dtype=DTYPE)
    q = predictive_quantiles(f, sigma, torch.cat([(1 - p) / 2, (1 + p) / 2]))
    width = (q[len(p):] - q[:len(p)]).mean(1) * data["y_std"]
    return {"u": pit_values(f, sigma, data["y_test"]).numpy(), "width": width.numpy()}


# ---------------------------------------------------------------------------
# All runs, cached per file
# ---------------------------------------------------------------------------

def _file_key(p: Path) -> tuple:
    st = p.stat()
    return str(p), st.st_size, int(st.st_mtime)


def collect(archs=("small", "medium"), datasets=("boston", "concrete", "energy", "yacht"),
            use_cache: bool = True) -> list[dict]:
    """One record per (architecture, table row, dataset, split) with PIT values
    and widths. Files that are new or changed since the last call are recomputed."""
    cache = {}
    if use_cache and CACHE_PATH.exists():
        c = pickle.loads(CACHE_PATH.read_bytes())
        if c["levels"] == REPORT_LEVELS:
            cache = c["runs"]

    raw, recs, n_new = {}, [], 0
    for arch in archs:
        for label, folder, stem, w in PAPER_ROWS[arch]:
            row = label if w is None else f"{label} (w={w:g})"
            for dataset in datasets:
                for split_dir in sorted((RESULTS_ROOT / folder / dataset).glob("split_*")):
                    p = next((split_dir / n for n in (f"{stem}.pt", f"grid_{stem}.pt")
                              if (split_dir / n).exists()), None)
                    if p is None:
                        continue
                    key = _file_key(p)
                    if key not in cache:
                        run = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
                        if run["samples"].shape[0] == 0:
                            continue
                        if dataset not in raw:
                            raw.update(load_raw_datasets((dataset,)))
                        split_id = int(split_dir.name.split("_")[-1])
                        data = make_split(*raw[dataset], seed=BASE_SEED + split_id, dtype=DTYPE, device="cpu")
                        assert np.isclose(run["y_std"], data["y_std"]), f"{p}, split does not match the run"
                        cache[key] = run_coverage(run, data)
                        n_new += 1
                    recs.append({"arch": arch, "row": row, "stem": stem, "w": w, "dataset": dataset,
                                 "split": split_dir.name, **cache[key]})
    if use_cache and n_new:
        CACHE_PATH.write_bytes(pickle.dumps({"levels": REPORT_LEVELS, "runs": cache}))
    print(f"{len(recs)} runs ({n_new} computed, {len(recs) - n_new} from cache)")
    return recs


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def per_split_frame(recs: list[dict]) -> pd.DataFrame:
    """One row per run with C(p), W(p) and the calibration error
    CE = mean over p in CURVE_LEVELS of |C(p) - p|."""
    out = []
    for r in recs:
        d = {k: r[k] for k in ("arch", "row", "dataset", "split")}
        d["N_test"] = len(r["u"])
        for p, c, wd in zip(REPORT_LEVELS, coverage(r["u"], REPORT_LEVELS), r["width"]):
            d[f"C({p:g})"], d[f"W({p:g})"] = c, wd
        d["CE"] = np.abs(coverage(r["u"], CURVE_LEVELS) - CURVE_LEVELS).mean()
        out.append(d)
    return pd.DataFrame(out)


def summary_table(df: pd.DataFrame, arch: str, dataset: str, cols=None, dec: int = 2) -> pd.DataFrame:
    """Mean and standard error over splits, rows in table order."""
    cols = cols or [c for c in df.columns if c[:2] in ("C(", "W(")] + ["CE"]
    sub = df[(df.arch == arch) & (df.dataset == dataset)]
    order = list(dict.fromkeys(sub["row"]))
    g = sub.groupby("row", sort=False)[cols]
    m, se = g.mean(), g.sem()
    tab = m.map(lambda v: f"{v:.{dec}f}") + " ± " + se.map(lambda v: f"{v:.{dec}f}")
    tab.insert(0, "splits", g.size())
    return tab.loc[order]


def overview(df: pd.DataFrame, arch: str, col: str = "CE", dec: int = 3) -> pd.DataFrame:
    """One statistic for every model (rows) and dataset (columns), mean over splits."""
    sub = df[df.arch == arch]
    order = list(dict.fromkeys(sub["row"]))
    tab = sub.pivot_table(index="row", columns="dataset", values=col, aggfunc="mean")
    return tab.loc[order].round(dec)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

FAMILY_COLORS = {"zigzag": "C2", "boomerang": "C0", "nuts": "k", "lbbnn": "C3"}


def style(stem: str, w) -> dict:
    """Colour by family, solid for sticky w=0.3, dotted for sticky w=0.1, dashed otherwise."""
    fam = stem.replace("sticky_", "")
    ls = "--" if not stem.startswith("sticky") else ("-" if w == 0.3 else ":")
    return {"color": FAMILY_COLORS.get(fam, "0.5"), "ls": ls}


def calibration_figure(recs: list[dict], arch: str, datasets, rows=None, figsize=None):
    """C(p) - p against p, one panel per dataset, PIT values pooled over splits.
    The grey band is +-2 binomial standard errors for the pooled test set."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(datasets), figsize=figsize or (4.2 * len(datasets), 3.8),
                             sharey=True, squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = [r for r in recs if r["arch"] == arch and r["dataset"] == dataset
               and (rows is None or r["row"] in rows)]
        by_row = {}
        for r in sub:
            by_row.setdefault(r["row"], []).append(r)
        N = 0
        for row, rs in by_row.items():
            u = np.concatenate([r["u"] for r in rs])
            N = len(u)
            ax.plot(CURVE_LEVELS, coverage(u, CURVE_LEVELS) - CURVE_LEVELS, lw=1.8,
                    label=row, **style(rs[0]["stem"], rs[0]["w"]))
        if N:
            band = 2 * np.sqrt(CURVE_LEVELS * (1 - CURVE_LEVELS) / N)
            ax.fill_between(CURVE_LEVELS, -band, band, color="0.8", alpha=0.5, lw=0)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("Nominal level $p$")
    axes[0, 0].set_ylabel("$C(p) - p$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    return fig
