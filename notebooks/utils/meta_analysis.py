"""Mixed-effects meta-analysis of the UCI results (notebooks/meta_analysis.ipynb).

Each table compares the methods of PAPER_ROWS across all datasets and splits at once,
instead of one dataset at a time. One row per (method, dataset, split) enters a regression
with the method as a fixed effect, relative to a baseline method, and random intercepts
for the dataset and for the split within the dataset. The intercepts absorb that datasets
differ in difficulty and that all methods share the same split.

    RMSE, CRPS   Gaussian mixed model on the log metric, so a coefficient b is a ratio,
                 reported as the percentage difference 100 (exp(b) - 1) to the baseline.
    NLL          Gaussian mixed model on the NLL itself (it can be negative), so b is a
                 difference in NLL units.
    Coverage     Binomial mixed model on whether each test response lies inside the
                 central 90% predictive interval, fitted by variational Bayes. Reported
                 as the coverage implied for an average dataset and split.

Positive differences in RMSE, CRPS and NLL mean worse than the baseline.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import expit
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from . import coverage_utils as cu
from .uci_results_utils import PAPER_ROWS, collect

DATASETS = ("boston", "concrete", "energy", "yacht")
BASELINE = "Sticky ZigZag (w=0.3)"
LEVEL = 0.9


def method_label(label: str, w) -> str:
    return label if w is None else f"{label} (w={w:g})"


def results_frame(archs=("small", "medium"), datasets=DATASETS) -> pd.DataFrame:
    """One row per (arch, method, dataset, split) with RMSE, NLL, CRPS and sparsity."""
    rows = []
    for arch in archs:
        for label, folder, stem, w in PAPER_ROWS[arch]:
            df = collect(folder, datasets=list(datasets), samplers=[stem], verbose=False)
            for _, r in df.iterrows():
                rows.append(dict(arch=arch, method=method_label(label, w), dataset=r["dataset"],
                                 split=int(r["split_id"]), RMSE=r["RMSE"], NLL=r["NLL"],
                                 CRPS=r["CRPS"], sparsity=r["sparsity"]))
    return pd.DataFrame(rows)


def coverage_points(archs=("small", "medium"), datasets=DATASETS, level=LEVEL) -> pd.DataFrame:
    """One row per test point and run, covered = 1 if the response lies inside the
    central `level` predictive interval (from the PIT values of coverage_utils)."""
    out = []
    for r in cu.collect(archs, datasets):
        covered = (np.abs(r["u"] - 0.5) <= level / 2).astype(int)
        out.append(pd.DataFrame(dict(arch=r["arch"], method=r["row"].replace(" (w=", " (w="),
                                     dataset=r["dataset"], split=int(r["split"].split("_")[-1]),
                                     covered=covered)))
    return pd.concat(out, ignore_index=True)


def _formula(response: str, baseline: str) -> str:
    return f"{response} ~ C(method, Treatment(reference={baseline!r}))"


def _method_name(term: str) -> str:
    return term.split("[T.")[-1].rstrip("]")


def fit_lmm(df: pd.DataFrame, metric: str, log: bool, baseline: str = BASELINE) -> pd.DataFrame:
    """Gaussian mixed model of the (log) metric on the method, random intercepts for the
    dataset and the split within it. Returns one row per method against the baseline."""
    d = df.copy()
    d["y"] = np.log(d[metric]) if log else d[metric]
    m = smf.mixedlm(_formula("y", baseline), d, groups=d["dataset"], re_formula="1",
                    vc_formula={"split": "0 + C(split)"}).fit(reml=True)
    ci = m.conf_int()
    rows = []
    for term in m.fe_params.index:
        if not term.startswith("C(method"):
            continue
        b, lo, hi, p = m.fe_params[term], ci.loc[term, 0], ci.loc[term, 1], m.pvalues[term]
        if log:
            b, lo, hi = (100 * (np.exp(v) - 1) for v in (b, lo, hi))
        rows.append(dict(method=_method_name(term), estimate=b, lo=lo, hi=hi, p=p))
    unit = "% vs baseline" if log else "difference vs baseline"
    out = pd.DataFrame(rows).set_index("method")
    out.attrs.update(metric=metric, unit=unit, baseline=baseline, converged=m.converged)
    return out


def fit_coverage(points: pd.DataFrame, baseline: str = BASELINE, level: float = LEVEL) -> pd.DataFrame:
    """Binomial mixed model of per-point coverage on the method, random intercepts for the
    dataset and for the split within it (variational Bayes). Returns the coverage implied
    for an average dataset and split, with an approximate 95% interval."""
    d = points.copy()
    d["ds_split"] = d["dataset"] + "_" + d["split"].astype(str)
    m = BinomialBayesMixedGLM.from_formula(
        _formula("covered", baseline), {"dataset": "0 + C(dataset)", "split": "0 + C(ds_split)"}, d,
    ).fit_vb()
    names, mean, sd = m.model.exog_names, m.fe_mean, m.fe_sd
    b0, sd0 = mean[0], sd[0]
    rows = [dict(method=baseline, coverage=expit(b0), lo=expit(b0 - 1.96 * sd0), hi=expit(b0 + 1.96 * sd0))]
    for name, b, s in zip(names[1:], mean[1:], sd[1:]):
        # the method's own uncertainty only, the intercept's is shown in the baseline row
        rows.append(dict(method=_method_name(name), coverage=expit(b0 + b),
                         lo=expit(b0 + b - 1.96 * s), hi=expit(b0 + b + 1.96 * s)))
    out = pd.DataFrame(rows).set_index("method")
    out.attrs.update(metric=f"coverage of the central {level:.0%} interval", baseline=baseline)
    return out


def summary(df: pd.DataFrame, points: pd.DataFrame, arch: str, baseline: str = BASELINE) -> pd.DataFrame:
    """All metrics for one architecture side by side, one row per method."""
    d, pts = df[df.arch == arch], points[points.arch == arch]
    parts = {"RMSE (%)": fit_lmm(d, "RMSE", log=True, baseline=baseline),
             "CRPS (%)": fit_lmm(d, "CRPS", log=True, baseline=baseline),
             "NLL (diff)": fit_lmm(d, "NLL", log=False, baseline=baseline)}
    cols = {}
    for name, t in parts.items():
        cols[name] = t.apply(lambda r: f"{r.estimate:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]"
                             if "NLL" in name else f"{r.estimate:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]", axis=1)
    cov = fit_coverage(pts, baseline=baseline)
    cols[f"C({LEVEL:g})"] = cov.apply(lambda r: f"{100 * r.coverage:.0f}% [{100 * r.lo:.0f}, {100 * r.hi:.0f}]", axis=1)
    out = pd.DataFrame(cols)
    order = [method_label(l, w) for l, _, _, w in PAPER_ROWS[arch]]
    out = out.reindex([m for m in order if m in out.index])
    out.loc[baseline, ["RMSE (%)", "CRPS (%)", "NLL (diff)"]] = "baseline"
    return out


def to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    t = table.copy()
    t = t.apply(lambda c: c.str.replace("%", r"\%", regex=False).str.replace("[", "$[$", regex=False)
                .str.replace("]", "$]$", regex=False))
    t.columns = [c.replace("%", r"\%") for c in t.columns]
    return t.to_latex(escape=False, caption=caption, label=label, position="t")
