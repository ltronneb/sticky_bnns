"""Helpers for notebooks/uci_full_split.ipynb."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.distributions import Normal

from sazz.gpu_friendly.models.neural_networks import get_activation
from sazz.gpu_friendly.scripts.uci_bnn_grid import load_raw_datasets, make_split, BASE_SEED

RESULTS_ROOT = Path("results/paper_v2")
DTYPE = torch.float64

# Row order of every table follows this dict.
SAMPLER_LABELS = {
    "zigzag":                "ZigZag",
    "sticky_zigzag":         "Sticky ZigZag",
    "boomerang":             "Boomerang",
    "sticky_boomerang":      "Sticky Boomerang",
    "nuts":                  "NUTS",
    "nuts_horseshoe":        "NUTS-HS",
    "lbbnn":                 "LBBNN",
    "tf_boomerang":          "TF Boomerang",
    "tf_bps":                "TF BPS",
}

CRPS_SUBSAMPLE = 256
PRED_CHUNK = 2048

# Bump when a metric definition changes so stale cache rows get recomputed.
CACHE_VERSION = 1
CACHE_NAME = "_metrics_cache.csv"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def weight_mask(layer_sizes) -> Tensor:
    """Bool [D], True on weight-matrix coords. Biases and the trailing
    log_sigma are False. Layout matches FFN.named_parameters()."""
    mask = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        mask += [True] * (n_in * n_out)
        mask += [False] * n_out
    mask += [False]
    return torch.tensor(mask, dtype=torch.bool)


@torch.no_grad()
def predict(samples: Tensor, layer_sizes, activation: str, X: Tensor,
            chunk: int = PRED_CHUNK) -> Tensor:
    """Network outputs [S, N] for every draw, batched over draws with bmm.
    Same result as looping functional_call(bm.module, ...) per draw, without
    having to build the BayesianModule (build_target runs a full MAP fit)."""
    act = get_activation(activation)
    n_layers = len(layer_sizes) - 1
    out = []
    for s in samples.split(chunk):
        B = s.shape[0]
        h = X.expand(B, *X.shape)
        idx = 0
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            W = s[:, idx:idx + n_in * n_out].view(B, n_out, n_in)
            idx += n_in * n_out
            b = s[:, idx:idx + n_out]
            idx += n_out
            h = torch.baddbmm(b[:, None, :], h, W.transpose(1, 2))
            if i < n_layers - 1:
                h = act(h)
        out.append(h.squeeze(-1))
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Metrics, all on the original y scale. The predictive is the S-component
# mixture (1/S) sum_s N(y, f_s(x), sigma_s^2).
# ---------------------------------------------------------------------------

def rmse(y: Tensor, preds: Tensor, y_std: float) -> float:
    return float(((preds.mean(0) - y) ** 2).mean().sqrt()) * y_std


def nll_mixture(y: Tensor, preds: Tensor, sigma: Tensor, y_std: float) -> float:
    lp = Normal(preds, sigma[:, None]).log_prob(y)                  # [S, N]
    ll = torch.logsumexp(lp, dim=0) - math.log(preds.shape[0])      # [N]
    return float(-ll.mean() + math.log(y_std))


def _crps_A(m: Tensor, s: Tensor) -> Tensor:
    """E|X - m| for X ~ N(0, s^2)."""
    d = Normal(0.0, 1.0)
    z = m / s
    return m * (2 * d.cdf(z) - 1) + 2 * s * d.log_prob(z).exp()


def crps_mixture(y: Tensor, preds: Tensor, sigma: Tensor, y_std: float,
                 n_sub: int = CRPS_SUBSAMPLE, seed: int = 0) -> float:
    """Exact Gaussian-mixture CRPS
        E_s A(y - mu_s, sigma_s) - 0.5 E_{s,s'} A(mu_s - mu_s', sqrt(sigma_s^2 + sigma_s'^2)).
    O(S^2) in memory, so it runs on a fixed random subsample of n_sub draws
    (same seed for every sampler)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(preds.shape[0], generator=g)[:n_sub]

    mu = preds[idx] * y_std                                         # [M, N]
    sd = (sigma[idx] * y_std)[:, None].expand_as(mu)                # [M, N]
    yt = y * y_std

    term1 = _crps_A(yt[None, :] - mu, sd).mean(0)
    diff = mu[:, None, :] - mu[None, :, :]
    s2 = (sd[:, None, :] ** 2 + sd[None, :, :] ** 2).sqrt()
    term2 = _crps_A(diff, s2).mean((0, 1))
    return float((term1 - 0.5 * term2).mean())


def run_metrics(run: dict, data: dict) -> dict:
    samples = run["samples"].to(DTYPE)
    preds = predict(samples[:, :-1], run["layer_sizes"], run["activation"], data["X_test"])
    sigma = samples[:, -1].exp()
    y, y_std = data["y_test"], data["y_std"]
    w = weight_mask(run["layer_sizes"])
    return {
        "RMSE":       rmse(y, preds, y_std),
        "NLL":        nll_mixture(y, preds, sigma, y_std),
        "CRPS":       crps_mixture(y, preds, sigma, y_std),
        "grad_evals": int(run.get("gradient_evals") or 0),
        "sparsity":   100 * float((samples[:, w] == 0).double().mean()),
        "n_samples":  int(samples.shape[0]),
    }


# ---------------------------------------------------------------------------
# Loading with a per-file cache
# ---------------------------------------------------------------------------

def _file_key(p: Path) -> tuple:
    st = p.stat()
    return str(p), st.st_size, int(st.st_mtime)


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    c = pd.read_csv(path)
    c = c[c["cache_version"] == CACHE_VERSION]
    return {(r["path"], r["size"], r["mtime"]): r for r in c.to_dict("records")}


def collect(variant: str, datasets=None, samplers=None, use_cache: bool = True,
            verbose: bool = True) -> pd.DataFrame:
    """One row per (dataset, split, sampler) run under RESULTS_ROOT/variant.
    Metrics are cached next to the runs and only recomputed for files that
    are new or changed since the last call."""
    root = RESULTS_ROOT / variant
    cache_path = root / CACHE_NAME
    cache = _load_cache(cache_path) if use_cache else {}

    datasets = datasets or sorted(p.name for p in root.iterdir() if p.is_dir())
    samplers = samplers or list(SAMPLER_LABELS)

    raw, rows, n_new = {}, [], 0
    for dataset in datasets:
        for split_dir in sorted((root / dataset).glob("split_*")):
            split_id = int(split_dir.name.split("_")[-1])
            data = None
            for sampler in samplers:
                p = split_dir / f"{sampler}.pt"
                if not p.exists():
                    continue
                key = _file_key(p)
                if key in cache:
                    rows.append(cache[key])
                    continue

                run = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
                if run["samples"].shape[0] == 0:
                    if verbose:
                        print(f"  skip {variant}/{dataset}/split_{split_id:02d}/{sampler}, 0 samples")
                    continue
                if data is None:
                    if dataset not in raw:
                        raw.update(load_raw_datasets((dataset,)))
                    X, y = raw[dataset]
                    data = make_split(X, y, seed=BASE_SEED + split_id, dtype=DTYPE, device="cpu")
                assert np.isclose(run["y_std"], data["y_std"]), f"{p}, split does not match the run"

                m = run_metrics(run, data)
                rows.append({
                    "path": key[0], "size": key[1], "mtime": key[2],
                    "cache_version": CACHE_VERSION,
                    "variant": variant, "dataset": dataset,
                    "split_id": split_id, "sampler": sampler, **m,
                })
                n_new += 1
                if verbose:
                    print(f"  {variant}/{dataset}/split_{split_id:02d}/{sampler:<22}"
                          f"RMSE={m['RMSE']:.3f}  NLL={m['NLL']:.3f}  CRPS={m['CRPS']:.3f}  "
                          f"sparsity={m['sparsity']:.1f}%")

    df = pd.DataFrame(rows)
    if use_cache and n_new:
        old = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
        pd.concat([old, df]).drop_duplicates(["path", "size", "mtime", "cache_version"], keep="last") \
            .to_csv(cache_path, index=False)
    if verbose:
        print(f"{variant}: {len(df)} runs ({n_new} computed, {len(df) - n_new} from cache)")
    return df.drop(columns=["path", "size", "mtime", "cache_version"], errors="ignore")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

TABLE_COLS = ["RMSE", "NLL", "CRPS", "Grad evals (M)", "Sparsity (%)"]


def _pm(x: pd.Series, dec: int) -> str:
    if len(x) < 2:
        return f"{x.mean():.{dec}f}"
    return f"{x.mean():.{dec}f} ± {x.sem():.{dec}f}"


def summary_table(df: pd.DataFrame, dataset: str, dec: int = 2) -> pd.DataFrame:
    """Mean ± SEM over splits, one row per sampler."""
    sub = df[df["dataset"] == dataset]
    n_max = sub.groupby("sampler").size().max()
    rows = {}
    for sampler in [s for s in SAMPLER_LABELS if s in set(sub["sampler"])]:
        g = sub[sub["sampler"] == sampler]
        if len(g) < n_max:
            print(f"  warning, {dataset}/{sampler} has {len(g)} of {n_max} splits")
        sp = g["sparsity"]
        rows[SAMPLER_LABELS[sampler]] = {
            "RMSE": _pm(g["RMSE"], dec),
            "NLL":  _pm(g["NLL"], dec),
            "CRPS": _pm(g["CRPS"], dec),
            "Grad evals (M)": f"{(g['grad_evals'] / 1e6).mean():.2f}",
            "Sparsity (%)":   _pm(sp, 1) if (sp > 0).any() else "0",
        }
    table = pd.DataFrame.from_dict(rows, orient="index")[TABLE_COLS]
    table.index.name = "Sampler"
    return table


def to_latex(table: pd.DataFrame, caption: str = None, label: str = None) -> str:
    t = table.rename(columns={"Grad evals (M)": r"Grad evals ($\times 10^6$)",
                              "Sparsity (%)": r"Sparsity (\%)"})
    t = t.apply(lambda c: c.str.replace("±", r"$\pm$", regex=False)).reset_index()
    return t.to_latex(index=False, escape=False, caption=caption, label=label, position="t")
