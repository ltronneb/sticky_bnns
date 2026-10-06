"""Helpers for notebooks/uci_convergence.ipynb.

Four chains per PDMP run for UCI boston split 0, small and medium network, at 1e6 or
1e7 gradient evaluations per chain (chain_paths, budget). Every chain starts from its
own MAP and Laplace covariance. All chains keep S = 4000 draws, uniform in time after
a 20 % time burn-in, stored in time order. NUTS has 4 runs of 4 chains, one run per
MAP 0 to 3.
"""

import math
from pathlib import Path

import numpy as np
import torch

from .uci_results_utils import predict, weight_mask

V2 = Path("results/paper_v2")
# gradient budget per PDMP chain -> root of the convergence chains
CONV = {"1e6": Path("results/paper_v2_convergence"), "1e7": Path("results/paper_v2_convergence_10M")}

# (network, w) -> folder of the published run (chain 0 at 1e6) and of the convergence chains
CHAIN0 = {("small", 0.3): V2 / "shallow", ("small", 0.1): V2 / "shallow_piw_0.1",
          ("medium", 0.3): V2 / "deep_narrow", ("medium", 0.1): V2 / "deep_narrow" / "piw_0.1"}
FOLDER = {("small", 0.3): "small", ("small", 0.1): "small_piw0.1",
          ("medium", 0.3): "medium", ("medium", 0.1): "medium_piw0.1"}
NUTS = {"small": V2 / "shallow", "medium": V2 / "deep_narrow"}
SAMPLERS_AT = {0.3: ["zigzag", "sticky_zigzag", "boomerang", "sticky_boomerang"],
               0.1: ["sticky_zigzag", "sticky_boomerang"]}


def chain_paths(network: str, w: float, sampler: str, dataset: str = "boston", split: int = 0,
                budget: str = "1e6") -> list[tuple[int, Path]]:
    """(chain id, path) of every chain. Chain c starts at map_c.
    1e6  chain 0 is the published run, chains 1 to 3 come from step 11.
    1e7  every chain in its own folder, medium 0 to 3 (chain 0 continues the published
         run), small 1 to 4 (step 11d, no chain 0)."""
    sd = f"{dataset}/split_{split:02d}"
    root = CONV[budget] / FOLDER[network, w] / sd
    if budget == "1e6":
        return ([(0, CHAIN0[network, w] / sd / f"{sampler}.pt")]
                + [(c, root / f"chain_{c}" / f"{sampler}.pt") for c in (1, 2, 3)])
    found = sorted((int(p.parent.name.split("_")[1]), p) for p in root.glob(f"chain_*/{sampler}.pt"))
    assert found, f"no {sampler} chains under {root}"
    return found


def nuts_path(network: str, dataset: str = "boston", split: int = 0) -> Path:
    return NUTS[network] / f"{dataset}/split_{split:02d}" / "nuts.pt"


# every NUTS run is 4 chains of 1000 draws from one MAP, stored one after another
N_NUTS_CHAINS = 4


def nuts_paths(network: str, dataset: str = "boston", split: int = 0,
               maps: tuple[int, ...] = (0, 1, 2, 3)) -> list[Path]:
    """NUTS from the given MAPs, the ones that exist. MAP 0 is the published run, MAPs 1 to
    3 come from the 1e6 folders (Status step 11b), MAP 4 (small only) from the 1e7 folder,
    so the small 1e7 chains 1 to 4 have NUTS from the same MAPs with maps=(1, 2, 3, 4)."""
    sd = f"{dataset}/split_{split:02d}"

    def path(c):
        if c == 0:
            return nuts_path(network, dataset, split)
        return CONV["1e6" if c < 4 else "1e7"] / FOLDER[network, 0.3] / sd / f"chain_{c}" / "nuts.pt"

    return [p for p in map(path, maps) if p.exists()]


def nuts_chains(stats: dict, key: str) -> np.ndarray:
    """A NUTS statistic split into its chains, [N_NUTS_CHAINS, draws per chain, ...]."""
    a = stats[key]
    return a.reshape(N_NUTS_CHAINS, -1, *a.shape[1:])


@torch.no_grad()
def draw_stats(path: Path, data: dict) -> dict:
    """Per-draw statistics that do not depend on how hidden units are ordered, plus the
    predictions at the test points and the run's MAP (x_ref) evaluated the same way."""
    run = torch.load(path, map_location="cpu", weights_only=False)
    z, L = run["samples"].double(), run["layer_sizes"]
    out = {"log sigma": z[:, -1].numpy(), "grads": int(run.get("gradient_evals") or run.get("grad_evals") or 0)}
    for split in ("train", "test"):
        X, y = data[f"X_{split}"], data[f"y_{split}"]
        f = predict(z[:, :-1], L, "tanh", X, chunk=500)
        out[f"{split} RMSE"] = (((f - y) ** 2).mean(1).sqrt() * data["y_std"]).numpy()
        # if split == "test":
        #     out["f_test"] = (f * data["y_std"]).numpy()
        # if run.get("x_ref") is not None:
        #     f_map = predict(run["x_ref"].double()[None, :-1], L, "tanh", X)
        #     out[f"MAP {split} RMSE"] = float(((f_map - y) ** 2).mean().sqrt() * data["y_std"])
        if split == "test":
            out["f_test"] = (f * data["y_std"]).numpy()
            out["test mean prediction"] = out["f_test"].mean(1)          # per draw, y units
            # NLL of each draw's own Gaussian, averaged over test points, on the original y
            # scale. Not the NLL of the predictive mixture in the tables, which is lower.
            s = z[:, -1:].exp()
            out["test NLL"] = (0.5 * math.log(2 * math.pi) + s.log() + math.log(data["y_std"])
                               + (f - y) ** 2 / (2 * s ** 2)).mean(1).numpy()
        if run.get("x_ref") is not None:
            f_map = predict(run["x_ref"].double()[None, :-1], L, "tanh", X)
            out[f"MAP {split} RMSE"] = float(((f_map - y) ** 2).mean().sqrt() * data["y_std"])
            if split == "test":
                out["MAP test mean prediction"] = float(f_map.mean() * data["y_std"])

    if run.get("x_ref") is not None:
        out["MAP log sigma"] = float(run["x_ref"][-1])
    w = weight_mask(L)
    out["zero %"] = 100 * (z[:, w] == 0).double().mean(1).numpy()
    return out


# Large network, full batch, 1e7 gradients per PDMP chain (Status step 14)
LARGE = V2 / "deep_wide_convergence_fullbatch" / "deep_wide"


def large_paths(sampler: str, dataset: str = "boston", split: int = 0,
                chains: tuple[int, ...] = (0, 1, 2, 3)) -> list[tuple[int, Path]]:
    """(chain id, path) of the finished chains. The gpu_friendly driver writes
    grid_<sampler>.pt, renamed files drop the prefix, both are accepted."""
    sd = LARGE / f"{dataset}/split_{split:02d}"
    out = []
    for c in chains:
        p = next((sd / f"chain_{c}" / n for n in (f"{sampler}.pt", f"grid_{sampler}.pt")
                  if (sd / f"chain_{c}" / n).exists()), None)
        if p is not None:
            out.append((c, p))
    return out


def cached_draw_stats(path: Path, data: dict, cache_path: Path) -> dict:
    """draw_stats with a cache keyed on (path, size, mtime), since a large run is 0.7 to
    1.4 GB and takes a while to load and predict."""
    import pickle
    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    st = path.stat()
    key = (str(path), st.st_size, int(st.st_mtime))
    if key not in cache or "test NLL" not in cache[key]:   # entries from before test NLL existed
        cache[key] = draw_stats(path, data)
        cache_path.write_bytes(pickle.dumps(cache))
    return cache[key]


def lag1(x: np.ndarray) -> float:
    x = x - x.mean()
    return float((x[1:] * x[:-1]).mean() / x.var())
