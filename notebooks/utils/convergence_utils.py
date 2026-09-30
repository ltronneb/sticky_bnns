"""Helpers for notebooks/uci_convergence.ipynb.

Four chains per PDMP run for UCI boston split 0, small and medium network. Chain 0 is
the published run, chains 1 to 3 (Status step 11) each start from their own random
MAP and Laplace covariance. All chains keep S = 4000 draws, uniform in time after a
20 % time burn-in, stored in time order. NUTS has one chain, started at the same MAP
as chain 0.
"""

from pathlib import Path

import numpy as np
import torch

from .uci_results_utils import predict, weight_mask

V2 = Path("results/paper_v2")
CONV = Path("results/paper_v2_convergence")

# (network, w) -> folder of chain 0 (published) and of chains 1 to 3
CHAIN0 = {("small", 0.3): V2 / "shallow", ("small", 0.1): V2 / "shallow_piw_0.1",
          ("medium", 0.3): V2 / "deep_narrow", ("medium", 0.1): V2 / "deep_narrow" / "piw_0.1"}
CHAINS = {("small", 0.3): CONV / "small", ("small", 0.1): CONV / "small_piw0.1",
          ("medium", 0.3): CONV / "medium", ("medium", 0.1): CONV / "medium_piw0.1"}
NUTS = {"small": V2 / "shallow", "medium": V2 / "deep_narrow"}
SAMPLERS_AT = {0.3: ["zigzag", "sticky_zigzag", "boomerang", "sticky_boomerang"],
               0.1: ["sticky_zigzag", "sticky_boomerang"]}


def chain_paths(network: str, w: float, sampler: str, dataset: str = "boston", split: int = 0) -> list[Path]:
    sd = f"{dataset}/split_{split:02d}"
    return ([CHAIN0[network, w] / sd / f"{sampler}.pt"]
            + [CHAINS[network, w] / sd / f"chain_{c}" / f"{sampler}.pt" for c in (1, 2, 3)])


def nuts_path(network: str, dataset: str = "boston", split: int = 0) -> Path:
    return NUTS[network] / f"{dataset}/split_{split:02d}" / "nuts.pt"


# every NUTS run is 4 chains of 1000 draws from one MAP, stored one after another
N_NUTS_CHAINS = 4


def nuts_paths(network: str, dataset: str = "boston", split: int = 0) -> list[Path]:
    """NUTS from MAP 0 (the published run) and from MAPs 1 to 3 (Status step 11b), the
    ones that exist."""
    sd = f"{dataset}/split_{split:02d}"
    paths = [nuts_path(network, dataset, split)] + [CHAINS[network, 0.3] / sd / f"chain_{c}" / "nuts.pt"
                                                    for c in (1, 2, 3)]
    return [p for p in paths if p.exists()]


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
        if split == "test":
            out["f_test"] = (f * data["y_std"]).numpy()
        if run.get("x_ref") is not None:
            f_map = predict(run["x_ref"].double()[None, :-1], L, "tanh", X)
            out[f"MAP {split} RMSE"] = float(((f_map - y) ** 2).mean().sqrt() * data["y_std"])
    w = weight_mask(L)
    out["zero %"] = 100 * (z[:, w] == 0).double().mean(1).numpy()
    return out


def lag1(x: np.ndarray) -> float:
    x = x - x.mean()
    return float((x[1:] * x[:-1]).mean() / x.var())
