"""Tables of the predictive metrics in the paper, from the result files of the
experiment scripts (definitions in metrics.py).

    python -m sazz.paper_ready.scripts.analyse toy --results results/toy_bnns
    python -m sazz.paper_ready.scripts.analyse uci --results results/uci
    python -m sazz.paper_ready.scripts.analyse images --results results/images

toy     per data set, Sparsity, RMSE, NLL, CRPS and 90% coverage. Evaluated on the
        training range |x| <= 3, except Hernandez on the full test grid.
uci     per network (folder under --results) and data set, the mean and standard
        error over the splits.
images  per model, the SGD network, the pruned MAP and the sticky samplers, with
        Acc, NLL, ECE and Brier of the model-averaged class probabilities.
--latex also prints each table as LaTeX.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data import toy_data, uci_split
from ..metrics import classification_metrics, regression_metrics, sparsity
from ..models.bnn import freezable
from ..models.networks import FFN

ORDER = ["zigzag", "boomerang", "nuts", "sticky_zigzag", "sticky_boomerang", "lbbnn"]
LABELS = {"zigzag": "ZigZag", "boomerang": "Boomerang", "nuts": "NUTS", "sticky_zigzag": "Sticky ZigZag",
          "sticky_boomerang": "Sticky Boomerang", "lbbnn": "LBBNN"}
TOY_X_RANGE = {"hernandez": None, "gap": 3.0, "sharp": 3.0, "multiscale": 3.0}
N_DRAWS = 4000


def run_file(folder: Path, sampler: str):
    """The result file of a sampler, with or without a grid_ prefix, else None."""
    return next((folder / n for n in (f"{sampler}.pt", f"grid_{sampler}.pt") if (folder / n).exists()), None)


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def ffn_predict(beta: torch.Tensor, layer_sizes, X: torch.Tensor, activation: str = "tanh",
                chunk: int = 500) -> torch.Tensor:
    """Outputs [S, N] of the fully connected network for every draw of beta [S, D_net]."""
    module = FFN(layer_sizes, activation).to(torch.float64)
    names = [n for n, _ in module.named_parameters()]
    shapes = [p.shape for _, p in module.named_parameters()]
    sizes = [s.numel() for s in shapes]

    def f(b):
        params = {n: x.view(s) for n, s, x in zip(names, shapes, b.split(sizes))}
        return torch.func.functional_call(module, params, (X,)).squeeze(-1)

    return torch.cat([torch.func.vmap(f)(beta[i:i + chunk]) for i in range(0, beta.shape[0], chunk)])


def regression_row(run: dict, data: dict, keep=None, noise_std=None) -> dict:
    """Metrics of one regression run. noise_std given means the noise is fixed (toys)."""
    z = run["samples"][:N_DRAWS].double()
    L = run["layer_sizes"]
    beta = z if noise_std is not None else z[:, :-1]
    sigma = (torch.full((z.shape[0],), noise_std, dtype=torch.float64) if noise_std is not None
             else z[:, -1].exp())
    X, y = data["X_test"].double(), data["y_test"].double()
    if keep is not None:
        X, y = X[keep], y[keep]
    preds = ffn_predict(beta, L, X)
    mask = freezable(FFN(L))
    return {**regression_metrics(y, preds, sigma, data["y_std"]), "Sparsity": sparsity(beta, mask),
            "Grads": int(run.get("grad_evals") or run.get("gradient_evals") or 0)}


def show(table: pd.DataFrame, title: str, latex: bool):
    print(f"\n=== {title}")
    print(table.to_string())
    if latex:
        print(table.to_latex(escape=False))


# ------------------------------------------------------------------ toys
def toy(args):
    for ds_dir in sorted(p for p in args.results.iterdir() if p.is_dir() and p.name in TOY_X_RANGE):
        ds = ds_dir.name
        data = toy_data(ds)
        r = TOY_X_RANGE[ds]
        keep = None if r is None else data["x_test_raw"].abs() <= r
        rows = {}
        for s in ORDER:
            f = run_file(ds_dir / "split_00", s)
            if f is not None:
                rows[LABELS[s]] = regression_row(load(f), data, keep, noise_std=data["noise_std"])
        table = pd.DataFrame(rows).T[["Sparsity", "RMSE", "NLL", "CRPS", "Cov"]]
        sparse = [LABELS[s] for s in ("sticky_zigzag", "sticky_boomerang", "lbbnn")]
        table["Sparsity"] = [f"{v:.0f}%" if i in sparse else "--" for i, v in table["Sparsity"].items()]
        show(table.round(3), f"toy {ds}" + ("" if r is None else f", |x| <= {r:g}"), args.latex)


# ------------------------------------------------------------------- UCI
def _pm(x: pd.Series, dec: int = 2) -> str:
    return f"{x.mean():.{dec}f} ± {x.sem():.{dec}f}" if len(x) > 1 else f"{x.mean():.{dec}f}"


def uci(args):
    data_cache = {}
    for net_dir in sorted(p for p in args.results.iterdir() if p.is_dir()):
        for ds_dir in sorted(p for p in net_dir.iterdir() if p.is_dir()):
            rows = []
            for split_dir in sorted(ds_dir.glob("split_*")):
                split = int(split_dir.name.split("_")[1])
                for s in ORDER:
                    f = run_file(split_dir, s)
                    if f is None:
                        continue
                    key = (ds_dir.name, split)
                    if key not in data_cache:
                        data_cache[key] = uci_split(ds_dir.name, split)
                    rows.append({"sampler": LABELS[s], "split": split,
                                 **regression_row(load(f), data_cache[key])})
            if not rows:
                continue
            df = pd.DataFrame(rows)
            g = df.groupby("sampler", sort=False)
            table = pd.DataFrame({
                "splits": g.size(),
                "Sparsity": g["Sparsity"].mean().map(lambda v: f"{v:.0f}%"),
                **{k: g[k].apply(_pm) for k in ("RMSE", "NLL", "CRPS")},
                "Cov": g["Cov"].mean().map(lambda v: f"{100 * v:.0f}%"),
                "Grads (M)": g["Grads"].mean().map(lambda v: f"{v / 1e6:.2f}"),
            })
            table.loc[~table.index.isin(["Sticky ZigZag", "Sticky Boomerang", "LBBNN"]), "Sparsity"] = "--"
            show(table, f"UCI {net_dir.name} / {ds_dir.name}", args.latex)


# ---------------------------------------------------------------- images
def images(args):
    from ..common import DEVICE
    from ..data import image_data
    from .image_reference import MODELS, SMOKE_DATA, build_target

    refs = args.references or args.results / "references"
    for model in MODELS:
        model_dir = args.results / model
        map_file = refs / f"{model}_map.pt"
        if not map_file.exists():
            continue
        cfg = MODELS[model]
        ref = load(map_file)
        sizes = SMOKE_DATA if ref.get("smoke") else {}
        data = image_data(cfg["data"], n_val=sizes.get("n_val", 2000), n_train=sizes.get("n_train"),
                          n_test=sizes.get("n_test"), flatten=cfg["flatten"], dtype=ref["x_ref"].dtype,
                          device=DEVICE)
        bm = build_target(model, data, ref["module_state_dict"])
        mask = freezable(bm.module)
        X, y = data["X_test"], data["y_test"].cpu()

        def probs_of(draws: torch.Tensor, b=bm) -> torch.Tensor:
            return sum(torch.softmax(b.predict(d.to(b.X.dtype), X), -1).cpu().double() for d in draws) / len(draws)

        rows = {}
        sgd_file = refs / f"{model}_sgd.pt"
        if sgd_file.exists():
            sgd = load(sgd_file)
            bm_sgd = build_target(model, data, sgd["state_dict"])
            rows["SGD"] = {**classification_metrics(y, probs_of(sgd["beta"][None], bm_sgd)), "Sparsity": 0.0}
        x_ref = ref["x_ref"]
        rows["MAP (pruned)"] = {**classification_metrics(y, probs_of(x_ref[None])),
                                "Sparsity": sparsity(x_ref[None], mask)}
        g = torch.Generator().manual_seed(0)
        for s in ("sticky_zigzag", "sticky_boomerang"):
            f = run_file(model_dir, s)
            if f is None:
                continue
            z = load(f)["samples"]
            idx = torch.randperm(z.shape[0], generator=g)[:args.n_draws]
            rows[LABELS[s]] = {**classification_metrics(y, probs_of(z[idx])), "Sparsity": sparsity(z, mask)}
        table = pd.DataFrame(rows).T[["Acc", "NLL", "ECE", "Brier", "Sparsity"]]
        table["Sparsity"] = table["Sparsity"].map(lambda v: f"{v:.0f}%")
        show(table.round(3), f"images {model}, {len(y)} test images, {args.n_draws} draws", args.latex)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=["toy", "uci", "images"])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--references", type=Path, default=None, help="images only, default <results>/references")
    p.add_argument("--n-draws", type=int, default=300, help="images only, draws averaged per model")
    p.add_argument("--latex", action="store_true")
    args = p.parse_args()
    {"toy": toy, "uci": uci, "images": images}[args.experiment](args)


if __name__ == "__main__":
    main()
