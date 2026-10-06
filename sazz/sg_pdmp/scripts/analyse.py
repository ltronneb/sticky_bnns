"""Tables of the SG-PDMP runs with the metrics of sazz.paper_ready (metrics.py),
one row per sampler, step size and batch size. --paper adds the paper's samplers
from their results folder (same layout) to the same tables. For UCI --paper is
either the root with one folder per network or a single network's folder.

    python -m sazz.sg_pdmp.scripts.analyse toy --results results/sg_pdmp/toy_bnns --paper results/toy_bnns
    python -m sazz.sg_pdmp.scripts.analyse uci --results results/sg_pdmp/uci --paper results/uci
    python -m sazz.sg_pdmp.scripts.analyse images --results results/sg_pdmp/images \\
        --paper results/images --references results/images/references

The extra columns are the run time in minutes and, for UCI, the number of
diverged splits. Grads are full-data gradient equivalents for the SG-PDMPs and
the paper's count for its samplers.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from sazz.paper_ready.data import toy_data, uci_split
from sazz.paper_ready.metrics import classification_metrics, sparsity
from sazz.paper_ready.models.bnn import freezable
from sazz.paper_ready.scripts.analyse import (LABELS as PAPER_LABELS, ORDER as PAPER_ORDER, TOY_X_RANGE,
                                              _pm, load, regression_row, run_file, show)

from ..common import label, parse_run_name

SPARSE = ("SG-SZZ", "Sticky ZigZag", "Sticky Boomerang", "LBBNN")


def runs(sg_dir: Path, paper_dir: Optional[Path]):
    """(label, path) of the SG runs in sg_dir and the paper's runs in paper_dir."""
    out = []
    if paper_dir is not None and paper_dir.is_dir():
        out += [(PAPER_LABELS[s], f) for s in PAPER_ORDER if (f := run_file(paper_dir, s)) is not None]
    if sg_dir.is_dir():
        stems = [f.stem for f in sg_dir.glob("sg_*.pt") if parse_run_name(f.stem)]
        for stem in sorted(stems, key=lambda s: (parse_run_name(s)[0], -parse_run_name(s)[1], s)):
            out.append((label(stem), sg_dir / f"{stem}.pt"))
    return out


def extra(run: dict) -> dict:
    return {"Min": run.get("elapsed_sec", float("nan")) / 60, "Diverged": bool(run.get("diverged", False))}


def is_sparse(lbl: str) -> bool:
    return lbl.startswith(SPARSE)


# ------------------------------------------------------------------ toys
def toy(args):
    for ds in TOY_X_RANGE:
        found = runs(args.results / ds / "split_00", args.paper and args.paper / ds / "split_00")
        if not found:
            continue
        data = toy_data(ds)
        r = TOY_X_RANGE[ds]
        keep = None if r is None else data["x_test_raw"].abs() <= r
        rows = {}
        for lbl, f in found:
            run = load(f)
            rows[lbl] = {**regression_row(run, data, keep, noise_std=data["noise_std"]), **extra(run)}
        table = pd.DataFrame(rows).T[["Sparsity", "RMSE", "NLL", "CRPS", "Cov", "Grads", "Min"]]
        table["Sparsity"] = [f"{v:.0f}%" if is_sparse(i) else "--" for i, v in table["Sparsity"].items()]
        table["Grads"] = table["Grads"].map(lambda v: f"{v:,.0f}")
        show(table.round(3), f"toy {ds}" + ("" if r is None else f", |x| <= {r:g}"), args.latex)


# ------------------------------------------------------------------- UCI
def uci(args):
    data_cache = {}
    for net_dir in sorted(p for p in args.results.iterdir() if p.is_dir()):
        for ds_dir in sorted(p for p in net_dir.iterdir() if p.is_dir()):
            rows = []
            for split_dir in sorted(ds_dir.glob("split_*")):
                split = int(split_dir.name.split("_")[1])
                paper = None
                if args.paper:   # either the paper's results root or one network's folder in it
                    root = args.paper / net_dir.name if (args.paper / net_dir.name).is_dir() else args.paper
                    paper = root / ds_dir.name / split_dir.name
                for lbl, f in runs(split_dir, paper):
                    if (ds_dir.name, split) not in data_cache:
                        data_cache[ds_dir.name, split] = uci_split(ds_dir.name, split)
                    run = load(f)
                    rows.append({"sampler": lbl, "split": split,
                                 **regression_row(run, data_cache[ds_dir.name, split]), **extra(run)})
            if not rows:
                continue
            g = pd.DataFrame(rows).groupby("sampler", sort=False)
            table = pd.DataFrame({
                "splits": g.size(),
                "Diverged": g["Diverged"].sum(),
                "Sparsity": g["Sparsity"].mean().map(lambda v: f"{v:.0f}%"),
                **{k: g[k].apply(_pm) for k in ("RMSE", "NLL", "CRPS")},
                "Cov": g["Cov"].mean().map(lambda v: f"{100 * v:.0f}%"),
                "Grads (M)": g["Grads"].mean().map(lambda v: f"{v / 1e6:.2f}"),
                "Min": g["Min"].mean().map(lambda v: f"{v:.0f}"),
            })
            table.loc[[not is_sparse(i) for i in table.index], "Sparsity"] = "--"
            show(table, f"UCI {net_dir.name} / {ds_dir.name}", args.latex)


# ---------------------------------------------------------------- images
def images(args):
    from sazz.paper_ready.common import DEVICE
    from sazz.paper_ready.data import image_data
    from sazz.paper_ready.scripts.image_reference import MODELS, SMOKE_DATA, build_target

    refs = args.references or (args.paper or args.results) / "references"
    for model in MODELS:
        map_file = refs / f"{model}_map.pt"
        found = runs(args.results / model, args.paper and args.paper / model)
        if not map_file.exists() or not found:
            continue
        cfg, ref = MODELS[model], load(map_file)
        sizes = SMOKE_DATA if ref.get("smoke") else {}
        data = image_data(cfg["data"], n_val=sizes.get("n_val", 2000), n_train=sizes.get("n_train"),
                          n_test=sizes.get("n_test"), flatten=cfg["flatten"], dtype=ref["x_ref"].dtype,
                          device=DEVICE)
        bm = build_target(model, data, ref["module_state_dict"])
        mask = freezable(bm.module)
        X, y = data["X_test"], data["y_test"].cpu()

        def probs_of(draws: torch.Tensor) -> torch.Tensor:
            return sum(torch.softmax(bm.predict(d.to(bm.X.dtype), X), -1).cpu().double() for d in draws) / len(draws)

        rows = {"MAP (pruned)": {**classification_metrics(y, probs_of(ref["x_ref"][None])),
                                 "Sparsity": sparsity(ref["x_ref"][None], mask), "Min": float("nan")}}
        g = torch.Generator().manual_seed(0)
        for lbl, f in found:
            run = load(f)
            z = run["samples"]
            idx = torch.randperm(z.shape[0], generator=g)[:args.n_draws]
            metrics = (classification_metrics(y, probs_of(z[idx])) if torch.isfinite(z).all()
                       else dict.fromkeys(("Acc", "NLL", "ECE", "Brier"), float("nan")))
            rows[lbl] = {**metrics, "Sparsity": sparsity(z, mask), **extra(run)}
        table = pd.DataFrame(rows).T[["Acc", "NLL", "ECE", "Brier", "Sparsity", "Min"]]
        table["Sparsity"] = table["Sparsity"].map(lambda v: f"{v:.0f}%")
        show(table.round(3), f"images {model}, {len(y)} test images, {args.n_draws} draws", args.latex)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=["toy", "uci", "images"])
    p.add_argument("--results", type=Path, required=True, help="the SG-PDMP results")
    p.add_argument("--paper", type=Path, default=None, help="the paper's results, same layout")
    p.add_argument("--references", type=Path, default=None,
                   help="images only, default <paper or results>/references")
    p.add_argument("--n-draws", type=int, default=300, help="images only, draws averaged per run")
    p.add_argument("--latex", action="store_true")
    args = p.parse_args()
    {"toy": toy, "uci": uci, "images": images}[args.experiment](args)


if __name__ == "__main__":
    main()
