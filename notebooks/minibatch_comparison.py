# %%
# ==========================================================================
# Minibatch vs full batch, small UCI network, v1 results (grid bound).
#   full batch  results/paper/shallow/<dataset>/split_XX/grid_<sampler>.pt
#   minibatch   results/paper/shallow/minibatch_32/<dataset>/split_XX/grid_<sampler>.pt
# Both arms start from the same MAP. Metrics are the ones in the UCI tables
# (uci_results_utils.run_metrics). Delta = full - minibatch, so for RMSE, NLL
# and CRPS (lower is better) a negative Delta means minibatching is worse.
# ==========================================================================
import os
import sys
from pathlib import Path

if Path.cwd().name == "notebooks":
    os.chdir("..")
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import torch

from notebooks.utils.uci_results_utils import run_metrics, DTYPE
from sazz.gpu_friendly.scripts.uci_bnn_grid import load_raw_datasets, make_split, BASE_SEED

FULL_DIR = Path("results/paper/shallow")
MB_DIR = FULL_DIR / "minibatch_32"
DATASETS = ["boston", "energy", "concrete", "yacht"]          # table column order
SAMPLERS = {"grid_boomerang": "Boomerang", "grid_zigzag": "ZigZag",
            "grid_sticky_boomerang": "Sticky Boomerang", "grid_sticky_zigzag": "Sticky ZigZag"}
METRICS = ["RMSE", "NLL", "CRPS"]
RELATIVE = False   # True gives (full - mb) / full, unstable for NLL near zero

# %%
rows = []
raw = load_raw_datasets(tuple(DATASETS))
for dataset in DATASETS:
    for split_dir in sorted((MB_DIR / dataset).glob("split_*")):
        split_id = int(split_dir.name.split("_")[-1])
        data = make_split(*raw[dataset], seed=BASE_SEED + split_id, dtype=DTYPE, device="cpu")
        for stem in SAMPLERS:
            paths = {"mb": split_dir / f"{stem}.pt",
                     "full": FULL_DIR / dataset / split_dir.name / f"{stem}.pt"}
            if not all(p.exists() for p in paths.values()):
                print(f"  missing {dataset}/{split_dir.name}/{stem}")
                continue
            for arm, p in paths.items():
                run = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
                assert np.isclose(run["y_std"], data["y_std"]), f"{p}, split does not match the run"
                m = run_metrics(run, data)
                rows.append({"dataset": dataset, "split": split_id, "sampler": stem, "arm": arm,
                             "n_events": run.get("n_events"), **m})
df = pd.DataFrame(rows)
print(f"{len(df)} runs")

# %%
# Budgets differ between the arms, print them so the table is read correctly
print(df.groupby(["arm", "sampler"])[["grad_evals", "n_events", "n_samples"]].mean().round(0))

# %%
wide = df.pivot_table(index=["dataset", "split", "sampler"], columns="arm", values=METRICS)
delta = pd.DataFrame({m: wide[(m, "full")] - wide[(m, "mb")] for m in METRICS})
if RELATIVE:
    delta = delta / pd.DataFrame({m: wide[(m, "full")] for m in METRICS})
g = delta.groupby(["sampler", "dataset"])
mean, sem = g.mean(), g.sem()

table = pd.DataFrame(
    {(ds, m): [f"{mean.loc[(s, ds), m]:+.2f} ± {sem.loc[(s, ds), m]:.2f}" for s in SAMPLERS]
     for ds in DATASETS for m in METRICS},
    index=list(SAMPLERS.values()))
print("Delta = full - minibatch, mean ± SEM over splits" + (" (relative)" if RELATIVE else ""))
print(table.to_string())

# Level of each metric in the full-batch arm, to judge the size of Delta
print(df[df.arm == "full"].groupby(["dataset"])[METRICS].mean().loc[DATASETS].round(3))

# %%
# LaTeX body in the layout of tab:mb, means only
def _f(v):
    s = f"{v:.2f}"
    return s.replace("0.", ".", 1) if abs(v) < 1 else s

print(" & ".join(["$\\Delta$"] + METRICS * len(DATASETS)) + r" \\")
print(r"\midrule")
for s, label in SAMPLERS.items():
    cells = [_f(mean.loc[(s, ds), m]) for ds in DATASETS for m in METRICS]
    print(f"{label:<17}& " + " & ".join(cells) + r" \\")
