"""Load experiment artifacts. Use from a notebook for analysis."""
import pickle
from pathlib import Path
import pandas as pd


def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all(results_dir="results", pattern="*.pkl"):
    """Return a list of artifact dicts for everything in results_dir."""
    return [load_artifact(p) for p in sorted(Path(results_dir).glob(pattern))]


def summary_frame(artifacts):
    """Flatten artifacts into a dataframe (one row per run, no samples)."""
    rows = []
    for a in artifacts:
        rows.append({
            "target": a["target"]["name"],
            "sampler": a["config"]["sampler"]["name"],
            "seed": a["config"]["seed"],
            "N": a["config"]["N"],
            "wall_s": a["wall_seconds"],
            "n_grad_evals": a["diagnostics"].get("n_grad_evals"),
            "tag": a.get("tag", ""),
        })
    return pd.DataFrame(rows)
