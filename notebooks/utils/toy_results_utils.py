"""Helpers for notebooks/toy_results.ipynb."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal

from sazz.gpu_friendly.scripts.toy_bnn_grid import build_target, DATASET_CONFIGS, BNNConfig
from sazz.utils.metrics import ess_per_coord

DATA_DIR = Path("datasets/toy_1d")

SAMPLER_STYLE = {
    "grid_boomerang":        ("C0", "Boomerang"),
    "grid_sticky_boomerang": ("C1", "Sticky Boomerang"),
    "grid_zigzag":           ("C2", "Zig Zag"),
    "grid_sticky_zigzag":    ("C3", "Sticky Zig Zag"),
    "nuts":                  ("C4", "NUTS"),
    "svi":                   ("C5", "SVI"),
    "tf_boomerang":          ("C6", "TFB"),
}
GRID_SAMPLERS = ["grid_boomerang", "grid_sticky_boomerang", "grid_zigzag", "grid_sticky_zigzag"]

RUN_TREES = {
    "fullbatch": Path("results/paper/toy_bnns"),
    "minibatch": Path("results/paper/toy_bnns_minibatch"),
    "negmap":    Path("results/paper/toy_bnns_negative_map"),
}
RUN_STYLE = {
    "fullbatch": ("full batch",   "-",  1.6),
    "minibatch": ("minibatch",    "--", 1.3),
    "negmap":    ("neg-MAP init", ":",  1.6),
}
RUN_COLOR = {"fullbatch": "black", "minibatch": "C1", "negmap": "C3"}

# Skeleton subfolder names on disk (results/paper/toy_bnns/skeletons/<arm>/...,
# written by toy_bnn_grid*.py's --save-skeleton) -- "negmap" here vs.
# "negative_map" on disk is the one place the notebook's short run_key and
# the scripts' own naming diverge.
SKELETON_ARM = {"fullbatch": "fullbatch", "minibatch": "minibatch", "negmap": "negative_map"}


def runs_present():
    return {k: v for k, v in RUN_TREES.items() if v.is_dir()}


def discover(tree: Path):
    out = set()
    for p in tree.glob("*/split_*/*.pt"):
        if p.stem in GRID_SAMPLERS:
            out.add((p.parent.parent.name, int(p.parent.name.split("_")[1])))
    return sorted(out)


def load_run(run_key, dataset, split_id, sampler, skeletons=False):
    """skeletons=True: ALWAYS looks under the fullbatch tree's own
    skeletons/<arm>/... subfolder (results/paper/toy_bnns/skeletons/<arm>/...)
    regardless of run_key's own RUN_TREES entry -- every arm's --save-skeleton
    output is consolidated there, not under e.g. RUN_TREES["negmap"]'s
    separate tree. skeletons=False (the resampled-draws .pt) is unaffected,
    still read from run_key's own RUN_TREES[run_key]."""
    if skeletons:
        arm = SKELETON_ARM.get(run_key, run_key)
        p = RUN_TREES["fullbatch"] / "skeletons" / arm / dataset / f"split_{split_id:02d}" / f"{sampler}_skeleton.pt"
        return torch.load(p, weights_only=False) if p.exists() else None
    tree = RUN_TREES.get(run_key)
    if tree is None:
        return None
    p = tree / dataset / f"split_{split_id:02d}" / f"{sampler}.pt"
    return torch.load(p, weights_only=False) if p.exists() else None


def load_dataset(dataset):
    return torch.load(DATA_DIR / f"{dataset}.pt", weights_only=False)


def make_bm(dataset, data):
    cfg = BNNConfig(**DATASET_CONFIGS[dataset], noise_std=data["noise_std"])
    bm, _, _ = build_target(data, cfg)
    return bm


def any_payload(dataset, split_id):
    return next((load_run(r, dataset, split_id, s)
                 for s in GRID_SAMPLERS for r in RUN_STYLE
                 if load_run(r, dataset, split_id, s) is not None), None)


@torch.no_grad()
def predictive_summary(samples, bm, payload, data, x_grid):
    x_mean, x_std = float(data["x_mean"]), float(data["x_std"])
    y_mean, y_std = float(data["y_mean"]), float(data["y_std"])

    X_grid = torch.tensor(x_grid[:, None], dtype=torch.float64)
    X_grid_std = (X_grid - x_mean) / x_std

    if bm.learns_noise:
        weight_samples = samples[:, :-1]
        noise_std_scale = float(samples[:, -1].exp().mean())
    else:
        weight_samples = samples
        noise_std_scale = float(payload["noise_std"])

    preds = torch.stack([
        torch.func.functional_call(bm.module, bm.param_dict_fn(beta), (X_grid_std,)).squeeze(-1)
        for beta in weight_samples
    ])

    mean = preds.mean(0).numpy() * y_std + y_mean
    epi = preds.std(0).numpy() * y_std
    total = np.sqrt(epi**2 + (noise_std_scale * y_std)**2)
    return mean, epi, total


@torch.no_grad()
def predict_on_test(samples, bm, data):
    X_test = data["X_test"].to(dtype=torch.float64)
    if bm.learns_noise:
        weight_samples = samples[:, :-1]
        noise_std = float(samples[:, -1].exp().mean())
    else:
        weight_samples = samples
        noise_std = float(data["noise_std"])
    preds = torch.stack([
        torch.func.functional_call(bm.module, bm.param_dict_fn(beta), (X_test,)).squeeze(-1)
        for beta in weight_samples
    ])
    return preds.mean(0), preds.std(0), noise_std


def compute_rmse(y_true, mean_pred, y_std):
    return float(((mean_pred - y_true) ** 2).mean().sqrt()) * y_std


def compute_nll(y_true, mean_pred, total_std, y_std):
    ll = (-0.5 * ((y_true - mean_pred) / total_std) ** 2
          - total_std.log() - 0.5 * math.log(2 * math.pi)).mean()
    return float(-ll + math.log(y_std))


def compute_crps(y_true, mean_pred, total_std, y_std):
    d = Normal(0.0, 1.0)
    sigma = total_std * y_std
    z = (y_true * y_std - mean_pred * y_std) / sigma
    return float((sigma * (z * (2 * d.cdf(z) - 1) + 2 * d.log_prob(z).exp()
                            - 1 / math.sqrt(math.pi))).mean())


def predictive_metrics(payload, bm, data):
    y_test = data["y_test"].to(dtype=torch.float64)
    y_std = float(data["y_std"])
    mean_pred, epist_std, noise_std = predict_on_test(payload["samples"], bm, data)
    total_std = (epist_std ** 2 + noise_std ** 2).sqrt()
    return {
        "RMSE": compute_rmse(y_test, mean_pred, y_std),
        "NLL":  compute_nll(y_test, mean_pred, total_std, y_std),
        "CRPS": compute_crps(y_test, mean_pred, total_std, y_std),
    }


def flip_mask(bm):
    """Bool [D] mask of sign-flipped coords: every param except the last
    Linear's bias. Mirrors build_negate_mask in toy_bnn_grid_negative_map.py."""
    names = [n for n, _ in bm.module.named_parameters()]
    last_bias = f"layers.{max(int(n.split('.')[1]) for n in names if n.startswith('layers.'))}.bias"
    mask = torch.zeros(bm.D, dtype=torch.bool)
    idx = 0
    for name, p in bm.module.named_parameters():
        n = p.numel()
        if name != last_bias:
            mask[idx:idx + n] = True
        idx += n
    return mask


def weight_mask(bm):
    """Bool [D]: True for weight-matrix coords only (excludes ALL biases,
    including the first layer's, which flip_mask still includes)."""
    m = torch.zeros(bm.D, dtype=torch.bool)
    idx = 0
    for _, p in bm.module.named_parameters():
        n = p.numel()
        m[idx:idx + n] = p.dim() != 1
        idx += n
    return m


def reconstruct_path(skel, coords, x_ref=None, n_sub=20):
    """Dense continuous trajectory on `coords`, from a skeleton payload
    (positions/velocities/times). Interpolation formulas match
    sazz/gpu_friendly/utils/resample.py exactly:
      ZigZag:    x(t) = x_k + (t - t_k) * v_k
      Boomerang: x(t) = x_ref + (x_k - x_ref)*cos(t - t_k) + v_k*sin(t - t_k)
    (x_ref given -> Boomerang; x_ref=None -> ZigZag). Returns array
    [n_events * n_sub, len(coords)]."""
    pos = skel["positions"][:, coords].double()
    vel = skel["velocities"][:, coords].double()
    tim = skel["times"].double()
    frac = torch.linspace(0.0, 1.0, n_sub)

    out = []
    for k in range(pos.shape[0] - 1):
        dt = frac * (tim[k + 1] - tim[k])
        if x_ref is not None:
            xr = x_ref[coords].double()
            out.append(xr + (pos[k] - xr) * torch.cos(dt)[:, None] + vel[k] * torch.sin(dt)[:, None])
        else:
            out.append(pos[k] + vel[k] * dt[:, None])
    return torch.cat(out, dim=0).numpy()


def mirror_projection(samples, mask, x_ref_pos):
    """+1 deep in the positive mode, -1 deep in the mirror mode."""
    s = torch.sign(x_ref_pos[mask])
    denom = x_ref_pos[mask].abs().sum().clamp_min(1e-12)
    return (samples[:, mask].double() * s).sum(dim=1) / denom


def reference_x_ref(dataset, split_id, bm, data):
    x_ref_pos = None
    for run_key in ("fullbatch", "minibatch"):
        for sampler in GRID_SAMPLERS:
            p = load_run(run_key, dataset, split_id, sampler)
            if p is not None and p.get("x_ref") is not None:
                x_ref_pos = p["x_ref"].double()
                break
        if x_ref_pos is not None:
            break
    if x_ref_pos is None:
        _, x_ref_pos, _ = build_target(
            data, BNNConfig(**DATASET_CONFIGS[dataset], noise_std=data["noise_std"]))
        x_ref_pos = x_ref_pos.double()
    return x_ref_pos
