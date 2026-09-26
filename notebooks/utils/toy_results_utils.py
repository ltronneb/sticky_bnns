"""Helpers for notebooks/toy_results.ipynb."""

from pathlib import Path

import numpy as np
import torch

from sazz.gpu_friendly.scripts.toy_bnn_grid import build_target, DATASET_CONFIGS, BNNConfig

DATA_DIR = Path("datasets/toy_1d")
RESULTS_DIR = Path("results/paper_v2/toy_bnns")

# Skeletons from toy_bnn_grid*.py --save-skeleton, one subfolder per arm.
# Only the results/paper tree has them so far.
SKELETON_DIR = Path("results/paper/toy_bnns/skeletons")
ARMS = {"MAP": "fullbatch", "Symmetric MAP": "negative_map"}

SAMPLER_STYLE = {
    "grid_boomerang":        ("C0", "Boomerang"),
    "grid_sticky_boomerang": ("C1", "Sticky Boomerang"),
    "grid_zigzag":           ("C2", "Zig Zag"),
    "grid_sticky_zigzag":    ("C3", "Sticky Zig Zag"),
    "nuts":                  ("C4", "NUTS"),
    "svi":                   ("C5", "SVI"),
    "tf_boomerang":          ("C6", "TFB"),
    "lbbnn":                 ("C6", "LBBNN"),
}


def load_dataset(dataset):
    return torch.load(DATA_DIR / f"{dataset}.pt", weights_only=False)


def load_run(dataset, split_id, sampler):
    """Resampled draws from RESULTS_DIR, or None if the run is missing."""
    p = RESULTS_DIR / dataset / f"split_{split_id:02d}" / f"{sampler}.pt"
    return torch.load(p, weights_only=False) if p.exists() else None


def load_skeleton(arm, dataset, split_id, sampler):
    p = SKELETON_DIR / arm / dataset / f"split_{split_id:02d}" / f"{sampler}_skeleton.pt"
    return torch.load(p, weights_only=False) if p.exists() else None


def make_bm(dataset, data):
    cfg = BNNConfig(**DATASET_CONFIGS[dataset], noise_std=data["noise_std"])
    bm, _, _ = build_target(data, cfg)
    return bm


# ---------------------------------------------------------------------------
# Predictive fit
# ---------------------------------------------------------------------------

@torch.no_grad()
def predictive_summary(samples, bm, payload, data, x_grid):
    """Posterior predictive mean, epistemic sd and total sd on x_grid, all
    on the original y scale."""
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


# ---------------------------------------------------------------------------
# Mode crossing, MAP arm vs sign-flipped MAP arm
# ---------------------------------------------------------------------------

def flip_mask(bm):
    """Bool [D] mask of sign-flipped coords, every param except the last
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
    """Bool [D], True for weight-matrix coords only (all biases excluded)."""
    m = torch.zeros(bm.D, dtype=torch.bool)
    idx = 0
    for _, p in bm.module.named_parameters():
        n = p.numel()
        m[idx:idx + n] = p.dim() != 1
        idx += n
    return m


def reconstruct_path(skel, coords, n_sub=20, zero_tol=1e-12):
    """Dense continuous trajectory [n_events * n_sub, len(coords)] from a
    skeleton, with the same formulas as sazz/gpu_friendly/utils/resample.py.
        ZigZag     x(t) = x_k + (t - t_k) v_k
        Boomerang  x(t) = x_ref + (x_k - x_ref) cos(t - t_k) + v_k sin(t - t_k)
    Boomerang orbits the skeleton's own stored x_ref. For sticky samplers a
    coord that is frozen at the left end of an interval (pos and vel both 0)
    stays exactly 0 over that interval, as in resample_*_path_sticky_torch."""
    pos = skel["positions"][:, coords].double()
    vel = skel["velocities"][:, coords].double()
    tim = skel["times"].double()

    dt = (torch.linspace(0.0, 1.0, n_sub, dtype=torch.float64)[None, :]
          * (tim[1:] - tim[:-1])[:, None])[..., None]                  # [K, n_sub, 1]
    x0, v0 = pos[:-1, None, :], vel[:-1, None, :]                      # [K, 1, C]
    if "boomerang" in skel["sampler"]:
        xr = skel["x_ref"][coords].double()
        seg = xr + (x0 - xr) * torch.cos(dt) + v0 * torch.sin(dt)
    else:
        seg = x0 + v0 * dt
    if "sticky" in skel["sampler"]:
        frozen = (x0.abs() < zero_tol) & (v0.abs() < zero_tol)
        seg = torch.where(frozen, torch.zeros_like(seg), seg)
    return seg.reshape(-1, len(coords)).numpy()


def mode_crossing_paths(dataset, split_id, sampler, bm, n_dims=3, n_sub=20):
    """Paths of both arms on the n_dims flipped weight coords where the two
    mode centers are furthest apart. x_ref_pos is the MAP arm's reference and
    x_ref_neg its sign flip (which is exactly the Symmetric MAP arm's x_ref)."""
    skels = {label: load_skeleton(arm, dataset, split_id, sampler) for label, arm in ARMS.items()}
    if skels["MAP"] is None:
        raise FileNotFoundError(f"no MAP skeleton for {dataset}/{sampler} in {SKELETON_DIR}")

    mask = flip_mask(bm)
    x_ref_pos = skels["MAP"]["x_ref"].double()
    x_ref_neg = torch.where(mask, -x_ref_pos, x_ref_pos)

    flip_idx = (mask & weight_mask(bm)).nonzero().flatten()
    separation = (x_ref_pos[flip_idx] - x_ref_neg[flip_idx]).abs()
    coords = flip_idx[separation.argsort(descending=True)[:n_dims]].tolist()

    paths = {label: reconstruct_path(s, coords, n_sub=n_sub)
             for label, s in skels.items() if s is not None}
    return {
        "paths": paths,
        "centers": [x_ref_pos[coords].numpy(), x_ref_neg[coords].numpy()],
    }
