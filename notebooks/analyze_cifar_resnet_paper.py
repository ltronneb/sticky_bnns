"""
  A. PREDICTIVE QUALITY & CALIBRATION on the full CIFAR-10 test set.
     Standard reporting (accuracy / NLL / ECE / Brier) for SGD, the pruned
     MAP point, and the sticky-PDMP posteriors.

  B. CORRUPTION ROBUSTNESS -- CIFAR-10-C (Hendrycks & Dietterich 2019).
     Pooled accuracy vs severity (0 = clean, 1-5) for a chosen set of 4
     corruption types, one 2x2 grid of curves, one line per model.
"""

from __future__ import annotations

# %%
# ==========================================================================
# 0. Config, imports, device
# ==========================================================================
import os
import sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # any op MPS lacks -> CPU
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Find the repo root (dir containing "sazz/") regardless of where this is
# launched from, chdir there so relative result paths resolve, and make
# `import sazz...` work even when run as a plain script.
_here = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
for _root in [_here, *_here.parents]:
    if (_root / "sazz").is_dir():
        os.chdir(_root)
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break
print("cwd:", Path.cwd())

FORCE_CPU = False
if FORCE_CPU:
    DEVICE = torch.device("cpu")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
DTYPE = torch.float32
print("device:", DEVICE)

RUN_DIR = Path("results/paper/cifar")
RUN_SPECS = [
    ("zigzag", "grid_sticky_zigzag.pt"),
    ("boomerang", "grid_sticky_boomerang.pt"),   # slotted in automatically once present
]
RUN_DISPLAY = {"zigzag": "Sticky Zig-Zag", "boomerang": "Sticky Boomerang"}

# The v2 refit checkpoint -- verified bit-identical x_ref and cold_start_mask
# to grid_sticky_zigzag.pt (52.88% sparse, pretrained 180 epochs,
# test_acc_pruned_refit=0.90). The non-v2 file is a DIFFERENT, 33.6%-sparse
# MAP from an 80-epoch base and does NOT match this run.
MAP_REF_PATH = Path("results/maps/resnet/resnet20_reference_N50000_steps2000_v2_pruned_refit.pt")

# Pure-SGD pretrained ResNet-20 -- the frequentist baseline (Izmailov et al.
# 2021 Fig 15 role); resnet20_reference.py's MAP refinement starts from it.
SGD_REF_PATH = Path("results/maps/resnet/resnet20_pretrain_N50000_epochs80.pt")

# Prior hyper-params the run used (fast_cheap_cifar_resnet.py).
PRIOR_STD_W = 2.0
PRIOR_STD_B = 2.0
PRIOR_STD_BN_W = 1.0
FAN_IN_SCALING = True
BASE_SEED = 42          # fast_cifar_resnet.BASE_SEED -- same test split as the run

# --- clean metric table: full test set ---
N_PRED_DRAWS = 200      # posterior draws for the clean headline table
N_TEST = 10_000         # full CIFAR-10 test set

# --- corruption grid: CIFAR-10-C ---
CIFAR10C_DIR = Path("datasets/CIFAR-10-C")
# download:  mkdir -p datasets/CIFAR-10-C && cd datasets/CIFAR-10-C && \
#   curl -L -o CIFAR-10-C.tar "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1" && \
#   tar xf CIFAR-10-C.tar --strip-components=1
CORRUPTIONS_TO_PLOT = ["gaussian_noise", "motion_blur", "brightness", "fog"]  # pick any 4
SEVERITIES = [1, 2, 3, 4, 5]
N_TEST_CORRUPT = 2_000   # subset size per severity (full 10k x 5 sev x 4 corruptions is heavy)
N_DRAWS_CORRUPT = 50     # posterior draws per corruption point
LOAD_CORRUPT_FROM_SAVED = False   # True -> read table_corruption_accuracy.csv instead of recomputing

ZERO_TOL = 1e-8

SAVE_DIR = Path("results/plots/CIFAR/")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)   # uncomment when you start saving

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 15,
    "figure.dpi": 120,
})


# %%
# ==========================================================================
# 1. Load checkpoints + the pruned MAP reference
# ==========================================================================
runs = {}
for label, fname in RUN_SPECS:
    p = RUN_DIR / fname
    if not p.exists():
        print(f"[{label}] missing {p} -- skipped")
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    runs[label] = ck
    n, D = ck["samples"].shape
    print(f"[{label}] {n} draws x D={D}  "
          f"test_acc(ckpt)={ck['test_accuracy']:.4f}  "
          f"sparsity(ckpt)={ck['sparsity_frac']:.4f}  "
          f"prune_frac={ck['prune_frac']:.4f}  "
          f"wall={ck['elapsed_sec']/3600:.1f}h")
assert runs, f"no run files under {RUN_DIR}"

map_ck = torch.load(MAP_REF_PATH, map_location="cpu", weights_only=False)
MSD = map_ck["module_state_dict"]          # carries frozen BN running stats
X_REF = map_ck["x_ref"].to(DTYPE)          # pruned + refit MAP point
D = int(X_REF.shape[0])
print(f"\nMAP ref: {MAP_REF_PATH.name}")
print(f"  D={D}  train_acc={map_ck['train_acc']:.4f}  "
      f"test_acc(10k, ckpt)={map_ck.get('test_acc_pruned_refit', map_ck['test_acc']):.4f}  "
      f"sparsity={float((X_REF == 0).float().mean()):.4f}")

# Guard: the MAP file MUST be the exact cold-start point of every run.
for label, ck in runs.items():
    dmax = (ck["x_ref"].to(DTYPE) - X_REF).abs().max().item()
    cs_ok = torch.equal(ck["cold_start_mask"].bool(),
                        (X_REF == 0) if map_ck.get("cold_start_mask") is None
                        else map_ck["cold_start_mask"].bool())
    assert dmax < 1e-5 and cs_ok, (
        f"[{label}] MAP MISMATCH: max|x_ref - MAP.x_ref| = {dmax:.3e}, "
        f"cold_start_mask match = {cs_ok}. MAP_REF_PATH ({MAP_REF_PATH.name}) is "
        f"not the checkpoint this run cold-started from."
    )
    print(f"  [{label}] matches run x_ref (max|delta|={dmax:.1e}) and cold_start_mask OK")

# SGD baseline -- carries its own BatchNorm running stats.
sgd_ck = torch.load(SGD_REF_PATH, map_location="cpu", weights_only=False)
from sazz.gpu_friendly.models.neural_networks import ResNet20
_sgd_module = ResNet20(activation="relu")
_sgd_module.load_state_dict(sgd_ck["state_dict"])
X_SGD = torch.cat([p.detach().flatten() for p in _sgd_module.parameters()]).to(DTYPE)
assert int(X_SGD.shape[0]) == D, f"SGD flat D={X_SGD.shape[0]} != D {D}"
SGD_MSD = sgd_ck["state_dict"]
print(f"\nSGD ref: {SGD_REF_PATH.name}  test_acc(ckpt)={sgd_ck['test_acc']:.4f}  "
      f"epochs={sgd_ck['n_epochs']}  (dense, unrelated to MAP_REF_PATH)")
del _sgd_module


# %%
# ==========================================================================
# 2. Data -- full seeded CIFAR-10 test set
# ==========================================================================
from sazz.gpu_friendly.scripts.fast_cifar_resnet import load_cifar10_subset

_data = load_cifar10_subset(50_000, N_TEST, BASE_SEED, Path("datasets"),
                            dtype=DTYPE, device="cpu")
X_test = _data["X_test"]          # [N,3,32,32] normalised, CPU
y_test = _data["y_test"]
CIFAR10_STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
print("X_test:", tuple(X_test.shape), " class balance:",
      torch.bincount(y_test, minlength=10).tolist())

# CIFAR-10-C's .npy files hold the full 10k test set in torchvision's
# original order. load_cifar10_subset draws train_idx then test_idx from the
# SAME rng, so replaying just the test draw is wrong -- replay both to
# recover the underlying 0..9999 indices of X_test's images.
_rng = np.random.default_rng(BASE_SEED)
_ = _rng.choice(50_000, size=50_000, replace=False)          # consume the train draw
TEST_IDX = _rng.choice(10_000, size=N_TEST, replace=False)   # X_test's images, in order


# %%
# ==========================================================================
# 3. Predictive engine -- push a beta vector through ResNet-20
# ==========================================================================
from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.priors import build_fan_in_prior_precision_resnet as _bprec

_module = ResNet20(activation="relu").to(dtype=DTYPE, device=DEVICE)
_module.load_state_dict(MSD, strict=False)
_module.eval()
_prec_dev = _bprec(_module, PRIOR_STD_W, PRIOR_STD_B, PRIOR_STD_BN_W,
                   FAN_IN_SCALING, dtype=DTYPE, device=DEVICE)
_bm = BayesianModule.build(_module, likelihood="categorical",
                           X=X_test[:2].to(DEVICE), y=y_test[:2].to(DEVICE),
                           prior_precision=_prec_dev, dtype=DTYPE, device=DEVICE)
_param_dict_fn = _bm.param_dict_fn

# SGD and MAP each carry their own BatchNorm running stats -- build one
# eval-mode module per state_dict and reuse it (mismatched BN buffers
# collapse accuracy to chance).
_MODULE_CACHE: dict[int, torch.nn.Module] = {}


def _module_for(state_dict):
    if state_dict is None:
        return _bm.module
    key = id(state_dict)
    if key not in _MODULE_CACHE:
        m = ResNet20(activation="relu").to(dtype=DTYPE, device=DEVICE)
        m.load_state_dict(state_dict, strict=False)
        m.eval()
        _MODULE_CACHE[key] = m
    return _MODULE_CACHE[key]


@torch.no_grad()
def predict_probs(beta: torch.Tensor, X: torch.Tensor, batch: int = 256,
                  module: torch.nn.Module | None = None) -> torch.Tensor:
    """[N, 10] softmax probs for one parameter vector beta on inputs X."""
    module = module if module is not None else _bm.module
    beta = beta.to(dtype=DTYPE, device=DEVICE)
    out = []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch].to(dtype=DTYPE, device=DEVICE)
        logits = torch.func.functional_call(module, _param_dict_fn(beta), (xb,))
        out.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def posterior_mean_probs(samples: torch.Tensor, X: torch.Tensor,
                         n_draws: int, seed: int = 0,
                         module: torch.nn.Module | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (mean_probs [N,10], stacked_draw_probs [n_draws,N,10])."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(samples.shape[0], generator=g)[:min(n_draws, samples.shape[0])]
    dp = torch.stack([predict_probs(samples[i], X, module=module) for i in idx])
    return dp.mean(0), dp


def normalise_uint8(arr_hwc: np.ndarray) -> torch.Tensor:
    """[...,32,32,3] uint8 -> [...,3,32,32] normalised float tensor."""
    t = torch.from_numpy(np.ascontiguousarray(arr_hwc).copy()).float().div(255.0)
    t = t.permute(*range(t.ndim - 3), t.ndim - 1, t.ndim - 3, t.ndim - 2)  # HWC -> CHW
    return (t - CIFAR10_MEAN) / CIFAR10_STD


# %%
# ==========================================================================
# 4. Metrics -- accuracy / NLL / ECE / Brier
# ==========================================================================
def calibration_metrics(y_true: torch.Tensor, probs: torch.Tensor, n_bins: int = 15) -> dict:
    y_true = y_true.long()
    conf, pred = probs.max(-1)
    correct = (pred == y_true).float()
    acc = correct.mean().item()
    p_true = probs.gather(-1, y_true.unsqueeze(-1)).squeeze(-1)
    nll = -p_true.clamp_min(1e-12).log().mean().item()
    onehot = F.one_hot(y_true, probs.shape[-1]).float()
    brier = ((probs - onehot) ** 2).sum(-1).mean().item()

    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(correct[m].mean().item() - conf[m].mean().item())
    return {"acc": acc, "nll": nll, "ece": ece, "brier": brier}


def accuracy_only(y_true: torch.Tensor, probs: torch.Tensor) -> float:
    return (probs.argmax(-1) == y_true.long()).float().mean().item()


# %%
# ==========================================================================
# ANALYSIS A -- headline predictive-metrics table, full CIFAR-10 test set
# ==========================================================================
summary_rows = []

_p_sgd = predict_probs(X_SGD, X_test, module=_module_for(SGD_MSD))
_m_sgd = calibration_metrics(y_test, _p_sgd)
summary_rows.append({"model": "SGD", **{k: _m_sgd[k] for k in ("acc", "ece", "nll", "brier")}})

_p_map = predict_probs(X_REF, X_test, module=_module_for(MSD))
_m_map = calibration_metrics(y_test, _p_map)
summary_rows.append({"model": "MAP (pruned x_ref)",
                     **{k: _m_map[k] for k in ("acc", "ece", "nll", "brier")}})

for label, ck in runs.items():
    mean_p, _ = posterior_mean_probs(ck["samples"], X_test, N_PRED_DRAWS)
    _m = calibration_metrics(y_test, mean_p)
    summary_rows.append({"model": RUN_DISPLAY[label],
                         **{k: _m[k] for k in ("acc", "ece", "nll", "brier")}})

summary_df = pd.DataFrame(summary_rows).set_index("model")
print(f"\n=== Predictive metrics on full CIFAR-10 test set (N={N_TEST}) ===")
print(summary_df.to_string(float_format=lambda v: f"{v:.4f}"))
# summary_df.to_csv(SAVE_DIR / "table_headline_metrics.csv")


# %%
# ==========================================================================
# ANALYSIS B -- CIFAR-10-C corruption robustness, 2x2 grid
# ==========================================================================
# For each corruption in CORRUPTIONS_TO_PLOT, sweep severities 0 (clean) and
# 1-5, on a fixed N_TEST_CORRUPT-image subset of X_test, and plot pooled
# accuracy vs severity, one line per model (SGD / MAP / samplers).
#
# LOAD_CORRUPT_FROM_SAVED=True reads the CSV written by a previous run
# instead of recomputing (which requires CIFAR-10-C and is slow); the figure
# below only needs corruption_curves, rebuilt here from the loaded table.
_corrupt_csv_path = SAVE_DIR / "table_corruption_accuracy.csv"
if LOAD_CORRUPT_FROM_SAVED:
    assert _corrupt_csv_path.exists(), f"no saved table at {_corrupt_csv_path}"
    corrupt_table = pd.read_csv(_corrupt_csv_path, index_col=0)
    corruption_curves = {}
    for col in corrupt_table.columns:
        corruption, name = col.split("/", 1)
        corruption_curves.setdefault(corruption, {})[name] = corrupt_table[col].tolist()
    print(f"loaded corruption table from {_corrupt_csv_path}")
else:
    _rng_c = np.random.default_rng(0)
    _corrupt_pos = _rng_c.choice(N_TEST, size=min(N_TEST_CORRUPT, N_TEST), replace=False)
    Xc_clean = X_test[_corrupt_pos]
    yc = y_test[_corrupt_pos]
    _corrupt_test_idx = TEST_IDX[_corrupt_pos]   # -> rows into the CIFAR-10-C .npy files

    model_specs = [("SGD", X_SGD.unsqueeze(0), _module_for(SGD_MSD)),
                  ("MAP (pruned x_ref)", X_REF.unsqueeze(0), _module_for(MSD))]
    for label, ck in runs.items():
        model_specs.append((RUN_DISPLAY[label], ck["samples"], _bm.module))

    corruption_curves = {}   # corruption -> {model: [acc(0), acc(1), ..., acc(5)]}
    for corruption in CORRUPTIONS_TO_PLOT:
        npy_path = CIFAR10C_DIR / f"{corruption}.npy"
        if not npy_path.exists():
            print(f"[{corruption}] missing {npy_path} -- skipped")
            continue
        arr = np.load(npy_path, mmap_mode="r")   # [50000,32,32,3] uint8; sev s = rows (s-1)*10000:s*10000

        levels = [0] + SEVERITIES
        curves = {name: [] for name, _, _ in model_specs}
        for lv in levels:
            Xc = Xc_clean if lv == 0 else normalise_uint8(np.asarray(arr[(lv - 1) * 10_000 + _corrupt_test_idx]))
            for name, samples, module in model_specs:
                n_draws = 1 if samples.shape[0] == 1 else N_DRAWS_CORRUPT
                mean_p, _ = posterior_mean_probs(samples, Xc, n_draws, module=module)
                curves[name].append(accuracy_only(yc, mean_p))
        corruption_curves[corruption] = curves
        print(f"[{corruption}] done: {len(model_specs)} models x {len(levels)} severities")

    corrupt_table = pd.DataFrame(
        {f"{corruption}/{name}": accs
         for corruption, curves in corruption_curves.items()
         for name, accs in curves.items()},
        index=[f"severity={lv}" for lv in [0] + SEVERITIES],
    )

print(f"\n=== Pooled accuracy vs CIFAR-10-C severity ===")
print(corrupt_table.to_string(float_format=lambda v: f"{v:.3f}"))


# %%
# --- FIGURE: 2x2 grid, accuracy vs corruption severity ---
MODEL_COLORS = {"SGD": "#55A868", "MAP (pruned x_ref)": "0.35",
                "Sticky Zig-Zag": "#4C72B0", "Sticky Boomerang": "#DD8452"}

CORRUPTIONS_NAMES = {"gaussian_noise": "Gaussian noise", 
                     "motion_blur":    "Motion blur", 
                     "brightness":     "Brightness", 
                     "fog":            "Fog"}

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
levels = [0] + SEVERITIES
for ax, corruption in zip(axes.flat, CORRUPTIONS_TO_PLOT):
    curves = corruption_curves.get(corruption)
    if curves is None:
        ax.set_title(f"{corruption} (missing)")
        continue
    for name, accs in curves.items():
        if name =="MAP (pruned x_ref)":
            ax.plot(levels, accs, marker="o", ms=4,
                    color=MODEL_COLORS.get(name), label=r"$\beta_{ref}$")
        else:
            ax.plot(levels, accs, marker="o", ms=4,
                    color=MODEL_COLORS.get(name), label=name)
    ax.set(title=CORRUPTIONS_NAMES[corruption], xlabel="Severity", ylabel="Accuracy",
          xticks=levels, ylim=(0, 1))
axes.flat[0].legend(fontsize=15)
fig.suptitle("CIFAR-10-C: Accuracy vs corruption severity", y=1.01)
fig.tight_layout()
plt.show()
fig.savefig(SAVE_DIR / "fig_corruption_grid.pdf", bbox_inches="tight")
if not LOAD_CORRUPT_FROM_SAVED:
    corrupt_table.to_csv(_corrupt_csv_path)

# %%
