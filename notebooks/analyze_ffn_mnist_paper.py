"""

Two analyses, FFN / MNIST -- counterpart to analyze_lenet_mnist_paper.py.
The FFN is a plain fully-connected net (`FFN`, `LAYER_SIZES = [784, 256,
256, 10]`, relu, no BatchNorm, no conv/pool layers).

  A. PREDICTIVE QUALITY & CALIBRATION on clean MNIST, MAP vs posterior:
     accuracy / NLL / ECE / Brier / entropy + reliability diagram + entropy
     split (correct vs wrong).

  B. STRUCTURED-SPARSITY LEDGER: per-layer survival ("ever non-zero" vs
     "non-zero in every draw"), accuracy vs sparsity vs the MAP, and
     per-layer posterior displacement from MAP in prior-std units on the
     non-frozen coordinates. In place of LeNet5's conv-filter death, the
     FFN's structural analog is DEAD UNITS: hidden rows of `layers.0.
     weight`/`layers.1.weight` (and output rows of `layers.2.weight`) that
     are all-zero in every posterior draw -- a whole neuron switched off
     with posterior probability 1 -- plus dead INPUT COLUMNS of
     `layers.0.weight` (pixels the first layer ignores entirely).

Plus the ROTATION / PIXEL-NOISE corruption sweep from ffn_pixel_noise.ipynb
(shared with analyze_lenet_mnist_paper.py via notebooks/utils/image_nets.py):
accuracy / P(true) / confidence / entropy under rotation and additive pixel
noise, pooled over a handful of digit classes.

The saliency / localised border-vs-digit noise analysis that used to live
here has been dropped as redundant with the rotation/noise sweep below.

"""

from __future__ import annotations

# %%
# ==========================================================================
# 0. Config, imports, device
# ==========================================================================
import os
import sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Find the repo root (dir containing "sazz/") regardless of where this is
# launched from, chdir there so relative result paths resolve, and make
# `import sazz...` / `import notebooks.utils...` work even when run as a
# plain script.
_here = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
for _root in [_here, *_here.parents]:
    if (_root / "sazz").is_dir():
        os.chdir(_root)
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break
print("cwd:", Path.cwd())

from notebooks.utils import image_nets as inet

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

RUN_DIR = Path("results/paper/ffn_mnist")
RUN_SPECS = [
    ("zigzag", "grid_sticky_zigzag.pt"),
    ("boomerang", "grid_sticky_boomerang.pt"),
]
RUN_DISPLAY = {"zigzag": "Sticky Zig-Zag", "boomerang": "Sticky Boomerang"}

# pruned+refit MAP -- verified below to bit-match both runs.
MAP_REF_PATH = Path("results/maps/ffn_mnist_reference_N60000_steps10000_pruned_refit_tol03_longrefit.pt")

# Plain SGD baseline (ffn_sgd.py) -- genuinely unrelated to MAP_REF_PATH's
# pruned+refit x_ref: no prior term, no pruning, dense weights throughout.
# This is the frequentist reference the noise sweep needs to play the role
# Izmailov et al. (2021) Fig 15's "SGD" curve plays, since the MAP point is
# a pruned/refit object and not a fair stand-in for it.
SGD_REF_PATH = Path("results/maps/ffn_sgd_N60000_epochs40.pt")

# Prior hyper-params the run used (ffn_mnist_reference.py defaults).
PRIOR_STD_W = 2.0
PRIOR_STD_B = 2.0
FAN_IN_SCALING = True
BASE_SEED = 42
ACTIVATION = "relu"       # overridden from the checkpoint below if it disagrees
LAYER_SIZES = [28 * 28, 256, 256, 10]

N_PRED_DRAWS = 300  # FFN forwards are cheap; 300 draws is fine
N_TEST = 1000
ZERO_TOL = 1e-8

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# --- rotation / noise sweep config (ported from ffn_pixel_noise.ipynb) ---
CLASSES = [0, 2, 4, 6, 8]
N_PER_CLASS = 1_000
N_DRAWS_POOL = 500
POOL_SEED = 0
NOISE_SEED = 12345
ANGLES = [0, 15, 30, 45, 60, 90, 120, 150, 180]
SIGMAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]#, 4.0, 5.0, 7.0, 10.0]
SAMPLER_COLORS = {"zigzag": "#4C72B0", "boomerang": "#DD8452"}

SAVE_DIR = Path("results/plots/MNIST_FFN/")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11, "figure.dpi": 120,
})


# %%
# ==========================================================================
# 1. Load checkpoints + the pruned MAP; verify the MAP matches
# ==========================================================================
runs = inet.load_runs(RUN_DIR, RUN_SPECS)

# pick up activation from the checkpoint (authoritative)
_ck0 = next(iter(runs.values()))
ACTIVATION = _ck0["activation"]

map_ck = torch.load(MAP_REF_PATH, map_location="cpu", weights_only=False)
print(f"\nMAP ref: {MAP_REF_PATH.name}")
print(f"  train_acc={map_ck['train_acc']:.4f}  test_acc={map_ck['test_acc']:.4f}  "
      f"sparsity={float((map_ck['x_ref'] == 0).float().mean()):.4f}")
X_REF = inet.check_map_matches_runs(runs, map_ck, DTYPE)
D = int(X_REF.shape[0])

sgd_ck = torch.load(SGD_REF_PATH, map_location="cpu", weights_only=False)
X_SGD = sgd_ck["x_ref"].to(DTYPE)
assert int(X_SGD.shape[0]) == D, f"SGD x_ref dim {X_SGD.shape[0]} != D {D}"
print(f"\nSGD ref: {SGD_REF_PATH.name}")
print(f"  train_acc={sgd_ck['train_acc']:.4f}  test_acc={sgd_ck['test_acc']:.4f}  "
      f"sparsity={float((X_SGD == 0).float().mean()):.4f}  (dense, unrelated to MAP_REF_PATH)")


# %%
# ==========================================================================
# 2. Layer map + priors  (FFN fixed architecture)
# ==========================================================================
FFN_SHAPES = [
    ("layers.0.weight", (256, 784)), ("layers.0.bias", (256,)),
    ("layers.1.weight", (256, 256)), ("layers.1.bias", (256,)),
    ("layers.2.weight", (10, 256)), ("layers.2.bias", (10,)),
]
assert sum(int(np.prod(s)) for _, s in FFN_SHAPES) == D == 269322

rows, idx = [], 0
layer_slices = {}
for name, shape in FFN_SHAPES:
    n = int(np.prod(shape))
    layer_slices[name] = (idx, idx + n)
    kind = "bias" if len(shape) == 1 else "fc"
    rows.append({"layer": name, "kind": kind, "shape": shape, "start": idx, "stop": idx + n})
    idx += n
layer_df = pd.DataFrame(rows)
LAYERS = [r["layer"] for r in rows]
FC_WEIGHT_LAYERS = [r["layer"] for r in rows if r["kind"] == "fc"]

# Fan-in prior precision, same builder the FFN target used.
from sazz.gpu_friendly.models.neural_networks import FFN
from sazz.gpu_friendly.models.priors import build_fan_in_prior_precision

_ref_module = FFN(LAYER_SIZES, activation=ACTIVATION)
assert sum(p.numel() for p in _ref_module.parameters()) == D
prior_prec = build_fan_in_prior_precision(
    _ref_module, PRIOR_STD_W, PRIOR_STD_B, FAN_IN_SCALING,
    dtype=torch.float32, device="cpu",
)
prior_std = prior_prec.clamp(min=1e-12).rsqrt().cpu().numpy()

print(layer_df.to_string(index=False))
print("fc layers:", FC_WEIGHT_LAYERS)


# %%
# ==========================================================================
# 3. Data -- same seeded MNIST test subset the run evaluated on
# ==========================================================================
from sazz.gpu_friendly.scripts.fast_mnist_cnn import load_mnist_subset

_data = load_mnist_subset(60_000, N_TEST, BASE_SEED, Path("datasets"),
                          dtype=DTYPE, device="cpu")
X_test = _data["X_test"]          # [N,1,28,28] normalised, CPU
y_test = _data["y_test"]
print("X_test:", tuple(X_test.shape), " class balance:",
      torch.bincount(y_test, minlength=10).tolist())


# %%
# ==========================================================================
# 4. Predictive engine
# ==========================================================================
from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.priors import build_fan_in_prior_precision as _bpr

_module = FFN(LAYER_SIZES, activation=ACTIVATION).to(dtype=DTYPE, device=DEVICE)
_module.eval()
_prec_dev = _bpr(_module, PRIOR_STD_W, PRIOR_STD_B, FAN_IN_SCALING, dtype=DTYPE, device=DEVICE)
_bm = BayesianModule.build(_module, likelihood="categorical",
                           X=torch.flatten(X_test[:2], 1).to(DEVICE),
                           y=y_test[:2].to(DEVICE),
                           prior_precision=_prec_dev, dtype=DTYPE, device=DEVICE)

predict_probs = inet.make_predict_probs(_bm, DTYPE, DEVICE, flatten=True)
posterior_mean_probs = inet.make_posterior_mean_probs(predict_probs)


# %%
# ==========================================================================
# ANALYSIS A -- clean predictive quality & calibration
# ==========================================================================
clean_probs, clean_rows = {}, []

_p_map = predict_probs(X_REF, X_test)
clean_probs["MAP"] = {"post_mean": _p_map}
_m = inet.calibration_metrics(y_test, _p_map)
clean_rows.append({"model": "MAP (pruned x_ref)",
                   **{k: _m[k] for k in ("acc", "nll", "ece", "brier", "mean_entropy")}})

for label, ck in runs.items():
    mean_p, draw_p = posterior_mean_probs(ck["samples"], X_test, N_PRED_DRAWS)
    clean_probs[label] = {"post_mean": mean_p, "post_draws": draw_p}
    _m = inet.calibration_metrics(y_test, mean_p)
    pt = draw_p.gather(-1, y_test.long().view(1, -1, 1)
                       .expand(draw_p.shape[0], -1, 1)).squeeze(-1)
    clean_rows.append({"model": RUN_DISPLAY[label],
                       **{k: _m[k] for k in ("acc", "nll", "ece", "brier", "mean_entropy")},
                       "draw_std_P(true)": pt.std(0).mean().item()})

clean_df = pd.DataFrame(clean_rows).set_index("model")
print("\n=== Analysis A: clean MNIST test subset ===")
print(clean_df.to_string(float_format=lambda v: f"{v:.4f}"))
# clean_df.to_csv(SAVE_DIR / "tableA_clean_metrics.csv")



# %%
# --- FIGURE A: predictive entropy, correct vs wrong ---
fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4), squeeze=False)
for ax, (label, _) in zip(axes[0], runs.items()):
    mp = clean_probs[label]["post_mean"]
    ent = inet.entropy(mp)
    ok = mp.argmax(-1) == y_test
    ax.hist(ent[ok].numpy(), bins=40, histtype="step", lw=1.5, label="correct")
    ax.hist(ent[~ok].numpy(), bins=40, histtype="step", lw=1.5, ls="--", label="wrong")
    ax.set(xlabel="predictive entropy", ylabel="count", yscale="log",
           title=f"{RUN_DISPLAY[label]}: entropy by outcome")
    ax.legend()
fig.tight_layout()
plt.show()
plt.close()
# fig.savefig(SAVE_DIR / "figA2_entropy_split.pdf", bbox_inches="tight")


# %%
# ==========================================================================
# ANALYSIS B -- structured-sparsity ledger
# ==========================================================================
# ledgers = {label: inet.sparsity_ledger(ck["samples"], ZERO_TOL) for label, ck in runs.items()}

# # --- B1: per-layer survival ---
# c1_df = inet.layer_survival_table(ledgers, RUN_DISPLAY, LAYERS, layer_slices)
# print("\n=== Analysis B1: per-layer weight survival ===")
# print(c1_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# # c1_df.to_csv(SAVE_DIR / "tableB1_layer_survival.csv", index=False)

# inet.print_static_vs_dynamic_sparsity(runs, ledgers, RUN_DISPLAY, ZERO_TOL)

# fig = inet.plot_layer_survival(c1_df, ledgers, RUN_DISPLAY, LAYERS,
#                                "Per-layer weight survival under the sticky-PDMP posterior")
# plt.show()
# # fig.savefig(SAVE_DIR / "figB1_layer_survival.pdf", bbox_inches="tight")


# %%
# --- B2: dead UNITS with posterior prob 1 (FFN's structural analog of
#         LeNet5's dead conv filters). For each weight layer [out, in]:
#   dead_output_units -- a hidden unit / output logit whose entire incoming
#                         weight row is zero in EVERY draw: that unit's
#                         pre-activation is input-independent (constant, set
#                         by its bias alone) with posterior probability 1 --
#                         a whole neuron switched off.
#   dead_input_units  -- (layers.0.weight only) a column, i.e. an input
#                         pixel, that is zero in every draw for every hidden
#                         unit: the network provably never reads that pixel.
# c2_rows = []
# for label, ck in runs.items():
#     s = ck["samples"]
#     for lname in FC_WEIGHT_LAYERS:
#         a, b = layer_slices[lname]
#         shp = dict(_ref_module.named_parameters())[lname].shape   # [out, in]
#         w = s[:, a:b].reshape(s.shape[0], *shp)
#         wz = (w.abs() < ZERO_TOL)
#         row_dead = wz.all(dim=(0, 2))      # [out] -- dead output units
#         col_dead = wz.all(dim=(0, 1))      # [in]  -- dead input units
#         c2_rows.append({
#             "run": RUN_DISPLAY[label], "layer": lname,
#             "out": shp[0], "in": shp[1],
#             "dead_output_units": int(row_dead.sum()), "dead_output_frac": row_dead.float().mean().item(),
#             "dead_input_units": int(col_dead.sum()), "dead_input_frac": col_dead.float().mean().item(),
#         })
# c2_df = pd.DataFrame(c2_rows)
# print("\n=== Analysis B2: dead units (posterior prob 1) ===")
# print(c2_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# c2_df.to_csv(SAVE_DIR / "tableB2_dead_units.csv", index=False)


# %%
# --- B3: accuracy vs sparsity, posterior vs MAP ---
# c3_df = inet.accuracy_vs_sparsity_table(clean_df, X_REF, runs, RUN_DISPLAY, ZERO_TOL)
# print("\n=== Analysis B3: accuracy vs sparsity ===")
# print(c3_df.to_string(float_format=lambda v: f"{v:.4f}"))
# c3_df.to_csv(SAVE_DIR / "tableB3_acc_vs_sparsity.csv")


# %%
# --- B4: per-layer posterior displacement from MAP (non-frozen coords) ---
# c4_df = inet.displacement_table(runs, ledgers, RUN_DISPLAY, LAYERS, layer_slices,
#                                 X_REF, prior_std, D)
# print("\n=== Analysis B4: posterior displacement from MAP (moving coords) ===")
# print(c4_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# # c4_df.to_csv(SAVE_DIR / "tableB4_displacement.csv", index=False)

# fig = inet.plot_displacement(c4_df, runs, RUN_DISPLAY, LAYERS,
#                              "Posterior exploration around the MAP, per layer (moving coords only)")
# plt.show()
# fig.savefig(SAVE_DIR / "figB4_displacement.pdf", bbox_inches="tight")


# %%
# ==========================================================================
# ANALYSIS C -- rotation / pixel-noise corruption sweep
#   (ported from ffn_pixel_noise.ipynb; MAP flows through as a 1-draw point
#   estimate so it renders on the same axes as the samplers)
# ==========================================================================
sweep_runs = dict(runs)
sweep_runs["map"] = {"samples": X_REF.unsqueeze(0)}
sweep_runs["sgd"] = {"samples": X_SGD.unsqueeze(0)}
SWEEP_DISPLAY = {**RUN_DISPLAY, "map": r"$\beta_{\mathrm{ref}}$", "sgd": "SGD"}
SWEEP_COLORS = {**SAMPLER_COLORS, "map": "0.35", "sgd": "#55A868"}


def _rotate(X, lv, seed=0):
    return inet.rotate_batch(X, lv, MNIST_MEAN, MNIST_STD)


def _noise(X, lv, seed=0):
    return inet.noise_batch(X, lv, MNIST_MEAN, MNIST_STD, seed=seed)


rot_levels, rot_spans, rot_X, rot_agg = inet.run_shift_sweep(
    ANGLES, _rotate, CLASSES, y_test, X_test, sweep_runs, predict_probs,
    N_PER_CLASS, N_DRAWS_POOL, POOL_SEED, NOISE_SEED)
noi_levels, noi_spans, noi_X, noi_agg = inet.run_shift_sweep(
    SIGMAS, _noise, CLASSES, y_test, X_test, sweep_runs, predict_probs,
    N_PER_CLASS, N_DRAWS_POOL, POOL_SEED, NOISE_SEED)
print(f"[rotation] {len(sweep_runs)} models x {len(CLASSES)} digits x {len(ANGLES)} levels")
print(f"[noise]    {len(sweep_runs)} models x {len(CLASSES)} digits x {len(SIGMAS)} levels")

noise_table = pd.DataFrame(
    {SWEEP_DISPLAY.get(lbl, lbl): [noi_agg[lbl][("pool", lv)]["acc"] for lv in SIGMAS]
     for lbl in sweep_runs},
    index=[f"sigma={lv:g}" for lv in SIGMAS],
)
print("\n=== Pooled test accuracy vs Gaussian noise sigma (raw [0,1]-pixel units) ===")
print(noise_table.to_string(float_format=lambda v: f"{v:.3f}"))


# %%
# fig = inet.shift_figure("noise", noi_levels, noi_agg, sweep_runs, SWEEP_DISPLAY, CLASSES,
#                         xlabel=r"Pixel-noise $\sigma$", title="MNIST FFN under pixel noise",
#                         point_labels=("map", "sgd"))
# plt.show()
# # fig.savefig(SAVE_DIR / "MNIST_noise_paper.pdf", bbox_inches="tight")

# fig = inet.shift_figure("rotation", rot_levels, rot_agg, sweep_runs, SWEEP_DISPLAY, CLASSES,
#                         xlabel="Rotation (degrees)", title="MNIST FFN under rotation",
#                         point_labels=("map", "sgd"))
# plt.show()
# fig.savefig(SAVE_DIR / "MNIST_rotation_paper.pdf", bbox_inches="tight")


# %%
# --- Izmailov et al. (2021) Figure-15-style single panel: pooled accuracy
#     (NOT split by true digit) vs noise sigma, SGD vs MAP vs PDMP samplers
#     on one axis, directly comparable in spirit to their SGD-vs-HMC plot.
fig = inet.paper_style_figure(
    noi_levels, noi_agg, SWEEP_DISPLAY, SWEEP_COLORS,
    xlabel=r" Noise Scale $\sigma$", #title="MNIST FFN: robustness to pixel noise",
    point_labels=("map", "sgd"), order=["sgd", "map", "zigzag", "boomerang"],
)
plt.show()
fig.savefig(SAVE_DIR / "MNIST_noise_paperstyle.pdf", bbox_inches="tight")


# %%
# fig = inet.bar_grid_appendix("noise", noi_levels, noi_agg,
#                              sweep_runs, SWEEP_DISPLAY, CLASSES, SWEEP_COLORS,
#                              point_labels=("map", "sgd"))
# plt.show()
# # fig.savefig(SAVE_DIR / "MNIST_noise_bars_appendix.pdf", bbox_inches="tight")

# fig = inet.bar_grid_appendix("rotation", rot_levels, rot_agg,
#                              sweep_runs, SWEEP_DISPLAY, CLASSES, SWEEP_COLORS,
#                              point_labels=("map", "sgd"))
# plt.show()
# fig.savefig(SAVE_DIR / "MNIST_rotation_bars_appendix.pdf", bbox_inches="tight")


# %%
# ==========================================================================
# 5. One-screen summary for the paper's text
# ==========================================================================
# print("\n" + "=" * 70)
# print("SUMMARY -- numbers to quote (FFN / MNIST)")
# print("=" * 70)
# mrow = clean_df.loc["MAP (pruned x_ref)"]
# print(f"\nMAP (pruned x_ref): acc={mrow['acc']:.3f}  NLL={mrow['nll']:.3f}  "
#       f"ECE={mrow['ece']:.3f}  Brier={mrow['brier']:.3f}  H={mrow['mean_entropy']:.3f}  "
#       f"sparsity={c3_df.loc['MAP (pruned x_ref)', 'weight_sparsity']:.3f}")
# for label, ck in runs.items():
#     name = RUN_DISPLAY[label]
#     L = ledgers[label]
#     row = clean_df.loc[name]
#     print(f"\n[{name}]")
#     print(f"  clean:  acc={row['acc']:.3f}  NLL={row['nll']:.3f}  ECE={row['ece']:.3f}  "
#           f"Brier={row['brier']:.3f}  H={row['mean_entropy']:.3f}")
#     per_draw = (np.abs(ck["samples"].numpy()) < ZERO_TOL).mean(1).mean()
#     churn = 1 - L["always_zero"].mean() - L["always_nonzero"].mean()
#     print(f"  weight sparsity: per-draw {per_draw:.3f}  |  zero in EVERY draw "
#           f"{L['always_zero'].mean():.3f} ({int(L['always_zero'].sum()):,}/{D:,})  |  "
#           f"churning {churn:.3f}")
#     dead_u = c2_df[c2_df['run'] == name]['dead_output_units'].sum()
#     tot_u = c2_df[c2_df['run'] == name]['out'].sum()
#     print(f"  hidden/output units dead w.p. 1: {dead_u} / {tot_u}")
#     sev3_noise = noi_agg[label][("pool", SIGMAS[-1])]
#     print(f"  noise sigma={SIGMAS[-1]:g}: acc={sev3_noise['acc']:.3f}  "
#           f"conf={sev3_noise['conf']:.3f}  entropy={sev3_noise['entropy']:.3f}")
#     rot180 = rot_agg[label][("pool", 180)]
#     print(f"  rotation 180deg:      acc={rot180['acc']:.3f}  "
#           f"conf={rot180['conf']:.3f}  entropy={rot180['entropy']:.3f}")
