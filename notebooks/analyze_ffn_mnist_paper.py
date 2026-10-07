from __future__ import annotations

# %%
# ==========================================================================
# 0. Config, imports, device
# ==========================================================================
import os
import pickle
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

RUN_DIR = Path("results/paper_v2/ffn_mnist")
# paper_v2 files have no grid_ prefix (renamed after sync), fall back to it if not renamed yet
def _run_file(name):
    return next((n for n in (name, "grid_" + name) if (RUN_DIR / n).exists()), name)


RUN_SPECS = [
    ("zigzag", _run_file("sticky_zigzag.pt")),
    ("boomerang", _run_file("sticky_boomerang.pt")),
    ("dense", _run_file("zigzag.pt")),   # dense ZigZag from the unpruned MAP, step 19
]
RUN_DISPLAY = {"zigzag": "Sticky Zig-Zag", "boomerang": "Sticky Boomerang", "dense": "ZigZag"}

# pruned+refit MAP -- verified below to bit-match both runs.
MAP_REF_PATH = Path("results/maps/ffn_mnist_reference_N60000_steps10000_pruned_refit_tol03_longrefit.pt")
MAP_unpruned_REF_PATH = Path("results/maps/ffn_mnist_reference_N60000_steps10000.pt")

# Plain SGD baseline (ffn_sgd.py) -- genuinely unrelated to the samplers
SGD_REF_PATH = Path("results/maps/ffn_sgd_N60000_epochs40.pt")

# Prior hyper-params the run used (ffn_mnist_reference.py defaults).
PRIOR_STD_W = 2.0
PRIOR_STD_B = 2.0
FAN_IN_SCALING = True
BASE_SEED = 42
ACTIVATION = "relu"       # overridden from the checkpoint below if it disagrees
LAYER_SIZES = [28 * 28, 256, 256, 10]

N_PRED_DRAWS = 500  # same number of draws as the noise sweep (N_DRAWS_POOL)
N_TEST = 10_000  # full MNIST test pool
ZERO_TOL = 1e-8

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# --- rotation / noise sweep config (ported from ffn_pixel_noise.ipynb) ---
CLASSES = list(range(10))   # all digits, every test image of each (N_PER_CLASS is only a cap)
N_PER_CLASS = 1_000
N_DRAWS_POOL = 500
POOL_SEED = 0
NOISE_SEED = 12345
ANGLES = [0, 15, 30, 45, 60, 90, 120, 150, 180]
SIGMAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]#, 4.0, 5.0, 7.0, 10.0]
SAMPLER_COLORS = {"zigzag": "#4C72B0", "boomerang": "#DD8452"}

SAVE_DIR = Path("results/plots_v2/MNIST_FFN/")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)
# noise sweep results, reused while the settings below match. Delete the file to recompute.
NOISE_CACHE = SAVE_DIR / "noise_sweep_cache.pkl"
NOISE_META = {"test_set": "8000 without pruning images", "sigmas": SIGMAS, "classes": CLASSES, "n_per_class": N_PER_CLASS,
              "n_draws_pool": N_DRAWS_POOL, "pool_seed": POOL_SEED, "noise_seed": NOISE_SEED,
              "runs": [f for _, f in RUN_SPECS if (RUN_DIR / f).exists()]}   # only runs present, so a new run invalidates the cache

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
X_REF = inet.check_map_matches_runs({k: v for k, v in runs.items() if k != "dense"}, map_ck, DTYPE)   # the dense run starts unpruned
D = int(X_REF.shape[0])

sgd_ck = torch.load(SGD_REF_PATH, map_location="cpu", weights_only=False)
X_SGD = sgd_ck["x_ref"].to(DTYPE)
assert int(X_SGD.shape[0]) == D, f"SGD x_ref dim {X_SGD.shape[0]} != D {D}"
print(f"\nSGD ref: {SGD_REF_PATH.name}")
print(f"  train_acc={sgd_ck['train_acc']:.4f}  test_acc={sgd_ck['test_acc']:.4f}  "
      f"sparsity={float((X_SGD == 0).float().mean()):.4f}  (dense, unrelated to MAP_REF_PATH)")

# unpruned MAP, the dense ZigZag's start
X_MAP_FULL = torch.load(MAP_unpruned_REF_PATH, map_location="cpu", weights_only=False)["x_ref"].to(DTYPE)
assert int(X_MAP_FULL.shape[0]) == D, f"unpruned MAP dim {X_MAP_FULL.shape[0]} != D {D}"
print(f"\nunpruned MAP ref: {MAP_unpruned_REF_PATH.name}  "
      f"sparsity={float((X_MAP_FULL == 0).float().mean()):.4f}")


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
# Leave out the 2000 images that chose the pruning threshold of the MAP. load_mnist_subset
# draws the training set, then permutes the test set, so replay that to get X_test's indices.
_rng = np.random.default_rng(BASE_SEED)
_ = _rng.choice(60_000, size=60_000, replace=False)
TEST_IDX = _rng.permutation(10_000)[:N_TEST]
_keep = torch.as_tensor(~np.isin(TEST_IDX, inet.pruning_sweep_indices(60_000)))
X_test, y_test, TEST_IDX = X_test[_keep], y_test[_keep], TEST_IDX[_keep.numpy()]
N_TEST = len(y_test)              # 8000, disjoint from the pruning images
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

_p_map_full = predict_probs(X_MAP_FULL, X_test)
clean_probs["MAP (unpruned)"] = {"post_mean": _p_map_full}
_m = inet.calibration_metrics(y_test, _p_map_full)
clean_rows.append({"model": "MAP (unpruned)",
                   **{k: _m[k] for k in ("acc", "nll", "ece", "brier", "mean_entropy")}})

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



# --- FIGURE A2: predictive entropy, correct vs wrong ---
_p_sgd = predict_probs(X_SGD, X_test)
_ent_sgd = inet.entropy(_p_sgd)
_ok_sgd = _p_sgd.argmax(-1) == y_test

import numpy as np
from scipy.stats import gaussian_kde

fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4), squeeze=False)

for ax, (label, _) in zip(axes[0], runs.items()):
    mp = clean_probs[label]["post_mean"]
    ent = inet.entropy(mp).numpy()
    ok = (mp.argmax(-1) == y_test).numpy()
    ent_sgd = _ent_sgd.numpy()
    ok_sgd = _ok_sgd.numpy()
    for x, color, ls, lbl in [
        (ent[ok], "tab:green", "-", "correct"),
        (ent[~ok], "tab:red", "-", "wrong"),
        (ent_sgd[ok_sgd], "tab:green", "--", "SGD correct"),
        (ent_sgd[~ok_sgd], "tab:red", "--", "SGD wrong"),
    ]:
        kde = gaussian_kde(x)
        xx = np.linspace(0, max(ent.max(), ent_sgd.max()), 500)
        ax.plot(xx, kde(xx), color=color, ls=ls, lw=1.8, label=lbl)
    ax.set(
        xlabel="predictive entropy", ylabel="density", yscale="log", title=f"{RUN_DISPLAY[label]}: entropy by outcome",
    )
    ax.legend()
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "figA2_entropy_split.pdf", bbox_inches="tight")


# %%
# ==========================================================================
# ANALYSIS B -- headline predictive-metrics table: SGD, MAP, both samplers
# ==========================================================================
#_p_sgd = predict_probs(X_SGD, X_test)
_m_sgd = inet.calibration_metrics(y_test, _p_sgd)

summary_rows = [{"model": "SGD", **{k: _m_sgd[k] for k in ("acc", "ece", "nll", "brier")}}]
for row in clean_rows:
    summary_rows.append({"model": row["model"],
                         **{k: row[k] for k in ("acc", "ece", "nll", "brier")}})

summary_df = pd.DataFrame(summary_rows).set_index("model")
print(f"\n=== Analysis B: predictive metrics on the MNIST test images not used for pruning (N={N_TEST}) ===")
print(summary_df.to_string(float_format=lambda v: f"{v:.4f}"))
# summary_df.to_csv(SAVE_DIR / "tableB_headline_metrics.csv")


# %%
# ==========================================================================
# ANALYSIS C -- noise / rotation
# ==========================================================================
sweep_runs = dict(runs)
sweep_runs["map"] = {"samples": X_REF.unsqueeze(0)}
sweep_runs["sgd"] = {"samples": X_SGD.unsqueeze(0)}
SWEEP_DISPLAY = {**RUN_DISPLAY, "map": r"$\beta_{\mathrm{ref}}$", "sgd": "SGD"}
SWEEP_COLORS = {**SAMPLER_COLORS, "map": "0.35", "sgd": "#55A868"}

# unpruned MAP (the dense ZigZag's start), a single network like the pruned MAP and SGD
sweep_runs["map_full"] = {"samples": X_MAP_FULL.unsqueeze(0)}
SWEEP_DISPLAY["map_full"] = "MAP"


def _rotate(X, lv, seed=0):
    return inet.rotate_batch(X, lv, MNIST_MEAN, MNIST_STD)


def _noise(X, lv, seed=0):
    return inet.noise_batch(X, lv, MNIST_MEAN, MNIST_STD, seed=seed)


# rot_levels, rot_spans, rot_X, rot_agg = inet.run_shift_sweep(
#     ANGLES, _rotate, CLASSES, y_test, X_test, sweep_runs, predict_probs,
#     N_PER_CLASS, N_DRAWS_POOL, POOL_SEED, NOISE_SEED)
_cache = pickle.loads(NOISE_CACHE.read_bytes()) if NOISE_CACHE.exists() else None
if _cache is not None and _cache["meta"] == NOISE_META:
    noi_levels, noi_agg = _cache["levels"], _cache["agg"]
    print(f"[noise] loaded {NOISE_CACHE}")
    # models added after the cache was written are swept on their own and merged in;
    # the pooled images and the noise come from the same seeds, so the results are comparable
    _missing = {lbl: run for lbl, run in sweep_runs.items() if lbl not in noi_agg}
    if _missing:
        _, _, _, _new_agg = inet.run_shift_sweep(
            SIGMAS, _noise, CLASSES, y_test, X_test, _missing, predict_probs,
            N_PER_CLASS, N_DRAWS_POOL, POOL_SEED, NOISE_SEED)
        noi_agg.update(_new_agg)
        NOISE_CACHE.write_bytes(pickle.dumps({"meta": NOISE_META, "levels": noi_levels, "agg": noi_agg}))
        print(f"[noise] added {list(_missing)} to {NOISE_CACHE}")

else:
    noi_levels, noi_spans, noi_X, noi_agg = inet.run_shift_sweep(
        SIGMAS, _noise, CLASSES, y_test, X_test, sweep_runs, predict_probs,
        N_PER_CLASS, N_DRAWS_POOL, POOL_SEED, NOISE_SEED)
    NOISE_CACHE.write_bytes(pickle.dumps({"meta": NOISE_META, "levels": noi_levels, "agg": noi_agg}))
    print(f"[noise] {len(sweep_runs)} models x {len(CLASSES)} digits x {len(SIGMAS)} levels, "
          f"saved to {NOISE_CACHE}")
#print(f"[rotation] {len(sweep_runs)} models x {len(CLASSES)} digits x {len(ANGLES)} levels")

noise_table = pd.DataFrame(
    {SWEEP_DISPLAY.get(lbl, lbl): [noi_agg[lbl][("pool", lv)]["acc"] for lv in SIGMAS]
     for lbl in sweep_runs},
    index=[f"sigma={lv:g}" for lv in SIGMAS],
)
print("\n=== Pooled test accuracy vs Gaussian noise sigma (raw [0,1]-pixel units) ===")
print(noise_table.to_string(float_format=lambda v: f"{v:.3f}"))

# %%
# --- Paper figure, Izmailov et al. (2021) Figure 15 style. Reads only the cache,
# so after cell 0 this cell runs on its own in about a second. Edit and rerun freely.

def plot_noise(levels, agg, save_name="MNIST_noise_paperstyle"):
    TITLE, LABEL, TICK, LEGEND = 20, 20, 18, 20
    # order = ["sgd", "map", "dense", "zigzag"]#, "boomerang"]
    # point = {"sgd", "map"}   # dashed lines
    # display = {"sgd": "SGD", "map": r"$\beta_{\mathrm{ref}}$",
    #            "dense": "ZigZag", "zigzag": "Sticky ZigZag"} #, "boomerang": "Sticky Boomerang"}
    # colors = {"sgd": "C4", "map": "0.35", "dense": "C2", "zigzag": "C3"}#, "boomerang": "#DD8452"}
    order = ["sgd", "map_full", "map", "dense", "zigzag"]
    point = {"sgd", "map", "map_full"}   # dashed lines
    display = {"sgd": "SGD", "map_full": r"$\beta_{\mathrm{ref}}$", "map": r"$\beta_{\mathrm{ref}}^{\star}$",
               "dense": "ZigZag", "zigzag": "Sticky ZigZag"}
    colors = {"sgd": "C4", "map_full": "0.65", "map": "0.35", "dense": "C2", "zigzag": "C3"}

    panels = [("acc", "Accuracy", (-0.02, 1.02)),
              ("p_true", "P(true class)", (-0.02, 1.02)),
              ("entropy", "Predictive entropy", (-0.05, np.log(10) * 1.08))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, title, ylim) in zip(axes, panels):
        for lbl in [l for l in order if l in agg]:
            ax.plot(levels, [agg[lbl][("pool", lv)][key] for lv in levels],
                    marker="o", ms=5, lw=2.2, ls="--" if lbl in point else "-",
                    color=colors[lbl], label=display[lbl])
        ax.set_title(title, fontsize=TITLE)
        ax.set_xlabel(r"Noise scale $\sigma$", fontsize=LABEL)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=TICK)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    leg = fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(order),
                     fontsize=LEGEND, frameon=False, bbox_to_anchor=(0.5, 1.2), handlelength=3)
    for line in leg.get_lines():   # thicker lines and markers in the legend only
        line.set_linewidth(4)
        line.set_markersize(9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(SAVE_DIR / f"{save_name}.{ext}", bbox_inches="tight", dpi=200)
    plt.show()


_noise_cache = pickle.loads(NOISE_CACHE.read_bytes())
plot_noise(_noise_cache["levels"], _noise_cache["agg"])



# %%
# import pickle
# NOISE_CACHE = SAVE_DIR / "noise_sweep_cache.pkl"
# NOISE_META = {"sigmas": SIGMAS, "classes": CLASSES, "n_per_class": N_PER_CLASS,
#               "n_draws_pool": N_DRAWS_POOL, "pool_seed": POOL_SEED, "noise_seed": NOISE_SEED,
#               "runs": [f for _, f in RUN_SPECS]}
# NOISE_CACHE.write_bytes(pickle.dumps({"meta": NOISE_META, "levels": noi_levels, "agg": noi_agg}))

# # %%
