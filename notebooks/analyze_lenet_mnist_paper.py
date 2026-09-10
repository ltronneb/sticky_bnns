"""

Three analyses:

  A. PREDICTIVE QUALITY & CALIBRATION on clean MNIST, MAP vs posterior:
     accuracy / NLL / ECE / Brier / entropy + reliability diagram + entropy
     split (correct vs wrong).

  B. THE DEAD BORDER (the MNIST-specific bit you actually want). MNIST
     digits sit centred in a 28x28 frame; a ring of edge pixels is black in
     every image. Two questions:
       B1. Input-pixel saliency |d max-logit / d pixel|, averaged over the
           test set: has the net (MAP, and the posterior mean) learned to
           put ~zero sensitivity on never-ink pixels? Report the
           border/digit saliency ratio and the posterior's across-draw
           saliency spread. (The conv1-*weight* level does NOT carry this
           signal -- the never-ink frame is too thin/irregular for any 5x5
           tap to be border-only -- so this lives at the input level.)
       B2. Localised-corruption robustness: add Gaussian noise to the
           never-ink (BORDER) pixels only vs the DIGIT pixels only, at 3
           severities. A calibrated model should be near-invariant to
           border noise and appropriately less confident under digit noise.
           Compare MAP vs posterior degradation -- this is the direct,
           prediction-level version of B1.

  C. STRUCTURED-SPARSITY LEDGER: per-layer survival ("ever non-zero" vs
     "non-zero in every draw"), conv-filter / (filter,in-channel) death
     with posterior probability 1, accuracy vs sparsity vs the MAP, and
     per-layer posterior displacement from MAP in prior-std units on the
     non-frozen coordinates.

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

RUN_DIR = Path("results/paper/mnist_cnn/split_00")
RUN_SPECS = [
    ("zigzag", "grid_sticky_zigzag.pt"),
    ("boomerang", "grid_sticky_boomerang.pt"),
]
RUN_DISPLAY = {"zigzag": "Sticky Zig-Zag", "boomerang": "Sticky Boomerang"}

# 88.8%-sparse pruned+refit MAP -- verified below to bit-match both runs.
MAP_REF_PATH = Path("results/maps/lenet_reference_N60000_pruned_refit_N60k.pt")

# Prior hyper-params the run used (fast_mnist_cnn.py defaults).
PRIOR_STD_W = 2.0
PRIOR_STD_B = 2.0
FAN_IN_SCALING = True
BASE_SEED = 42
ACTIVATION = "relu"
POOL = "max"        # overridden from the checkpoint below if it disagrees

N_PRED_DRAWS = 300  # LeNet forwards are cheap; 300 draws is fine
N_TEST = 1000
ZERO_TOL = 1e-8

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

SAVE_DIR = Path("results/paper/mnist_cnn/figs")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11, "figure.dpi": 120,
})


# %%
# ==========================================================================
# 1. Load checkpoints + the pruned MAP; verify the MAP matches
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
    print(f"[{label}] {n} draws x D={D}  test_acc(ckpt)={ck['test_accuracy']:.4f}  "
          f"sparsity(ckpt)={ck['sparsity_frac']:.4f}  prune_frac={ck['prune_frac']:.4f}  "
          f"pool={ck['pool']}  act={ck['activation']}  wall={ck['elapsed_sec']/3600:.1f}h")
assert runs, f"no run files under {RUN_DIR}"

# pick up pool/activation from the checkpoint (authoritative)
_ck0 = next(iter(runs.values()))
POOL = _ck0["pool"]
ACTIVATION = _ck0["activation"]

map_ck = torch.load(MAP_REF_PATH, map_location="cpu", weights_only=False)
X_REF = map_ck["x_ref"].to(DTYPE)
D = int(X_REF.shape[0])
print(f"\nMAP ref: {MAP_REF_PATH.name}")
print(f"  D={D}  train_acc={map_ck['train_acc']:.4f}  test_acc={map_ck['test_acc']:.4f}  "
      f"sparsity={float((X_REF == 0).float().mean()):.4f}")

for label, ck in runs.items():
    dmax = (ck["x_ref"].to(DTYPE) - X_REF).abs().max().item()
    cs_ok = torch.equal(ck["cold_start_mask"].bool(), (X_REF == 0))
    assert dmax < 1e-5 and cs_ok, (
        f"[{label}] MAP MISMATCH: max|x_ref - MAP.x_ref|={dmax:.3e}, cold_start match={cs_ok}. "
        f"MAP_REF_PATH is not this run's cold-start point."
    )
    print(f"  [{label}] matches run x_ref (max|Δ|={dmax:.1e}) and cold_start_mask OK")


# %%
# ==========================================================================
# 2. Layer map + priors  (LeNet5 fixed architecture)
# ==========================================================================
LENET5_SHAPES = [
    ("conv1.weight", (6, 1, 5, 5)), ("conv1.bias", (6,)),
    ("conv2.weight", (16, 6, 5, 5)), ("conv2.bias", (16,)),
    ("fc1.weight", (120, 400)), ("fc1.bias", (120,)),
    ("fc2.weight", (84, 120)), ("fc2.bias", (84,)),
    ("fc3.weight", (10, 84)), ("fc3.bias", (10,)),
]
assert sum(int(np.prod(s)) for _, s in LENET5_SHAPES) == D == 61706

rows, idx = [], 0
layer_slices = {}
for name, shape in LENET5_SHAPES:
    n = int(np.prod(shape))
    layer_slices[name] = (idx, idx + n)
    kind = "bias" if len(shape) == 1 else ("conv" if name.startswith("conv") else "fc")
    rows.append({"layer": name, "kind": kind, "shape": shape, "start": idx, "stop": idx + n})
    idx += n
layer_df = pd.DataFrame(rows)
CONV_LAYERS = [r["layer"] for r in rows if r["kind"] == "conv"]
WEIGHT_LAYERS = [r["layer"] for r in rows if r["kind"] in ("conv", "fc")]

coord_layer = np.empty(D, dtype=object)
for r in rows:
    coord_layer[r["start"]:r["stop"]] = r["layer"]

# Fan-in prior precision, same builder the LeNet target used.
from sazz.gpu_friendly.models.neural_networks import LeNet5
from sazz.gpu_friendly.models.priors import build_fan_in_prior_precision

_ref_module = LeNet5(activation=ACTIVATION, pool=POOL)
assert sum(p.numel() for p in _ref_module.parameters()) == D
prior_prec = build_fan_in_prior_precision(
    _ref_module, PRIOR_STD_W, PRIOR_STD_B, FAN_IN_SCALING,
    dtype=torch.float32, device="cpu",
)
prior_std = prior_prec.clamp(min=1e-12).rsqrt().cpu().numpy()

print(layer_df.to_string(index=False))
print("conv layers:", CONV_LAYERS)


# %%
# ==========================================================================
# 3. Data -- same seeded 1k MNIST test subset the run evaluated on
# ==========================================================================
from sazz.gpu_friendly.scripts.fast_mnist_cnn import load_mnist_subset

_data = load_mnist_subset(60_000, N_TEST, BASE_SEED, Path("datasets"),
                          dtype=DTYPE, device="cpu")
X_test = _data["X_test"]          # [N,1,28,28] normalised, CPU
y_test = _data["y_test"]
print("X_test:", tuple(X_test.shape), " class balance:",
      torch.bincount(y_test, minlength=10).tolist())

# De-normalised images and the data-driven "never-ink" pixel mask: pixels
# whose de-normalised value never exceeds INK_THRESH anywhere in the test
# subset. This is the honest dead-border definition -- not a fixed ring.
INK_THRESH = 0.05
_img = X_test * MNIST_STD + MNIST_MEAN            # ~[0,1]
_pix_max = _img.amax(dim=0).squeeze(0)            # [28,28]
never_ink = (_pix_max <= INK_THRESH).numpy()     # [28,28] bool
print(f"never-ink pixels (max de-norm <= {INK_THRESH}): {never_ink.sum()} / 784")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
ax[0].imshow(_img.mean(0).squeeze(0), cmap="gray"); ax[0].set_title("mean test image")
ax[1].imshow(_pix_max, cmap="magma"); ax[1].set_title("per-pixel max (de-norm)")
ax[2].imshow(never_ink, cmap="gray"); ax[2].set_title(f"never-ink mask ({never_ink.sum()} px)")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "fig_border_mask.pdf", bbox_inches="tight")


# %%
# ==========================================================================
# 4. Predictive engine
# ==========================================================================
from sazz.gpu_friendly.models.model import BayesianModule
from sazz.gpu_friendly.models.priors import build_fan_in_prior_precision as _bpr

_module = LeNet5(activation=ACTIVATION, pool=POOL).to(dtype=DTYPE, device=DEVICE)
_module.eval()
_prec_dev = _bpr(_module, PRIOR_STD_W, PRIOR_STD_B, FAN_IN_SCALING, dtype=DTYPE, device=DEVICE)
_bm = BayesianModule.build(_module, likelihood="categorical",
                           X=X_test[:2].to(DEVICE), y=y_test[:2].to(DEVICE),
                           prior_precision=_prec_dev, dtype=DTYPE, device=DEVICE)
_param_dict_fn = _bm.param_dict_fn


@torch.no_grad()
def predict_probs(beta: torch.Tensor, X: torch.Tensor, batch: int = 512) -> torch.Tensor:
    beta = beta.to(dtype=DTYPE, device=DEVICE)
    out = []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch].to(dtype=DTYPE, device=DEVICE)
        logits = torch.func.functional_call(_bm.module, _param_dict_fn(beta), (xb,))
        out.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def posterior_mean_probs(samples, X, n_draws, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(samples.shape[0], generator=g)[:min(n_draws, samples.shape[0])]
    dp = torch.stack([predict_probs(samples[i], X) for i in idx])
    return dp.mean(0), dp


# %%
# ==========================================================================
# 5. Metrics
# ==========================================================================
def _entropy(probs, eps=1e-12):
    return -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(-1)


def calibration_metrics(y_true, probs, n_bins=15):
    y_true = y_true.long()
    conf, pred = probs.max(-1)
    correct = (pred == y_true).float()
    acc = correct.mean().item()
    p_true = probs.gather(-1, y_true.unsqueeze(-1)).squeeze(-1)
    nll = -p_true.clamp_min(1e-12).log().mean().item()
    onehot = F.one_hot(y_true, probs.shape[-1]).float()
    brier = ((probs - onehot) ** 2).sum(-1).mean().item()
    ent = _entropy(probs).mean().item()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece, bin_stats = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ba, bc, w = correct[m].mean().item(), conf[m].mean().item(), m.float().mean().item()
            ece += w * abs(ba - bc)
            bin_stats.append((bc, ba, w))
    return {"acc": acc, "nll": nll, "ece": ece, "brier": brier,
            "mean_entropy": ent, "bin_stats": bin_stats}


# %%
# ==========================================================================
# ANALYSIS A -- clean predictive quality & calibration
# ==========================================================================
clean_probs, clean_rows = {}, []

_p_map = predict_probs(X_REF, X_test)
clean_probs["MAP"] = {"post_mean": _p_map}
_m = calibration_metrics(y_test, _p_map)
clean_rows.append({"model": "MAP (pruned x_ref)",
                   **{k: _m[k] for k in ("acc", "nll", "ece", "brier", "mean_entropy")}})

for label, ck in runs.items():
    mean_p, draw_p = posterior_mean_probs(ck["samples"], X_test, N_PRED_DRAWS)
    clean_probs[label] = {"post_mean": mean_p, "post_draws": draw_p}
    _m = calibration_metrics(y_test, mean_p)
    pt = draw_p.gather(-1, y_test.long().view(1, -1, 1)
                       .expand(draw_p.shape[0], -1, 1)).squeeze(-1)
    clean_rows.append({"model": RUN_DISPLAY[label],
                       **{k: _m[k] for k in ("acc", "nll", "ece", "brier", "mean_entropy")},
                       "draw_std_P(true)": pt.std(0).mean().item()})

clean_df = pd.DataFrame(clean_rows).set_index("model")
print("\n=== Analysis A: clean MNIST (1k test subset) ===")
print(clean_df.to_string(float_format=lambda v: f"{v:.4f}"))
# clean_df.to_csv(SAVE_DIR / "tableA_clean_metrics.csv")


# %%
# --- FIGURE A1: reliability diagram ---
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
for name, d in clean_probs.items():
    disp = name if name == "MAP" else RUN_DISPLAY.get(name, name)
    m = calibration_metrics(y_test, d["post_mean"])
    if not m["bin_stats"]:
        continue
    bc, ba, _ = zip(*m["bin_stats"])
    ax.plot(bc, ba, marker="o", ms=5, label=f"{disp} (ECE={m['ece']:.3f})")
ax.set(xlabel="confidence", ylabel="empirical accuracy",
       title="Reliability -- LeNet5 / MNIST", xlim=(0, 1), ylim=(0, 1))
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "figA1_reliability.pdf", bbox_inches="tight")


# %%
# --- FIGURE A2: predictive entropy, correct vs wrong ---
fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4), squeeze=False)
for ax, (label, _) in zip(axes[0], runs.items()):
    mp = clean_probs[label]["post_mean"]
    ent = _entropy(mp)
    ok = mp.argmax(-1) == y_test
    ax.hist(ent[ok].numpy(), bins=40, histtype="step", lw=1.5, label="correct")
    ax.hist(ent[~ok].numpy(), bins=40, histtype="step", lw=1.5, ls="--", label="wrong")
    ax.set(xlabel="predictive entropy", ylabel="count", yscale="log",
           title=f"{RUN_DISPLAY[label]}: entropy by outcome")
    ax.legend()
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "figA2_entropy_split.pdf", bbox_inches="tight")


# %%
# ==========================================================================
# ANALYSIS B1 -- input-pixel sensitivity: does the net (MAP and posterior)
#                learn to ignore the dead border?
# ==========================================================================
# The conv1-weight level does NOT carry a clean border signal here: with a
# thin, irregular never-ink frame (~180 px) no 5x5 conv1 tap has a
# border-only footprint, so a per-tap "border vs ink" zero-rate split is
# vacuous (checked: correlation ~0). The honest question is at the INPUT
# level: the saliency |d max-logit / d input pixel|, averaged over the test
# set. A model that has learned MNIST's frame should put near-zero saliency
# on never-ink pixels. We compare the MAP point against the posterior mean
# saliency, and report the border/digit saliency ratio (lower = better) and
# the posterior's across-draw saliency std (epistemic sensitivity).
N_SAL_DRAWS = 100
never_ink_t = torch.tensor(never_ink)


def saliency_map(beta, X, n_img: int = 256) -> torch.Tensor:
    """[28,28] mean_i |d max_k logit_k(x_i) / d x_i| over n_img test images."""
    Xr = X[:n_img].clone().to(dtype=DTYPE, device=DEVICE).requires_grad_(True)
    logits = torch.func.functional_call(
        _bm.module, _param_dict_fn(beta.to(dtype=DTYPE, device=DEVICE)), (Xr,))
    g, = torch.autograd.grad(logits.max(-1).values.sum(), Xr)
    return g.abs().mean(0).squeeze(0).detach().cpu()


sal_map = saliency_map(X_REF, X_test)
b1_rows = [{
    "model": "MAP (pruned x_ref)",
    "saliency_border": sal_map[never_ink_t].mean().item(),
    "saliency_digit": sal_map[~never_ink_t].mean().item(),
    "border/digit ratio": (sal_map[never_ink_t].mean() / sal_map[~never_ink_t].mean()).item(),
}]
post_sal = {}
for label, ck in runs.items():
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(ck["samples"].shape[0], generator=g)[:N_SAL_DRAWS]
    sal = torch.stack([saliency_map(ck["samples"][i], X_test) for i in idx])  # [n,28,28]
    post_sal[label] = sal
    sm = sal.mean(0)
    b1_rows.append({
        "model": RUN_DISPLAY[label],
        "saliency_border": sm[never_ink_t].mean().item(),
        "saliency_digit": sm[~never_ink_t].mean().item(),
        "border/digit ratio": (sm[never_ink_t].mean() / sm[~never_ink_t].mean()).item(),
        "draw_std border": sal.std(0)[never_ink_t].mean().item(),
        "draw_std digit": sal.std(0)[~never_ink_t].mean().item(),
    })
b1_df = pd.DataFrame(b1_rows).set_index("model")
print("\n=== Analysis B1: input-pixel saliency, border vs digit ===")
print(b1_df.to_string(float_format=lambda v: f"{v:.4f}"))
# b1_df.to_csv(SAVE_DIR / "tableB1_saliency.csv")

fig, axes = plt.subplots(1, 1 + len(runs), figsize=(4.2 * (1 + len(runs)), 4), squeeze=False)
vmax = sal_map.max().item()
axes[0, 0].imshow(sal_map, cmap="magma", vmin=0, vmax=vmax)
axes[0, 0].contour(never_ink, levels=[0.5], colors="cyan", linewidths=0.8)
axes[0, 0].set_title("MAP saliency\n(cyan = never-ink)")
for ax, (label, _) in zip(axes[0, 1:], runs.items()):
    ax.imshow(post_sal[label].mean(0), cmap="magma", vmin=0, vmax=vmax)
    ax.contour(never_ink, levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_title(f"{RUN_DISPLAY[label]}\nposterior-mean saliency")
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "figB1_saliency.pdf", bbox_inches="tight")


# %%
# ==========================================================================
# ANALYSIS B2 -- localised-corruption robustness: border noise vs digit noise
# ==========================================================================
SEVERITIES = [1, 2, 3]
NOISE_SIGMA = {1: 0.10, 2: 0.25, 3: 0.50}   # std of additive noise, in de-normalised [0,1] units

border_mask_t = torch.tensor(never_ink)                     # [28,28] where noise "shouldn't matter"
digit_mask_t = ~border_mask_t                                # everywhere else


def corrupt_region(X, region: str, sev: int, seed: int = 0):
    """Add Gaussian noise to only the border or only the digit region, in
    de-normalised space, then re-normalise. region in {'border','digit'}."""
    g = torch.Generator().manual_seed(seed + sev + (0 if region == "border" else 100))
    img = X * MNIST_STD + MNIST_MEAN
    noise = NOISE_SIGMA[sev] * torch.randn(X.shape, generator=g)
    m = (border_mask_t if region == "border" else digit_mask_t).view(1, 1, 28, 28)
    img = (img + noise * m).clamp(0, 1)
    return (img - MNIST_MEAN) / MNIST_STD


b2_rows = []
for region in ("border", "digit"):
    for sev in SEVERITIES:
        Xc = corrupt_region(X_test, region, sev)
        pm = predict_probs(X_REF, Xc)
        mm = calibration_metrics(y_test, pm)
        b2_rows.append({"region": region, "severity": sev, "model": "MAP",
                        **{k: mm[k] for k in ("acc", "nll", "ece", "brier", "mean_entropy")}})
        for label, ck in runs.items():
            mean_p, _ = posterior_mean_probs(ck["samples"], Xc, N_PRED_DRAWS)
            mm = calibration_metrics(y_test, mean_p)
            b2_rows.append({"region": region, "severity": sev, "model": RUN_DISPLAY[label],
                            **{k: mm[k] for k in ("acc", "nll", "ece", "brier", "mean_entropy")}})
    print(f"  done: {region}")

b2_df = pd.DataFrame(b2_rows)
print("\n=== Analysis B2: localised corruption (border vs digit) ===")
print(b2_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# b2_df.to_csv(SAVE_DIR / "tableB2_localised_corruption.csv", index=False)

# --- FIGURE B2: acc + NLL vs severity, border vs digit, MAP vs posterior ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
for col, region in enumerate(("border", "digit")):
    for row, metric in enumerate(("acc", "nll")):
        ax = axes[row, col]
        sub = b2_df[b2_df["region"] == region]
        for model in sub["model"].unique():
            clean_v = (clean_df.loc[model, metric] if model in clean_df.index
                       else clean_df.iloc[0][metric])
            ys = [clean_v] + sub[sub["model"] == model].sort_values("severity")[metric].tolist()
            ax.plot([0] + SEVERITIES, ys, marker="o", label=model)
        ax.set(title=f"{region} noise -- {metric.upper()}",
               xlabel="severity (0 = clean)", ylabel=metric.upper())
        if row == 0 and col == 0:
            ax.legend(fontsize=9)
fig.suptitle("Localised corruption: border noise should barely move; digit noise should", y=1.01)
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "figB2_localised_degradation.pdf", bbox_inches="tight")

# The headline number: border-noise invariance gap. How much does severity-3
# border noise move each metric, MAP vs posterior? (Smaller = better.)
print("\n  Severity-3 border-noise sensitivity (Δ from clean; smaller is better):")
for model in b2_df["model"].unique():
    clean_acc = clean_df.loc[model, "acc"] if model in clean_df.index else clean_df.iloc[0]["acc"]
    clean_nll = clean_df.loc[model, "nll"] if model in clean_df.index else clean_df.iloc[0]["nll"]
    r = b2_df[(b2_df["model"] == model) & (b2_df["region"] == "border") & (b2_df["severity"] == 3)].iloc[0]
    print(f"    {model:20s}: Δacc={r['acc']-clean_acc:+.4f}  ΔNLL={r['nll']-clean_nll:+.4f}")


# %%
# ==========================================================================
# ANALYSIS C -- structured-sparsity ledger
# ==========================================================================
def sparsity_ledger(samples):
    s = samples.numpy()
    is_zero = np.abs(s) < ZERO_TOL
    return {"ever_nonzero": ~is_zero.all(0),
            "always_zero": is_zero.all(0),
            "always_nonzero": (~is_zero).all(0)}


ledgers = {label: sparsity_ledger(ck["samples"]) for label, ck in runs.items()}

# --- C1: per-layer survival ---
c1_rows = []
for label, L in ledgers.items():
    for lname in [r["layer"] for r in rows]:
        a, b = layer_slices[lname]
        sl = slice(a, b)
        c1_rows.append({
            "run": RUN_DISPLAY[label], "layer": lname, "n": b - a,
            "frac_ever_nonzero": L["ever_nonzero"][sl].mean(),
            "frac_always_nonzero": L["always_nonzero"][sl].mean(),
            "frac_always_zero": L["always_zero"][sl].mean(),
        })
c1_df = pd.DataFrame(c1_rows)
print("\n=== Analysis C1: per-layer weight survival ===")
print(c1_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# c1_df.to_csv(SAVE_DIR / "tableC1_layer_survival.csv", index=False)

# Static vs dynamic sparsity: per-draw sparsity (what fraction is zero in a
# GIVEN draw) vs the fraction of coords zero in EVERY draw. A big gap means
# the same ~91-93% sparsity is realised by different coordinates each draw
# (freeze/thaw churn) rather than a fixed pruned subset -- the two samplers
# differ sharply here.
print("\n  static vs dynamic sparsity:")
for label, ck in runs.items():
    s = ck["samples"].numpy()
    per_draw = (np.abs(s) < ZERO_TOL).mean(1)
    L = ledgers[label]
    print(f"    {RUN_DISPLAY[label]:16s}: per-draw sparsity {per_draw.mean():.3f} "
          f"(min {per_draw.min():.3f}, max {per_draw.max():.3f})  |  "
          f"zero in EVERY draw {L['always_zero'].mean():.3f}  |  "
          f"never zero {L['always_nonzero'].mean():.3f}  |  "
          f"churns (neither) {1 - L['always_zero'].mean() - L['always_nonzero'].mean():.3f}")

fig, ax = plt.subplots(figsize=(11, 4))
layers = [r["layer"] for r in rows]
x = np.arange(len(layers))
width = 0.8 / max(len(ledgers), 1)
for k, (label, _) in enumerate(ledgers.items()):
    sub = c1_df[c1_df["run"] == RUN_DISPLAY[label]].set_index("layer").loc[layers]
    ax.bar(x + k * width, sub["frac_always_nonzero"], width,
           label=f"{RUN_DISPLAY[label]} -- always active")
    ax.bar(x + k * width, sub["frac_ever_nonzero"] - sub["frac_always_nonzero"], width,
           bottom=sub["frac_always_nonzero"], alpha=0.4,
           label=f"{RUN_DISPLAY[label]} -- sometimes active")
ax.set(xticks=x + width * (len(ledgers) - 1) / 2, ylim=(0, 1),
       ylabel="fraction of layer params",
       title="Per-layer weight survival under the sticky-PDMP posterior")
ax.set_xticklabels(layers, rotation=60, ha="right", fontsize=8)
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "figC1_layer_survival.pdf", bbox_inches="tight")


# %%
# --- C2: conv-filter / (filter, in-channel) death with posterior prob 1 ---
c2_rows = []
for label, ck in runs.items():
    s = ck["samples"]
    for lname in CONV_LAYERS:
        a, b = layer_slices[lname]
        shp = dict(_ref_module.named_parameters())[lname].shape   # [out,in,kh,kw]
        w = s[:, a:b].reshape(s.shape[0], *shp)
        wz = (w.abs() < ZERO_TOL)
        filt_dead = wz.all(dim=(0, 2, 3, 4))
        slab_dead = wz.all(dim=(0, 3, 4))
        c2_rows.append({
            "run": RUN_DISPLAY[label], "layer": lname,
            "out_ch": shp[0], "in_ch": shp[1],
            "dead_filters": int(filt_dead.sum()), "dead_filter_frac": filt_dead.float().mean().item(),
            "dead_slabs": int(slab_dead.sum()), "dead_slab_frac": slab_dead.float().mean().item(),
        })
c2_df = pd.DataFrame(c2_rows)
print("\n=== Analysis C2: structured death (posterior prob 1) ===")
print(c2_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# c2_df.to_csv(SAVE_DIR / "tableC2_structured_death.csv", index=False)


# %%
# --- C3: accuracy vs sparsity, posterior vs MAP ---
c3_rows = [{
    "model": "MAP (pruned x_ref)",
    "accuracy": clean_df.loc["MAP (pruned x_ref)", "acc"],
    "weight_sparsity": float((np.abs(X_REF.numpy()) < ZERO_TOL).mean()),
}]
for label, ck in runs.items():
    s = ck["samples"].numpy()
    c3_rows.append({
        "model": RUN_DISPLAY[label],
        "accuracy": clean_df.loc[RUN_DISPLAY[label], "acc"],
        "weight_sparsity": float((np.abs(s) < ZERO_TOL).mean()),
    })
c3_df = pd.DataFrame(c3_rows).set_index("model")
print("\n=== Analysis C3: accuracy vs sparsity ===")
print(c3_df.to_string(float_format=lambda v: f"{v:.4f}"))
# c3_df.to_csv(SAVE_DIR / "tableC3_acc_vs_sparsity.csv")


# %%
# --- C4: per-layer posterior displacement from MAP (non-frozen coords) ---
c4_rows = []
for label, ck in runs.items():
    s = ck["samples"].numpy()
    L = ledgers[label]
    mean_disp = (np.abs(s - X_REF.numpy()) / prior_std).mean(0)
    for lname in [r["layer"] for r in rows]:
        a, b = layer_slices[lname]
        m = np.zeros(D, dtype=bool); m[a:b] = True
        m &= L["ever_nonzero"]
        if m.sum() == 0:
            continue
        c4_rows.append({
            "run": RUN_DISPLAY[label], "layer": lname, "n_moving": int(m.sum()),
            "mean_disp_priorstd": float(mean_disp[m].mean()),
            "median_disp_priorstd": float(np.median(mean_disp[m])),
            "p95_disp_priorstd": float(np.quantile(mean_disp[m], 0.95)),
        })
c4_df = pd.DataFrame(c4_rows)
print("\n=== Analysis C4: posterior displacement from MAP (moving coords) ===")
print(c4_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# c4_df.to_csv(SAVE_DIR / "tableC4_displacement.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4))
layers = [r["layer"] for r in rows]
x = np.arange(len(layers))
width = 0.8 / max(len(runs), 1)
for k, (label, _) in enumerate(runs.items()):
    sub = c4_df[c4_df["run"] == RUN_DISPLAY[label]].set_index("layer").reindex(layers)
    ax.bar(x + k * width, sub["mean_disp_priorstd"], width, label=RUN_DISPLAY[label])
ax.set(xticks=x + width * (len(runs) - 1) / 2, ylabel="mean |draw - MAP| (prior-std)",
       title="Posterior exploration around the MAP, per layer (moving coords only)")
ax.set_xticklabels(layers, rotation=60, ha="right", fontsize=8)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "figC4_displacement.pdf", bbox_inches="tight")


# %%
# ==========================================================================
# 6. One-screen summary for the paper's text
# ==========================================================================
print("\n" + "=" * 70)
print("SUMMARY -- numbers to quote (LeNet5 / MNIST)")
print("=" * 70)
mrow = clean_df.loc["MAP (pruned x_ref)"]
print(f"\nMAP (pruned x_ref): acc={mrow['acc']:.3f}  NLL={mrow['nll']:.3f}  "
      f"ECE={mrow['ece']:.3f}  Brier={mrow['brier']:.3f}  H={mrow['mean_entropy']:.3f}  "
      f"sparsity={c3_df.loc['MAP (pruned x_ref)', 'weight_sparsity']:.3f}")
for label, ck in runs.items():
    name = RUN_DISPLAY[label]
    L = ledgers[label]
    row = clean_df.loc[name]
    print(f"\n[{name}]")
    print(f"  clean:  acc={row['acc']:.3f}  NLL={row['nll']:.3f}  ECE={row['ece']:.3f}  "
          f"Brier={row['brier']:.3f}  H={row['mean_entropy']:.3f}")
    per_draw = (np.abs(ck["samples"].numpy()) < ZERO_TOL).mean(1).mean()
    churn = 1 - L["always_zero"].mean() - L["always_nonzero"].mean()
    print(f"  weight sparsity: per-draw {per_draw:.3f}  |  zero in EVERY draw "
          f"{L['always_zero'].mean():.3f} ({int(L['always_zero'].sum()):,}/{D:,})  |  "
          f"churning {churn:.3f}")
    br = b1_df.loc[name]
    print(f"  input saliency border/digit ratio: {br['border/digit ratio']:.3f}  "
          f"(MAP: {b1_df.loc['MAP (pruned x_ref)', 'border/digit ratio']:.3f})")
    dead_f = c2_df[c2_df['run'] == name]['dead_filters'].sum()
    tot_f = c2_df[c2_df['run'] == name]['out_ch'].sum()
    print(f"  conv filters dead w.p. 1: {dead_f} / {tot_f}")
    clean_acc = row["acc"]; clean_nll = row["nll"]
    rb = b2_df[(b2_df["model"] == name) & (b2_df["region"] == "border") & (b2_df["severity"] == 3)].iloc[0]
    rd = b2_df[(b2_df["model"] == name) & (b2_df["region"] == "digit") & (b2_df["severity"] == 3)].iloc[0]
    print(f"  sev-3 BORDER noise:  Δacc={rb['acc']-clean_acc:+.4f}  ΔNLL={rb['nll']-clean_nll:+.4f}  (want ~0)")
    print(f"  sev-3 DIGIT  noise:  Δacc={rd['acc']-clean_acc:+.4f}  ΔNLL={rd['nll']-clean_nll:+.4f}")
    mb = b2_df[(b2_df["model"] == "MAP") & (b2_df["region"] == "border") & (b2_df["severity"] == 3)].iloc[0]
    print(f"    (MAP sev-3 border: Δacc={mb['acc']-mrow['acc']:+.4f}  ΔNLL={mb['nll']-mrow['nll']:+.4f})")
