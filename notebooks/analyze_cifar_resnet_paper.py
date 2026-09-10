"""
  A. PREDICTIVE QUALITY & CALIBRATION, and how it holds up under input shift.
     Standard ResNet reporting (accuracy / NLL / ECE / Brier / entropy) for
     the pruned MAP point vs the sticky-PDMP posterior average, on clean
     CIFAR-10 and under two cheap synthetic corruptions (Gaussian noise,
     brightness) at 3 severities. The headline claim: at equal accuracy the
     posterior is better calibrated and degrades more gracefully off-
     distribution than the single MAP point.

  B. STRUCTURED-SPARSITY LEDGER (the novel bit -- sparsity as a *posterior*
     statement, not one pruning mask). Per-stage surviving-weight fractions
     split into "ever non-zero" vs "non-zero in every draw", output-filter
     and (filter, in-channel) death counts with posterior probability 1,
     accuracy vs sparsity vs the MAP reference, and per-stage posterior
     displacement from MAP in prior-std units on the coordinates that are
     NOT frozen.

"""

from __future__ import annotations

# %%
# ==========================================================================
# 0. Config, imports, device
# ==========================================================================
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # any op MPS lacks -> CPU
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

if Path.cwd().name in ("scripts", "notebooks"):
    # allow running from the file's own dir
    while Path.cwd().name != "sticky_bnns" and Path.cwd() != Path.cwd().parent:
        os.chdir("..")
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

# Prior hyper-params the run used (fast_cheap_cifar_resnet.py).
PRIOR_STD_W = 2.0
PRIOR_STD_B = 2.0
PRIOR_STD_BN_W = 1.0
FAN_IN_SCALING = True
BASE_SEED = 42          # fast_cifar_resnet.BASE_SEED -- same test split as the run

# How many posterior draws to push through the network for the predictive
# metrics. 200 is plenty for stable ECE/NLL on a 1k test set; bump for the
# final paper numbers. Forward passes dominate runtime: on MPS one draw over
# the 1k test set is ~0.13 s, so the whole of Analysis A (clean + 2
# corruptions x 3 severities = 7 forward-pass sets) is ~3 min at 200 draws.
N_PRED_DRAWS = 200
N_TEST = 1000
ZERO_TOL = 1e-8         # |w| < ZERO_TOL counts as an exact zero (frozen)

SAVE_DIR = Path("results/paper/cifar/figs")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)   # uncomment when you start saving

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
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

# Guard: the MAP file MUST be the exact cold-start point of every run, or
# the MAP-vs-posterior comparison is comparing against the wrong reference
# (and the BN running stats in MSD would be from a different base model).
# grid_sticky_zigzag.pt stores its own x_ref -- it must match this file's.
for label, ck in runs.items():
    dmax = (ck["x_ref"].to(DTYPE) - X_REF).abs().max().item()
    cs_ok = torch.equal(ck["cold_start_mask"].bool(),
                        (X_REF == 0) if map_ck.get("cold_start_mask") is None
                        else map_ck["cold_start_mask"].bool())
    assert dmax < 1e-5 and cs_ok, (
        f"[{label}] MAP MISMATCH: max|x_ref - MAP.x_ref| = {dmax:.3e}, "
        f"cold_start_mask match = {cs_ok}. MAP_REF_PATH ({MAP_REF_PATH.name}) is "
        f"not the checkpoint this run cold-started from. Rsync the right one "
        f"(look for a *_v2_pruned_refit.pt with matching sparsity "
        f"{ck['cold_start_mask'].float().mean():.4f})."
    )
    print(f"  [{label}] matches run x_ref (max|Δ|={dmax:.1e}) and cold_start_mask ✓")


# %%
# ==========================================================================
# 2. Layer map -- unflatten coords -> (layer, kind), build the priors
# ==========================================================================
from sazz.gpu_friendly.models.neural_networks import ResNet20
from sazz.gpu_friendly.models.priors import (
    _is_batchnorm_weight, build_can_freeze_mask_resnet,
    build_fan_in_prior_precision_resnet,
)

_ref_module = ResNet20(activation="relu")
assert sum(p.numel() for p in _ref_module.parameters()) == D


def _kind(module, name, p):
    if _is_batchnorm_weight(module, name):
        return "bn_weight"
    if name.endswith(".bias") and p.dim() == 1:
        sub = name[: -len(".bias")]
        submod = module.get_submodule(sub) if sub else module
        return "bn_bias" if isinstance(submod, torch.nn.modules.batchnorm._BatchNorm) else "fc"
    if name.startswith("fc."):
        return "fc"
    return "conv"


def _stage_of(layer_name: str) -> str:
    if layer_name.startswith("stem"):
        return "stem"
    if layer_name.startswith("stage1"):
        return "stage1"
    if layer_name.startswith("stage2"):
        return "stage2"
    if layer_name.startswith("stage3"):
        return "stage3"
    if layer_name.startswith("fc"):
        return "fc"
    return "other"


rows, idx = [], 0
layer_slices = {}   # layer name -> (start, stop) coord range
for name, p in _ref_module.named_parameters():
    n = p.numel()
    layer_slices[name] = (idx, idx + n)
    rows.append({
        "layer": name, "kind": _kind(_ref_module, name, p),
        "stage": _stage_of(name), "shape": tuple(p.shape),
        "start": idx, "stop": idx + n,
    })
    idx += n
layer_df = pd.DataFrame(rows)
CONV_LAYERS = layer_df.loc[layer_df["kind"] == "conv", "layer"].tolist()

# coord -> stage / kind, as int/str arrays over all D
coord_stage = np.empty(D, dtype=object)
coord_kind = np.empty(D, dtype=object)
for r in rows:
    coord_stage[r["start"]:r["stop"]] = r["stage"]
    coord_kind[r["start"]:r["stop"]] = r["kind"]

can_freeze = build_can_freeze_mask_resnet(_ref_module).cpu().numpy()   # bool[D]
prior_prec = build_fan_in_prior_precision_resnet(
    _ref_module, PRIOR_STD_W, PRIOR_STD_B, PRIOR_STD_BN_W,
    FAN_IN_SCALING, dtype=torch.float32, device="cpu",
)
prior_std = prior_prec.clamp(min=1e-12).rsqrt().cpu().numpy()          # float[D]

print(layer_df.groupby("stage")["shape"].count().rename("n_layers"))
print("\ncoords by kind:", pd.Series(coord_kind).value_counts().to_dict())
print("freezable coords:", int(can_freeze.sum()), "/", D)


# %%
# ==========================================================================
# 3. Data -- same seeded 1k CIFAR-10 test subset the run evaluated on
# ==========================================================================
from sazz.gpu_friendly.scripts.fast_cifar_resnet import load_cifar10_subset

_data = load_cifar10_subset(50_000, N_TEST, BASE_SEED, Path("datasets"),
                            dtype=DTYPE, device="cpu")
X_test = _data["X_test"]          # [N,3,32,32] normalised, on CPU
y_test = _data["y_test"]
CIFAR10_STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
print("X_test:", tuple(X_test.shape), " class balance:",
      torch.bincount(y_test, minlength=10).tolist())


# %%
# ==========================================================================
# 4. Predictive engine -- push a beta vector through ResNet-20
# ==========================================================================
# One module in eval() (frozen BN running stats from MSD), reused for every
# beta via functional_call. This matches the sampler's forward exactly.
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


@torch.no_grad()
def predict_probs(beta: torch.Tensor, X: torch.Tensor, batch: int = 256) -> torch.Tensor:
    """[N, 10] softmax probs for one parameter vector beta on inputs X."""
    beta = beta.to(dtype=DTYPE, device=DEVICE)
    out = []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch].to(dtype=DTYPE, device=DEVICE)
        logits = torch.func.functional_call(_bm.module, _param_dict_fn(beta), (xb,))
        out.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def posterior_mean_probs(samples: torch.Tensor, X: torch.Tensor,
                         n_draws: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (mean_probs [N,10], stacked_draw_probs [n_draws,N,10])."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(samples.shape[0], generator=g)[:min(n_draws, samples.shape[0])]
    dp = torch.stack([predict_probs(samples[i], X) for i in idx])
    return dp.mean(0), dp


# %%
# ==========================================================================
# 5. Metrics -- accuracy / NLL / ECE / Brier / entropy
# ==========================================================================
def _entropy(probs, eps=1e-12):
    return -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(-1)


def calibration_metrics(y_true: torch.Tensor, probs: torch.Tensor, n_bins: int = 15) -> dict:
    y_true = y_true.long()
    conf, pred = probs.max(-1)
    correct = (pred == y_true).float()
    acc = correct.mean().item()
    p_true = probs.gather(-1, y_true.unsqueeze(-1)).squeeze(-1)
    nll = -p_true.clamp_min(1e-12).log().mean().item()
    onehot = F.one_hot(y_true, probs.shape[-1]).float()
    brier = ((probs - onehot) ** 2).sum(-1).mean().item()
    ent = _entropy(probs).mean().item()

    # ECE (equal-width confidence bins)
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            bin_acc = correct[m].mean().item()
            bin_conf = conf[m].mean().item()
            w = m.float().mean().item()
            ece += w * abs(bin_acc - bin_conf)
            bin_stats.append((bin_conf, bin_acc, w))
    return {"acc": acc, "nll": nll, "ece": ece, "brier": brier,
            "mean_entropy": ent, "bin_stats": bin_stats}


# %%
# ==========================================================================
# ANALYSIS A -- predictive quality & calibration on CLEAN CIFAR-10
# ==========================================================================
clean_probs = {}        # label -> {"map":..., "post_mean":..., "post_draws":...}
clean_rows = []

# MAP reference row
_p_map = predict_probs(X_REF, X_test)
clean_probs["MAP"] = {"post_mean": _p_map}
_m = calibration_metrics(y_test, _p_map)
clean_rows.append({"model": "MAP (pruned x_ref)", **{k: _m[k] for k in
                   ("acc", "nll", "ece", "brier", "mean_entropy")}})

for label, ck in runs.items():
    mean_p, draw_p = posterior_mean_probs(ck["samples"], X_test, N_PRED_DRAWS)
    clean_probs[label] = {"post_mean": mean_p, "post_draws": draw_p}
    _m = calibration_metrics(y_test, mean_p)
    # draw-to-draw disagreement on P(true class): how much the posterior
    # spreads, i.e. is it actually a distribution or 1000 copies of x_ref
    pt = draw_p.gather(-1, y_test.long().view(1, -1, 1)
                       .expand(draw_p.shape[0], -1, 1)).squeeze(-1)
    disagree = pt.std(0).mean().item()
    clean_rows.append({"model": RUN_DISPLAY[label],
                       **{k: _m[k] for k in ("acc", "nll", "ece", "brier", "mean_entropy")},
                       "draw_std_P(true)": disagree})

clean_df = pd.DataFrame(clean_rows).set_index("model")
print("\n=== Analysis A: clean CIFAR-10 (1k test subset) ===")
print(clean_df.to_string(float_format=lambda v: f"{v:.4f}"))

# --- TABLE 1 (paper): clean predictive metrics ---
# clean_df.to_csv(SAVE_DIR / "tableA1_clean_metrics.csv")
# clean_df.to_latex(SAVE_DIR / "tableA1_clean_metrics.tex", float_format="%.4f")


# %%
# --- FIGURE A1: reliability diagram, MAP vs posterior ---
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
for name, d in clean_probs.items():
    disp = name if name in ("MAP",) else RUN_DISPLAY.get(name, name)
    m = calibration_metrics(y_test, d["post_mean"])
    if not m["bin_stats"]:
        continue
    bc, ba, _ = zip(*m["bin_stats"])
    ax.plot(bc, ba, marker="o", ms=5, label=f"{disp} (ECE={m['ece']:.3f})")
ax.set(xlabel="confidence", ylabel="empirical accuracy",
       title="Reliability -- ResNet-20 / CIFAR-10", xlim=(0, 1), ylim=(0, 1))
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "figA1_reliability.pdf", bbox_inches="tight")


# %%
# --- FIGURE A2: predictive entropy, correct vs wrong (posterior) ---
fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4), squeeze=False)
for ax, (label, _) in zip(axes[0], runs.items()):
    mp = clean_probs[label]["post_mean"]
    ent = _entropy(mp)
    pred = mp.argmax(-1)
    ok = pred == y_test
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
# ANALYSIS A (cont.) -- degradation under input shift
# ==========================================================================
# Two cheap synthetic corruptions applied to the *normalised* test tensors,
# 3 severities each. Not a substitute for CIFAR-10-C, but enough to show
# whether the posterior's NLL / ECE degrade more slowly than the MAP's.
SEVERITIES = [1, 2, 3]
GAUSS_SIGMA = {1: 0.08, 2: 0.16, 3: 0.30}      # additive noise std, in normalised units
BRIGHT_DELTA = {1: 0.5, 2: 1.0, 3: 1.6}         # additive shift on the de-normalised [0,1] image


def corrupt(X: torch.Tensor, kind: str, sev: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + sev)
    if kind == "gaussian":
        return X + GAUSS_SIGMA[sev] * torch.randn(X.shape, generator=g)
    if kind == "brightness":
        img = (X * CIFAR10_STD + CIFAR10_MEAN)          # -> ~[0,1]
        img = (img + BRIGHT_DELTA[sev] * 0.1).clamp(0, 1)
        return (img - CIFAR10_MEAN) / CIFAR10_STD
    raise ValueError(kind)


shift_rows = []
for kind in ("gaussian", "brightness"):
    for sev in SEVERITIES:
        Xc = corrupt(X_test, kind, sev)
        # MAP
        pm = predict_probs(X_REF, Xc)
        mm = calibration_metrics(y_test, pm)
        shift_rows.append({"corruption": kind, "severity": sev, "model": "MAP",
                           **{k: mm[k] for k in ("acc", "nll", "ece", "brier", "mean_entropy")}})
        # posteriors
        for label, ck in runs.items():
            mean_p, _ = posterior_mean_probs(ck["samples"], Xc, N_PRED_DRAWS)
            mm = calibration_metrics(y_test, mean_p)
            shift_rows.append({"corruption": kind, "severity": sev,
                               "model": RUN_DISPLAY[label],
                               **{k: mm[k] for k in ("acc", "nll", "ece", "brier", "mean_entropy")}})
    print(f"  done: {kind}")

shift_df = pd.DataFrame(shift_rows)
print("\n=== Analysis A: under input shift ===")
print(shift_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# --- TABLE A2 (paper): shift metrics ---
# shift_df.to_csv(SAVE_DIR / "tableA2_shift_metrics.csv", index=False)


# %%
# --- FIGURE A3: NLL and ECE vs severity, MAP vs posterior ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
for col, kind in enumerate(("gaussian", "brightness")):
    for row, metric in enumerate(("nll", "ece")):
        ax = axes[row, col]
        sub = shift_df[shift_df["corruption"] == kind]
        # include clean (severity 0) as an anchor point
        for model in sub["model"].unique():
            xs = [0] + SEVERITIES
            clean_v = clean_df.loc[model, metric] if model in clean_df.index else \
                clean_df.loc[clean_df.index[0], metric]
            ys = [clean_v] + sub[sub["model"] == model].sort_values("severity")[metric].tolist()
            ax.plot(xs, ys, marker="o", label=model)
        ax.set(title=f"{kind} -- {metric.upper()}", xlabel="severity (0 = clean)",
               ylabel=metric.upper())
        if row == 0 and col == 0:
            ax.legend(fontsize=9)
fig.suptitle("Degradation under input shift: MAP vs sticky-PDMP posterior", y=1.01)
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "figA3_shift_degradation.pdf", bbox_inches="tight")


# %%
# ==========================================================================
# ANALYSIS B -- structured-sparsity ledger
# ==========================================================================
# All from the samples tensor + can_freeze + prior_std. No forward passes.
def sparsity_ledger(samples: torch.Tensor) -> dict:
    s = samples.numpy()
    is_zero = np.abs(s) < ZERO_TOL              # [n, D]
    ever_nonzero = ~is_zero.all(0)              # [D]  moved off zero at least once
    always_zero = is_zero.all(0)               # [D]  zero in every draw
    always_nonzero = (~is_zero).all(0)         # [D]  never zero in any draw
    return {"ever_nonzero": ever_nonzero, "always_zero": always_zero,
            "always_nonzero": always_nonzero}


ledgers = {label: sparsity_ledger(ck["samples"]) for label, ck in runs.items()}

# --- B1: per-stage surviving fraction ---
b1_rows = []
for label, L in ledgers.items():
    for stage in ("stem", "stage1", "stage2", "stage3", "fc"):
        m = coord_stage == stage
        n = int(m.sum())
        b1_rows.append({
            "run": RUN_DISPLAY[label], "stage": stage, "n_coords": n,
            "frac_ever_nonzero": L["ever_nonzero"][m].mean(),
            "frac_always_nonzero": L["always_nonzero"][m].mean(),
            "frac_always_zero": L["always_zero"][m].mean(),
        })
b1_df = pd.DataFrame(b1_rows)
print("\n=== Analysis B1: per-stage weight survival ===")
print(b1_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# b1_df.to_csv(SAVE_DIR / "tableB1_stage_survival.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4))
stages = ["stem", "stage1", "stage2", "stage3", "fc"]
x = np.arange(len(stages))
width = 0.8 / max(len(ledgers), 1)
for k, (label, L) in enumerate(ledgers.items()):
    sub = b1_df[b1_df["run"] == RUN_DISPLAY[label]].set_index("stage").loc[stages]
    ax.bar(x + k * width, sub["frac_always_nonzero"], width,
           label=f"{RUN_DISPLAY[label]} -- always active")
    ax.bar(x + k * width, sub["frac_ever_nonzero"] - sub["frac_always_nonzero"], width,
           bottom=sub["frac_always_nonzero"], alpha=0.4,
           label=f"{RUN_DISPLAY[label]} -- sometimes active")
ax.set(xticks=x + width * (len(ledgers) - 1) / 2, ylim=(0, 1),
       ylabel="fraction of stage parameters",
       title="Per-stage weight survival under the sticky-PDMP posterior")
ax.set_xticklabels(stages)
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "figB1_stage_survival.pdf", bbox_inches="tight")


# %%
# --- B2: structured death -- output filters and (filter, in-channel) slabs
#         that are zero with posterior probability 1 (every draw) ---
b2_rows = []
for label, ck in runs.items():
    s = ck["samples"]
    for lname in CONV_LAYERS:
        a, b = layer_slices[lname]
        shp = dict(_ref_module.named_parameters())[lname].shape   # [out, in, kh, kw]
        w = s[:, a:b].reshape(s.shape[0], *shp)                   # [n, out, in, kh, kw]
        wz = (w.abs() < ZERO_TOL)
        filt_dead = wz.all(dim=(0, 2, 3, 4))                      # [out]  filter zero in every draw
        slab_dead = wz.all(dim=(0, 3, 4))                         # [out, in]
        b2_rows.append({
            "run": RUN_DISPLAY[label], "layer": lname, "stage": _stage_of(lname),
            "out_ch": shp[0], "in_ch": shp[1],
            "dead_filters": int(filt_dead.sum()),
            "dead_filter_frac": filt_dead.float().mean().item(),
            "dead_slabs": int(slab_dead.sum()),
            "dead_slab_frac": slab_dead.float().mean().item(),
        })
b2_df = pd.DataFrame(b2_rows)
print("\n=== Analysis B2: structured death (posterior prob 1) ===")
print(b2_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
b2_stage = (b2_df.groupby(["run", "stage"])
            [["dead_filters", "out_ch", "dead_slabs"]].sum()
            .assign(dead_filter_frac=lambda d: d.dead_filters / d.out_ch))
print("\n  rolled up by stage:")
print(b2_stage.to_string(float_format=lambda v: f"{v:.4f}"))
# b2_df.to_csv(SAVE_DIR / "tableB2_structured_death.csv", index=False)

fig, ax = plt.subplots(figsize=(13, 4))
for k, (label, _) in enumerate(runs.items()):
    sub = b2_df[b2_df["run"] == RUN_DISPLAY[label]]
    ax.bar(np.arange(len(sub)) + k * 0.4, sub["dead_filter_frac"], 0.4,
           label=RUN_DISPLAY[label])
ax.set_xticks(np.arange(len(b2_df[b2_df["run"] == RUN_DISPLAY[list(runs)[0]]])))
ax.set_xticklabels(CONV_LAYERS, rotation=75, ha="right", fontsize=7)
ax.set(ylabel="fraction of output filters dead in every draw",
       title="Output-filter death by conv layer (posterior probability 1)")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "figB2_filter_death.pdf", bbox_inches="tight")


# %%
# --- B3: accuracy vs sparsity, posterior vs MAP ---
b3_rows = [{
    "model": "MAP (pruned x_ref)",
    "accuracy": clean_df.loc["MAP (pruned x_ref)", "acc"],
    "weight_sparsity": float((np.abs(X_REF.numpy()) < ZERO_TOL).mean()),
    "freezable_sparsity": float((np.abs(X_REF.numpy()[can_freeze]) < ZERO_TOL).mean()),
}]
for label, ck in runs.items():
    s = ck["samples"].numpy()
    b3_rows.append({
        "model": RUN_DISPLAY[label],
        "accuracy": clean_df.loc[RUN_DISPLAY[label], "acc"],
        "weight_sparsity": float((np.abs(s) < ZERO_TOL).mean()),
        "freezable_sparsity": float((np.abs(s[:, can_freeze]) < ZERO_TOL).mean()),
    })
b3_df = pd.DataFrame(b3_rows).set_index("model")
print("\n=== Analysis B3: accuracy vs sparsity ===")
print(b3_df.to_string(float_format=lambda v: f"{v:.4f}"))
# b3_df.to_csv(SAVE_DIR / "tableB3_acc_vs_sparsity.csv")


# %%
# --- B4: where does the posterior actually move? per-stage displacement
#         from MAP, in prior-std units, on the NON-frozen coordinates only ---
b4_rows = []
for label, ck in runs.items():
    s = ck["samples"].numpy()
    L = ledgers[label]
    disp = np.abs(s - X_REF.numpy()) / prior_std          # [n, D]
    mean_disp = disp.mean(0)                               # [D]
    for stage in ("stem", "stage1", "stage2", "stage3", "fc"):
        m = (coord_stage == stage) & L["ever_nonzero"]     # moving coords in this stage
        if m.sum() == 0:
            continue
        b4_rows.append({
            "run": RUN_DISPLAY[label], "stage": stage,
            "n_moving": int(m.sum()),
            "mean_disp_priorstd": float(mean_disp[m].mean()),
            "median_disp_priorstd": float(np.median(mean_disp[m])),
            "p95_disp_priorstd": float(np.quantile(mean_disp[m], 0.95)),
        })
b4_df = pd.DataFrame(b4_rows)
print("\n=== Analysis B4: posterior displacement from MAP (non-frozen coords) ===")
print(b4_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# b4_df.to_csv(SAVE_DIR / "tableB4_displacement.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
stages = ["stem", "stage1", "stage2", "stage3", "fc"]
x = np.arange(len(stages))
width = 0.8 / max(len(runs), 1)
for k, (label, _) in enumerate(runs.items()):
    sub = b4_df[b4_df["run"] == RUN_DISPLAY[label]].set_index("stage").reindex(stages)
    ax.bar(x + k * width, sub["mean_disp_priorstd"], width, label=RUN_DISPLAY[label])
ax.set(xticks=x + width * (len(runs) - 1) / 2, xticklabels=stages,
       ylabel="mean |draw - MAP|  (prior-std units)",
       title="Posterior exploration around the MAP, per stage (moving coords only)")
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
# fig.savefig(SAVE_DIR / "figB4_displacement.pdf", bbox_inches="tight")


# %%
# ==========================================================================
# 6. One-screen summary for the paper's text
# ==========================================================================
print("\n" + "=" * 70)
print("SUMMARY -- numbers to quote")
print("=" * 70)
for label, ck in runs.items():
    L = ledgers[label]
    name = RUN_DISPLAY[label]
    row = clean_df.loc[name]
    print(f"\n[{name}]")
    print(f"  clean:  acc={row['acc']:.3f}  NLL={row['nll']:.3f}  "
          f"ECE={row['ece']:.3f}  Brier={row['brier']:.3f}  "
          f"mean entropy={row['mean_entropy']:.3f}")
    mrow = clean_df.loc["MAP (pruned x_ref)"]
    print(f"  vs MAP: acc={mrow['acc']:.3f}  NLL={mrow['nll']:.3f}  "
          f"ECE={mrow['ece']:.3f}  Brier={mrow['brier']:.3f}  "
          f"mean entropy={mrow['mean_entropy']:.3f}")
    print(f"  weight sparsity: {b3_df.loc[name, 'weight_sparsity']:.3f} of D  "
          f"({b3_df.loc[name, 'freezable_sparsity']:.3f} of freezable)")
    print(f"  coords zero in EVERY draw: {int(L['always_zero'].sum()):,} / {D:,}")
    tot_filt = b2_df[b2_df['run'] == name]['out_ch'].sum()
    dead_filt = b2_df[b2_df['run'] == name]['dead_filters'].sum()
    print(f"  conv output filters dead w.p. 1: {dead_filt} / {tot_filt}")
    # biggest NLL gap under shift
    g = shift_df[shift_df["model"] == name].merge(
        shift_df[shift_df["model"] == "MAP"], on=["corruption", "severity"],
        suffixes=("_post", "_map"))
    g["nll_gap"] = g["nll_map"] - g["nll_post"]      # positive = posterior better
    worst = g.loc[g["severity"] == 3]
    print("  NLL improvement over MAP at severity 3:")
    for _, r in worst.iterrows():
        print(f"    {r['corruption']:11s}: MAP {r['nll_map']:.3f} -> post {r['nll_post']:.3f}  "
              f"(Δ={r['nll_gap']:+.3f})")
