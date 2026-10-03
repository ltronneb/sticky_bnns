"""LBBNN with a tuned, fixed noise variance on the UCI data (supplement).

Following the LBBNN authors, the noise variance is not learned but fixed at tau and
tuned. On standardized responses, fixing sigma^2 = tau is the same as multiplying the
KL term of the ELBO by tau with unit noise, so tuning tau tempers the posterior. The
inclusion indicators get a fixed Bernoulli(w) prior, the same prior as the sticky
samplers, and the slab and bias priors are those of uci_bnn.py.

For every (network, dataset, split), 20 % of the training data is held out for
validation. tau runs over the OLS residual variance of the standardized training
responses times TAU_SCALES, crossed with two initializations of the inclusion
logits. The pair with the lowest validation NLL (or RMSE, --select rmse) is refitted on
the full training split and evaluated on the test split. tau is never chosen on test
data. RMSE alone drives tau to the smallest value and overconfident intervals.

    python -m sazz.paper_ready.scripts.lbbnn_tempered --variant small --datasets boston --splits 0
    python -m sazz.paper_ready.scripts.lbbnn_tempered --variant medium --w 0.1 0.3
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..baselines import lbbnn
from ..common import save, seed_all
from ..data import uci_split
from .uci_bnn import DATASETS, VARIANTS

TAU_SCALES = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)
# inclusion logits ~ U(lo, hi). "high" starts the inclusion probabilities near 1 (the tuned
# UCI setting of uci_bnn.py), "polarized" is one of the original authors' initializations.
INITS = {"high": (3.5, 4.5), "polarized": (-1.5, 1.5)}
# The original authors train for 2000 epochs with their R package. This implementation
# needs longer, so we use the tuned UCI settings of uci_bnn.py (5e4 epochs, cosine decay).
EPOCHS, LR, TEMPER, VAL_FRAC = 50_000, 1e-2, 0.2, 0.2
DTYPE = torch.float64


def ols_residual_variance(X, y) -> float:
    A = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], 1)
    coef = torch.linalg.lstsq(A, y[:, None]).solution
    return float((y - (A @ coef)[:, 0]).var())


@torch.no_grad()
def predict_draws(draws, layers, X):
    from .analyse import ffn_predict
    return ffn_predict(draws.double(), layers, X.double())


def fit(data, layers, tau, init, w, seed, n_draws, epochs):
    draws, sec, evals, alpha = lbbnn(data, layers, "tanh", 1.0, 1.0, noise_std=math.sqrt(tau),
                                     batch_size=10_000, temper=TEMPER, lr=LR, epochs=epochs, cosine=True,
                                     learn_model_prior=False, prior_inclusion=w, lam_init=INITS[init],
                                     n_draws=n_draws, seed=seed, device="cpu", dtype=DTYPE)
    return draws, sec, evals, alpha


def run_one(args, variant, ds, split, w):
    sd = args.out / (variant if w is None else f"{variant}_w{w:g}") / ds / f"split_{split:02d}"
    if args.resume and (sd / "lbbnn.pt").exists():
        return
    data = uci_split(ds, split, dtype=DTYPE)
    layers = [data["X_train"].shape[1], *VARIANTS[variant]["hidden"], 1]

    # validation split carved from the training data
    g = np.random.default_rng(10_000 + split)
    perm = g.permutation(data["X_train"].shape[0])
    n_val = int(VAL_FRAC * len(perm))
    val, tr = perm[:n_val], perm[n_val:]
    inner = {"X_train": data["X_train"][tr], "y_train": data["y_train"][tr]}
    base_var = ols_residual_variance(inner["X_train"], inner["y_train"])

    grid = []
    for init in args.inits:
        for scale in args.scales:
            tau = base_var * scale
            draws, *_ = fit(inner, layers, tau, init, w, 42 + split, args.val_draws, args.epochs)
            from ..metrics import nll_mixture, rmse as rmse_fn
            f, yv = predict_draws(draws, layers, data["X_train"][val]), data["y_train"][val]
            sig = torch.full((f.shape[0],), math.sqrt(tau), dtype=f.dtype)
            r, n = rmse_fn(yv, f, data["y_std"]), nll_mixture(yv, f, sig, data["y_std"])
            grid.append(dict(init=init, scale=scale, tau=tau, val_rmse=r, val_nll=n))
            print(f"  [{variant} {ds} {split} w={w}] {init:<9} scale {scale:<5g} tau {tau:.5f}  "
                  f"val RMSE {r:.3f}  val NLL {n:.3f}")
    best = min(grid, key=lambda g_: g_[f"val_{args.select}"])

    seed_all(42 + split)
    draws, sec, evals, alpha = fit(data, layers, best["tau"], best["init"], w, 42 + split,
                                   args.n_draws, args.epochs)
    # fixed noise, stored as a constant log sigma column so the table code reads it like the others
    log_sigma = torch.full((draws.shape[0], 1), 0.5 * math.log(best["tau"]), dtype=draws.dtype)
    save(sd / "lbbnn.pt", samples=torch.cat([draws, log_sigma], 1), elapsed_sec=sec, grad_evals=evals,
         inclusion_probabilities=alpha, dataset=ds, split=split, layer_sizes=layers, activation="tanh",
         y_std=data["y_std"], x_ref=None, tau=best["tau"], init=best["init"], prior_inclusion=w,
         base_var=base_var, grid=grid, epochs=args.epochs, select=args.select)
    (sd / "grid.json").write_text(json.dumps(dict(best=best, base_var=base_var, grid=grid), indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["small", "medium"], default="small")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--w", nargs="+", type=float, default=[0.1],
                   help="prior inclusion probabilities, one run each")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--scales", nargs="+", type=float, default=list(TAU_SCALES),
                   help="tau grid as multiples of the OLS residual variance")
    p.add_argument("--inits", nargs="+", default=list(INITS), choices=list(INITS))
    p.add_argument("--select", choices=["nll", "rmse"], default="nll",
                   help="validation metric that picks tau and the initialization")
    p.add_argument("--val-draws", type=int, default=200, help="draws for the validation RMSE")
    p.add_argument("--n-draws", type=int, default=4000)
    p.add_argument("--out", type=Path, default=Path("results/lbbnn_tempered"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    for ds in args.datasets:
        for split in args.splits:
            for w in args.w:
                run_one(args, args.variant, ds, split, w)


if __name__ == "__main__":
    main()
