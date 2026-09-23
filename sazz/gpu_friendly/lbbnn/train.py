"""
Training driver for LBBNN fits, plus `run_lbbnn` -- the entry point the
benchmark scripts call.

`run_lbbnn` deliberately mirrors the signature and return contract of
toy_bnn_grid.py's `run_nuts` / `run_svi`:

    (flat_samples: np.ndarray [n_draws, D], elapsed_sec: float,
     gradient_evals: int)

so an LBBNN runner slots into SAMPLER_RUNNERS and feeds `save_run` with no
changes to the downstream analysis. The flat layout is [W0, b0, W1, b1, ...]
(+ trailing log_sigma when noise is learned) -- see
LBBNN.sample_posterior_flat.

GRADIENT-EVAL ACCOUNTING. The scripts here compare samplers under a common
`--grad-budget` currency, so what counts as one "gradient evaluation" matters.
`run_svi` counts one per optimizer step (one ELBO gradient over the full
dataset). LBBNN does `mc_samples` forward/backward passes' worth of work per
step, over a *minibatch* rather than the full dataset. The count reported here
is therefore

    steps * mc_samples * (batch_size / n_train)

i.e. full-dataset-gradient equivalents, which is the only version comparable
with the full-batch PDMP samplers. `raw_steps` is also returned in the info
dict for anyone who wants the un-normalized count.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from sazz.gpu_friendly.lbbnn.model import LBBNN, LBBNNConfig


def train_lbbnn(
    model: LBBNN,
    X: Tensor,
    y: Tensor,
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    verbose: bool = True,
    log_every: int = 25,
) -> dict[str, Any]:
    """Fit `model` on (X, y) by SVI on the ELBO. Returns a history dict.

    Minibatching is by a fresh permutation each epoch. The final partial batch
    is kept (the reference asserts the dataset divides evenly and would break
    here; model.sample_elbo is shape-independent, so there is no reason to
    drop data)."""
    cfg = model.cfg
    epochs = epochs if epochs is not None else cfg.epochs
    batch_size = batch_size if batch_size is not None else cfg.batch_size
    lr = lr if lr is not None else cfg.lr

    n_train = X.shape[0]
    batch_size = min(batch_size, n_train)
    num_batches = (n_train + batch_size - 1) // batch_size

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # NOTE on the logged "kl": it is the single-sample Monte Carlo estimate
    # of (log q - log p), not a true KL divergence, and it is routinely
    # NEGATIVE here. That is expected, not a bug -- these are log *densities*,
    # and the converged variational sigma (~0.01) is far tighter than the slab
    # prior's (~3), so log q(w) alone runs to +100 and up. Only its
    # expectation is non-negative. Read `mean_alpha` and `nll` to judge a fit;
    # the sign of `kl` carries no information.
    history = {"loss": [], "nll": [], "kl": [], "mean_alpha": []}
    steps = 0

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_train, generator=generator, device=X.device)
        epoch_loss = epoch_nll = epoch_kl = 0.0

        for b in range(num_batches):
            idx = perm[b * batch_size: (b + 1) * batch_size]
            xb, yb = X[idx], y[idx]

            model.zero_grad(set_to_none=True)
            loss, log_prior, log_q, nll = model.sample_elbo(xb, yb, num_batches)
            loss.backward()

            if cfg.cond_opt:
                # Reference COND_OPT: only let a weight's mean move where the
                # sampled mask included it.
                for layer in model.layers:
                    if layer.weight_mu.grad is not None and layer.gammas is not None:
                        layer.weight_mu.grad = layer.weight_mu.grad * layer.gammas.detach()

            optimizer.step()
            steps += 1

            epoch_loss += loss.item()
            epoch_nll += nll.item()
            epoch_kl += (log_q - log_prior).item() / num_batches

        mean_alpha = float(
            torch.cat([a.reshape(-1) for a in model.inclusion_probabilities()]).mean()
        )
        history["loss"].append(epoch_loss)
        history["nll"].append(epoch_nll)
        history["kl"].append(epoch_kl)
        history["mean_alpha"].append(mean_alpha)

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(
                f"      epoch {epoch:4d}/{epochs}  loss={epoch_loss:12.2f}  "
                f"nll={epoch_nll:12.2f}  kl={epoch_kl:10.2f}  "
                f"mean_alpha={mean_alpha:.4f}"
            )

    model.eval()
    history["steps"] = steps
    return history


def run_lbbnn(
    data: dict[str, Any],
    cfg: LBBNNConfig,
    seed: int,
    n_draws: int = 2_000,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> tuple[np.ndarray, float, int, dict[str, Any]]:
    """Fit an LBBNN on `data` and return posterior draws in this repo's flat
    convention.

    `data` needs "X_train"/"y_train" (the same dict the other runners get).
    Returns (samples, elapsed_sec, gradient_evals, info)."""
    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    X = data["X_train"].to(dtype=dtype, device=device)
    y = data["y_train"].to(dtype=dtype, device=device)
    if cfg.likelihood == "categorical":
        y = data["y_train"].to(device=device).long()

    model = LBBNN(cfg).to(dtype=dtype, device=device)

    t0 = time.perf_counter()
    history = train_lbbnn(model, X, y, generator=generator, verbose=verbose)
    elapsed = time.perf_counter() - t0

    n_train = X.shape[0]
    batch_size = min(cfg.batch_size, n_train)
    # Full-dataset-gradient equivalents -- see module docstring.
    gradient_evals = int(
        round(history["steps"] * cfg.mc_samples * batch_size / n_train)
    )

    samples = model.sample_posterior_flat(n_draws, generator=generator)

    # Variational-sigma collapse check. When the concrete temperature is too
    # high for the target, the ELBO can run away by shrinking q's sigma (see
    # uci_bnn_grid.py's LBBNN_TEMPER comment for the mechanism): the result
    # still "trains", the loss still improves, but the posterior degenerates
    # and the inclusion probabilities march back toward 1, so the run yields
    # no sparsity. That is silent unless someone reads the log closely, so
    # say it out loud.
    with torch.no_grad():
        q_sigma = torch.cat(
            [layer.weight.sigma.reshape(-1) for layer in model.layers]
        ).mean().item()
    if q_sigma < 1e-3:
        print(
            f"      WARNING: mean variational sigma has collapsed to "
            f"{q_sigma:.2e} (mean alpha {history['mean_alpha'][-1]:.3f}). "
            f"The posterior is degenerate and this run is unlikely to be a "
            f"usable sparsity baseline -- lower the concrete temperature "
            f"(cfg.temper/temper_prior, currently {cfg.temper}) and re-run."
        )

    info = {
        "history": history,
        "raw_steps": history["steps"],
        "mean_alpha": history["mean_alpha"][-1],
        "inclusion_probabilities": [
            a.cpu() for a in model.inclusion_probabilities()
        ],
        "model": model,
    }
    return samples.cpu().numpy().astype(np.float64), elapsed, gradient_evals, info
