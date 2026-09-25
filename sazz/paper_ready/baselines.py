"""Baselines for the regression BNNs, NUTS (NumPyro) and LBBNN.

Both return posterior draws in the same flat layout as the PDMPs,
[W0, b0, W1, b1, ...] (+ log_sigma when the noise is learned).
"""

import time
from typing import Optional

import numpy as np
import torch


def nuts(X, y, layer_sizes, activation: str, prior_std_weight: float, prior_std_bias: float,
         noise_std: Optional[float] = None, prior_sigma_scale: float = 1.0,
         x_init: Optional[torch.Tensor] = None, seed: int = 42, n_warmup: int = 1000,
         n_draws: int = 1000, n_chains: int = 4):
    """NUTS on the same fan-in scaled Gaussian prior as the PDMPs. noise_std=None
    learns sigma with a HalfNormal(prior_sigma_scale) prior. x_init (flat, as
    x_ref) initializes every chain. Returns (draws, elapsed_sec, gradient_evals)."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    act = jnp.tanh if activation == "tanh" else (lambda h: jnp.maximum(h, 0.0))
    X, y = jnp.asarray(X.cpu().numpy()), jnp.asarray(y.cpu().numpy())
    pairs = list(zip(layer_sizes[:-1], layer_sizes[1:]))

    def model(X, y):
        h = X
        for i, (n_in, n_out) in enumerate(pairs):
            W = numpyro.sample(f"W{i}", dist.Normal(jnp.zeros((n_out, n_in)),
                                                    prior_std_weight / jnp.sqrt(n_in)).to_event(2))
            b = numpyro.sample(f"b{i}", dist.Normal(jnp.zeros(n_out), prior_std_bias).to_event(1))
            h = h @ W.T + b
            if i < len(pairs) - 1:
                h = act(h)
        sigma = noise_std if noise_std is not None else numpyro.sample(
            "sigma", dist.HalfNormal(prior_sigma_scale))
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("y", dist.Normal(h.squeeze(-1), sigma), obs=y)

    init = None
    if x_init is not None:  # unconstrained space, so log_sigma for sigma
        x0, init, k = x_init.cpu().numpy(), {}, 0
        for i, (n_in, n_out) in enumerate(pairs):
            init[f"W{i}"] = x0[k:k + n_in * n_out].reshape(n_out, n_in)
            k += n_in * n_out
            init[f"b{i}"] = x0[k:k + n_out]
            k += n_out
        if noise_std is None:
            init["sigma"] = x0[k]
        init = {n: jnp.broadcast_to(jnp.asarray(a), (n_chains,) + np.shape(a)) for n, a in init.items()}

    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=n_warmup, num_samples=n_draws,
                num_chains=n_chains, progress_bar=True)
    t0 = time.perf_counter()
    mcmc.warmup(jax.random.PRNGKey(seed), X=X, y=y, init_params=init,
                extra_fields=("num_steps",), collect_warmup=True)
    evals = int(np.asarray(mcmc.get_extra_fields()["num_steps"]).sum())
    mcmc.run(mcmc.post_warmup_state.rng_key, X=X, y=y, extra_fields=("num_steps",))
    evals += int(np.asarray(mcmc.get_extra_fields()["num_steps"]).sum())
    elapsed = time.perf_counter() - t0

    post = mcmc.get_samples()
    flat = []
    for i in range(len(pairs)):
        W = np.asarray(post[f"W{i}"])
        flat += [W.reshape(W.shape[0], -1), np.asarray(post[f"b{i}"])]
    if noise_std is None:
        flat.append(np.log(np.asarray(post["sigma"]))[:, None])
    return torch.tensor(np.concatenate(flat, axis=1), dtype=torch.float64), elapsed, evals


def lbbnn(data: dict, layer_sizes, activation: str, prior_std_weight: float,
          prior_std_bias: float, noise_std: Optional[float] = None, prior_sigma_scale: float = 1.0,
          epochs: int = 15_000, temper: float = 0.2, batch_size: int = 10_000, lr: float = 1e-2,
          learn_model_prior: bool = True, n_draws: int = 4000, seed: int = 42,
          device="cpu", dtype=torch.float64):
    """Latent binary BNN by variational inference, see lbbnn.py. A fixed model
    prior is Beta-Binomial(1, 1). Returns (draws, elapsed_sec, gradient_evals,
    inclusion_probabilities)."""
    from .lbbnn import LBBNNConfig, run_lbbnn
    ab = (1.0, 1.1) if learn_model_prior else (1.0, 1.0)
    cfg = LBBNNConfig(layer_sizes=layer_sizes, activation=activation, noise_std=noise_std,
                      prior_sigma_scale=prior_sigma_scale, prior_std_weight=prior_std_weight,
                      prior_std_bias=prior_std_bias, temper=temper, prior_pa=ab, prior_pb=ab,
                      learn_model_prior=learn_model_prior, epochs=epochs,
                      batch_size=batch_size, lr=lr)
    return run_lbbnn(data, cfg, seed, n_draws=n_draws, device=device, dtype=dtype)
