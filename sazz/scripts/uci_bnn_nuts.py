from __future__ import annotations

RUN = False   # <- intentional safety: flip to True to actually run.


def model_def():
    """The NumPyro model for a tanh BNN with iid Gaussian priors.

    Kept as a function so this file is importable as a reference even when
    NumPyro is not installed.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def bnn(X, y=None, layer_sizes=(13, 50, 50, 1), prior_std=1.0,
            noise_std=0.5):
        h = X
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            scale = prior_std / jnp.sqrt(n_in)
            W = numpyro.sample(
                f"W{i}",
                dist.Normal(jnp.zeros((n_in, n_out)), scale).to_event(2),
            )
            b = numpyro.sample(
                f"b{i}",
                dist.Normal(jnp.zeros(n_out), prior_std).to_event(1),
            )
            h = h @ W + b
            if i < len(layer_sizes) - 2:
                h = jnp.tanh(h)
        mean = h.squeeze(-1)
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("y", dist.Normal(mean, noise_std), obs=y)

    return bnn


def run_nuts_for(dataset_name: str, X_train, y_train,
                 layer_sizes, noise_std,
                 num_warmup: int = 1000, num_samples: int = 4000,
                 seed: int = 0):
    """Reference implementation. Only invoked when RUN = True."""
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    bnn = model_def()
    kernel = NUTS(bnn)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed),
             X=jnp.asarray(X_train), y=jnp.asarray(y_train),
             layer_sizes=tuple(layer_sizes),
             noise_std=noise_std)
    return mcmc.get_samples()


def main():
    if not RUN:
        print(__doc__)
        print("RUN is False — exiting without running NUTS.")
        return

    # The actual run path. Uses the same data/configs as uci_bnn.py.
    import pandas as pd
    from uci_bnn import load_datasets, configs_for  # type: ignore

    datasets = load_datasets()
    cfgs = configs_for(datasets)
    for name, data in datasets.items():
        cfg = cfgs[name]
        print(f"=== NUTS on {name} (D ~ {cfg.layer_sizes}) ===")
        X = data["X_train"].cpu().numpy()
        y = data["y_train"].cpu().numpy()
        run_nuts_for(name, X, y, cfg.layer_sizes, cfg.noise_std)


if __name__ == "__main__":
    main()
