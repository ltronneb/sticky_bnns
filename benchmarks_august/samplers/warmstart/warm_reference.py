import numpy as np
from sazz.samplers.boomerang_sampler.utils import resample_pdmp_path, resample_sticky_pdmp_path

def warmup_reference(sampler, n_rounds=3, n_pilot=500, x0=None, tune_refresh=False, sticky=False, target=None):
    """
    Iterative warm-up: run short pilots, update (x_ref, Sigma_ref) from samples.
    Works for all Boomerang variants (plain, PLI, sticky, sticky-PLI).
    """
    # First round: use target's reference if available
    if target is not None and target.x_ref is not None and target.Sigma_inv is not None:
        sampler.preprocess(method="manual", x_ref=target.x_ref, 
                           Sigma_inv=target.Sigma_inv)
    else:
        sampler.preprocess(x0=x0, method="diagonal")

    for i in range(n_rounds):
        sampler.reset(N=n_pilot)
        sampler.sample_auto(diagnostics=False)

        if sticky:
            _, samples = resample_sticky_pdmp_path(sampler, n_samples=n_pilot, burnin_frac=0.0)
        else:
            _, samples = resample_pdmp_path(sampler, n_samples=n_pilot, burnin_frac=0.0)
        x_ref = np.mean(samples, axis=0)
        var_diag = np.var(samples, axis=0)
        var_diag = np.clip(var_diag, 1e-8, None)
        Sigma_inv = np.diag(1.0 / var_diag)

        sampler.preprocess(method="manual", x_ref=x_ref, Sigma_inv=Sigma_inv)

    if tune_refresh:
        df = sampler.diagnostics_df
        n_bounce = df[(df['event_type'] == 'bounce') & (df['accepted'] == True)].shape[0]
        T_total = sampler.Time[sampler.iteration - 1]
        if T_total > 0:
            lambda_refl = n_bounce / T_total
            sampler.refresh_rate = min(
                (0.7812 / 0.2188) * lambda_refl,
                sampler.refresh_rate
            )

    