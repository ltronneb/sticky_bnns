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
        _tune_refresh_rate(sampler, n_pilot, sticky)



    # if tune_refresh:
    #     df = sampler.diagnostics_df
    #     n_bounce = df[(df['event_type'] == 'bounce') & (df['accepted'] == True)].shape[0]
    #     T_total = sampler.Time[sampler.iteration - 1]
    #     if T_total > 0:
    #         lambda_refl = n_bounce / T_total
    #         sampler.refresh_rate = min(
    #             (0.7812 / 0.2188) * lambda_refl,
    #             sampler.refresh_rate
    #         )

    
    
def _estimate_diffusivity(sampler):
    """Estimate σ²(ρ) from energy differences at refreshment times."""
    df = sampler.diagnostics_df
    
    # Find which skeleton indices are refreshments
    # by matching recorded refresh times to sampler.Time
    refresh_times = df.loc[df['event_type'] == 'refresh', 'time'].values
    
    if len(refresh_times) < 3:
        return 0.0
    
    n_stored = sampler.iteration  # number of skeleton points stored
    skel_times = sampler.Time[:n_stored]
    
    energies = []
    for rt in refresh_times:
        idx = np.argmin(np.abs(skel_times - rt))
        if np.abs(skel_times[idx] - rt) < 1e-10:
            energies.append(float(sampler.E(sampler.Position[idx])))
    
    if len(energies) < 3:
        return 0.0
    
    energies = np.array(energies)
    diffs_sq = np.diff(energies) ** 2
    rho = sampler.refresh_rate
    return 4.0 * rho * np.mean(diffs_sq)


def _tune_refresh_rate(sampler, n_pilot, sticky):
    """Try a few refresh rates, pick the one with highest diffusivity."""
    rho_current = sampler.refresh_rate
    candidates = [rho_current * f for f in [0.25, 0.5, 1.0, 2.0, 4.0]]
    
    best_rho = rho_current
    best_sigma2 = -1.0
    
    for rho in candidates:
        sampler.refresh_rate = rho
        sampler.reset(N=n_pilot)
        sampler.sample_auto(diagnostics=True)
        
        sigma2 = _estimate_diffusivity(sampler)
        if sigma2 > best_sigma2:
            best_sigma2 = sigma2
            best_rho = rho
    
    sampler.refresh_rate = best_rho