import numpy as np
import torch
from .sampling import resample_pdmp_path

def warmup(sampler, n_rounds=3, n_pilot=500,
                     target=None, tune_refresh=False):
    """
    Iterative warm-up: run short pilots, update (x_ref, Sigma_inv)
    from resampled path moments.
    """
    dtype = sampler.dtype
    device = sampler.device
    D = sampler.D

    # --- Round 0: initial reference ---
    if target is not None and target.x_ref is not None:
        sampler.preprocess(
            x_ref=target.x_ref.to(dtype=dtype, device=device),
            Sigma_inv=target.Sigma_inv.to(dtype=dtype, device=device),
        )
    else:
        # No reference available — start with prior-like defaults
        sampler.preprocess(
            x_ref=torch.zeros(D, dtype=dtype, device=device),
            Sigma_inv=torch.eye(D, dtype=dtype, device=device),
        )

    # --- Iterative refinement ---
    for i in range(n_rounds):
        result = sampler.sample(N=n_pilot, diagnostics=False)

        pos_np = result["positions"].cpu().numpy()
        vel_np = result["velocities"].cpu().numpy()
        tim_np = result["times"].cpu().numpy()
        x_ref_np = sampler.x_ref.cpu().numpy()

        samples = resample_pdmp_path(
            pos_np, vel_np, tim_np, x_ref_np,
            N_resample=n_pilot, burnin_frac=0.0,
        )

        x_ref_new = np.mean(samples, axis=0)
        var_diag = np.clip(np.var(samples, axis=0), 1e-8, None)
        Sigma_inv_new = np.diag(1.0 / var_diag)

        sampler.preprocess(
            x_ref=torch.tensor(x_ref_new, dtype=dtype, device=device),
            Sigma_inv=torch.tensor(Sigma_inv_new, dtype=dtype, device=device),
        )

    # if tune_refresh:
    #     _tune_refresh_rate(sampler, n_pilot)
        
        
# ===========================================================================
# PDMP path resampling (uniform-in-time)
# ===========================================================================

# def resample_pdmp_path(positions, velocities, times, x_ref, N_resample, burnin_frac=0.1):
#     """
#     Given skeleton (positions, velocities, times), resample N_resample
#     points uniformly in trajectory time using the Boomerang dynamics:
#         x(t) = x_ref + (x_k - x_ref)*cos(t - t_k) + v_k*sin(t - t_k)

#     All inputs are numpy arrays.
#     """
#     N, D = positions.shape
#     n_burn = int(burnin_frac * N)
#     pos = positions[n_burn:]
#     vel = velocities[n_burn:]
#     tim = times[n_burn:]

#     T_start = float(tim[0])
#     T_end = float(tim[-1])
#     if T_end <= T_start:
#         raise ValueError("Skeleton times are not increasing after burnin.")

#     sample_times = np.random.uniform(T_start, T_end, size=N_resample)
#     sample_times.sort()

#     # For each sample time, find the skeleton interval it falls in
#     # tim[idx-1] <= t < tim[idx]
#     indices = np.searchsorted(tim, sample_times, side="right") - 1
#     indices = np.clip(indices, 0, len(tim) - 2)

#     samples = np.empty((N_resample, D))
#     for j in range(N_resample):
#         k = indices[j]
#         dt = sample_times[j] - tim[k]
#         dx = pos[k] - x_ref
#         samples[j] = x_ref + dx * np.cos(dt) + vel[k] * np.sin(dt)

#     return samples

