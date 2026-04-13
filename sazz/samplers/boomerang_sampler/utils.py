import numpy as np

def resample_pdmp_path(sampler, n_samples=4000, burnin_frac=0.1):
    t_sk = sampler.Time
    x_sk = sampler.Position
    v_sk = sampler.Velocity

    t_start = burnin_frac * t_sk[-1]
    t_grid = np.linspace(t_start, t_sk[-1], n_samples)
    
    idx = np.searchsorted(t_sk, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(t_sk) - 2)

    x_res = np.empty((n_samples, x_sk.shape[1]))
    for j, tg in enumerate(t_grid):
        i = idx[j]
        dt = tg - t_sk[i]
        x_res[j], _ = sampler.trajectory(dt, x_sk[i], v_sk[i])  # (t, x, v) not (x, v, t)

    return t_grid, x_res

def resample_sticky_pdmp_path(sampler, n_samples=4000, burnin_frac=0.1):
    t_sk = sampler.Time[:sampler.iteration]
    x_sk = sampler.Position[:sampler.iteration]
    v_sk = sampler.Velocity[:sampler.iteration]

    t_start = burnin_frac * t_sk[-1]
    t_grid = np.linspace(t_start, t_sk[-1], n_samples)
    idx = np.searchsorted(t_sk, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(t_sk) - 2)

    # precompute frozen mask from skeleton: frozen iff position is exactly 0
    fm_sk = np.abs(x_sk) < 1e-12  # (N, D) bool

    x_res = np.empty((n_samples, x_sk.shape[1]))
    for j, tg in enumerate(t_grid):
        i = idx[j]
        dt = tg - t_sk[i]
        x_t, _ = sampler.trajectory_sticky(dt, x_sk[i], v_sk[i])
        x_t[fm_sk[i]] = 0.0
        x_res[j] = x_t
    return t_grid, x_res