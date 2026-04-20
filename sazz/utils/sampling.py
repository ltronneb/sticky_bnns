import numpy as np

# ===========================================================================
# PDMP path resampling (uniform-in-time)
# ===========================================================================

def resample_pdmp_path(positions, velocities, times, x_ref, N_resample, burnin_frac=0.1):
    """
    Given skeleton (positions, velocities, times), resample N_resample
    points uniformly in trajectory time using the Boomerang dynamics:
        x(t) = x_ref + (x_k - x_ref)*cos(t - t_k) + v_k*sin(t - t_k)

    All inputs are numpy arrays.
    """
    N, D = positions.shape
    n_burn = int(burnin_frac * N)
    pos = positions[n_burn:]
    vel = velocities[n_burn:]
    tim = times[n_burn:]

    T_start = float(tim[0])
    T_end = float(tim[-1])
    if T_end <= T_start:
        raise ValueError("Skeleton times are not increasing after burnin.")

    sample_times = np.random.uniform(T_start, T_end, size=N_resample)
    sample_times.sort()

    # For each sample time, find the skeleton interval it falls in
    # tim[idx-1] <= t < tim[idx]
    indices = np.searchsorted(tim, sample_times, side="right") - 1
    indices = np.clip(indices, 0, len(tim) - 2)

    samples = np.empty((N_resample, D))
    for j in range(N_resample):
        k = indices[j]
        dt = sample_times[j] - tim[k]
        dx = pos[k] - x_ref
        samples[j] = x_ref + dx * np.cos(dt) + vel[k] * np.sin(dt)

    return samples


def resample_pdmp_path_sticky(positions, velocities, times, x_ref,
                               N_resample, burnin_frac=0.1, zero_tol=1e-12):
    N, D = positions.shape
    n_burn = int(burnin_frac * N)
    pos = positions[n_burn:]
    vel = velocities[n_burn:]
    tim = times[n_burn:]

    T_start = float(tim[0])
    T_end = float(tim[-1])
    if T_end <= T_start:
        raise ValueError("Skeleton times are not increasing after burnin.")

    sample_times = np.random.uniform(T_start, T_end, size=N_resample)
    sample_times.sort()

    indices = np.searchsorted(tim, sample_times, side="right") - 1
    indices = np.clip(indices, 0, len(tim) - 2)

    samples = np.empty((N_resample, D))
    for j in range(N_resample):
        k = indices[j]
        dt = sample_times[j] - tim[k]
        dx = pos[k] - x_ref
        # Standard Boomerang interpolation
        samples[j] = x_ref + dx * np.cos(dt) + vel[k] * np.sin(dt)
        # Override frozen coordinates: if x_k[i] = 0 and v_k[i] = 0,
        # the coordinate is frozen and should stay at zero
        frozen = (np.abs(pos[k]) < zero_tol) & (np.abs(vel[k]) < zero_tol)
        samples[j, frozen] = 0.0

    return samples