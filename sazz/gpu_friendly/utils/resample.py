"""
Vectorized torch versions of sazz.utils.sampling's resample_boomerang_path
[_sticky]. Same math (Boomerang interpolation between skeleton points),
but batched over all N_resample draws at once instead of a Python
`for j in range(N_resample)` loop -- that loop, not the .cpu().numpy()
transfer it forces on GPU/MPS runs, is the actual bottleneck (it's just
as slow on plain CPU). Stays entirely in torch/on-device throughout; no
numpy anywhere, so no host round-trip is needed regardless of device.

sazz/utils/sampling.py is left untouched (isolation requirement) -- these
are new, independent functions, not a replacement of that file.
"""

from typing import Optional

import torch
from torch import Tensor


def resample_boomerang_path_torch(
    positions: Tensor, velocities: Tensor, times: Tensor, x_ref: Tensor,
    N_resample: int, burnin_frac: float = 0.1,
) -> Tensor:
    """
    Given skeleton (positions, velocities, times), resample N_resample
    points uniformly in trajectory time using the Boomerang dynamics:
        x(t) = x_ref + (x_k - x_ref)*cos(t - t_k) + v_k*sin(t - t_k)

    All inputs are torch.Tensor, already on the device to compute on.
    Mirrors sazz.utils.sampling.resample_boomerang_path exactly, batched.
    """
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]
    n_burn = int(burnin_frac * N)
    pos = positions[n_burn:]
    vel = velocities[n_burn:]
    tim = times[n_burn:]

    T_start = tim[0]
    T_end = tim[-1]
    if T_end <= T_start:
        raise ValueError("Skeleton times are not increasing after burnin.")

    sample_times = torch.rand(N_resample, dtype=dtype, device=device) * (T_end - T_start) + T_start
    sample_times, _ = torch.sort(sample_times)

    # tim[idx-1] <= t < tim[idx]
    indices = torch.searchsorted(tim, sample_times, right=True) - 1
    indices = indices.clamp(0, tim.shape[0] - 2)

    k_pos = pos[indices]          # [N_resample, D]
    k_vel = vel[indices]          # [N_resample, D]
    k_tim = tim[indices]          # [N_resample]

    dt = (sample_times - k_tim).unsqueeze(-1)   # [N_resample, 1]
    dx = k_pos - x_ref                          # [N_resample, D]
    samples = x_ref + dx * torch.cos(dt) + k_vel * torch.sin(dt)

    return samples


def resample_boomerang_path_sticky_torch(
    positions: Tensor, velocities: Tensor, times: Tensor, x_ref: Tensor,
    N_resample: int, burnin_frac: float = 0.1, zero_tol: float = 1e-12,
) -> Tensor:
    """
    Sticky variant: overrides frozen coordinates (x_k[i] == 0 and
    v_k[i] == 0 at the anchoring skeleton point) to stay exactly at zero,
    rather than interpolating them. Mirrors
    sazz.utils.sampling.resample_boomerang_path_sticky exactly, batched.
    """
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]
    n_burn = int(burnin_frac * N)
    pos = positions[n_burn:]
    vel = velocities[n_burn:]
    tim = times[n_burn:]

    T_start = tim[0]
    T_end = tim[-1]
    if T_end <= T_start:
        raise ValueError("Skeleton times are not increasing after burnin.")

    sample_times = torch.rand(N_resample, dtype=dtype, device=device) * (T_end - T_start) + T_start
    sample_times, _ = torch.sort(sample_times)

    indices = torch.searchsorted(tim, sample_times, right=True) - 1
    indices = indices.clamp(0, tim.shape[0] - 2)

    k_pos = pos[indices]
    k_vel = vel[indices]
    k_tim = tim[indices]

    dt = (sample_times - k_tim).unsqueeze(-1)
    dx = k_pos - x_ref
    samples = x_ref + dx * torch.cos(dt) + k_vel * torch.sin(dt)

    frozen = (k_pos.abs() < zero_tol) & (k_vel.abs() < zero_tol)
    samples = torch.where(frozen, torch.zeros_like(samples), samples)

    return samples


def resample_zigzag_path_torch(
    positions: Tensor, velocities: Tensor, times: Tensor,
    N_resample: int, burnin_frac: float = 0.1,
) -> Tensor:
    """
    Given skeleton (positions, velocities, times), resample N_resample
    points uniformly in trajectory time using ZigZag's LINEAR dynamics:
        x(t) = x_k + (t - t_k) * v_k
    No x_ref argument -- unlike Boomerang, ZigZag has no reference measure.

    All inputs are torch.Tensor, already on the device to compute on.
    Mirrors sazz.utils.sampling.resample_zigzag_path exactly, batched
    (same searchsorted pattern resample_boomerang_path_torch already uses).

    Note: burnin_frac cuts by SKELETON INDEX, while resampling itself is
    uniform in TIME -- since skeleton points are not evenly spaced in time,
    the actual discarded time duration is not exactly burnin_frac of the
    total simulated clock (same caveat implicitly true of the existing
    Boomerang resamplers above).
    """
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]
    n_burn = int(burnin_frac * N)
    pos = positions[n_burn:]
    vel = velocities[n_burn:]
    tim = times[n_burn:]

    T_start = tim[0]
    T_end = tim[-1]
    if T_end <= T_start:
        raise ValueError("Skeleton times are not increasing after burnin.")

    sample_times = torch.rand(N_resample, dtype=dtype, device=device) * (T_end - T_start) + T_start
    sample_times, _ = torch.sort(sample_times)

    # tim[idx-1] <= t < tim[idx]
    indices = torch.searchsorted(tim, sample_times, right=True) - 1
    indices = indices.clamp(0, tim.shape[0] - 2)

    k_pos = pos[indices]          # [N_resample, D]
    k_vel = vel[indices]          # [N_resample, D]
    k_tim = tim[indices]          # [N_resample]

    dt = (sample_times - k_tim).unsqueeze(-1)   # [N_resample, 1]
    samples = k_pos + k_vel * dt

    return samples


def resample_zigzag_path_sticky_torch(
    positions: Tensor, velocities: Tensor, times: Tensor,
    N_resample: int, burnin_frac: float = 0.1, zero_tol: float = 1e-12,
) -> Tensor:
    """
    Sticky variant of resample_zigzag_path_torch: overrides coordinates
    frozen throughout an interval to stay exactly at zero, rather than
    interpolating them. Mirrors sazz.utils.sampling.resample_zigzag_path_sticky
    exactly, batched.

    Frozen-ness is decided from the LEFT skeleton endpoint of each interval
    (k_pos/k_vel via the same searchsorted-minus-one index resample_zigzag_
    path_torch already uses) -- this is why GridStickyZigZagSampler.sample()
    must hand-zero a newly-frozen coordinate's position/velocity at the
    freeze skeleton point itself: this resampler has no other way to know
    the coordinate was frozen over that interval. ZigZag velocities are
    exactly 0.0 when frozen and exactly +-1.0 when active, so the
    |pos|<zero_tol & |vel|<zero_tol check is unambiguous.
    """
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]
    n_burn = int(burnin_frac * N)
    pos = positions[n_burn:]
    vel = velocities[n_burn:]
    tim = times[n_burn:]

    T_start = tim[0]
    T_end = tim[-1]
    if T_end <= T_start:
        raise ValueError("Skeleton times are not increasing after burnin.")

    sample_times = torch.rand(N_resample, dtype=dtype, device=device) * (T_end - T_start) + T_start
    sample_times, _ = torch.sort(sample_times)

    indices = torch.searchsorted(tim, sample_times, right=True) - 1
    indices = indices.clamp(0, tim.shape[0] - 2)

    k_pos = pos[indices]          # [N_resample, D]
    k_vel = vel[indices]          # [N_resample, D]
    k_tim = tim[indices]          # [N_resample]

    dt = (sample_times - k_tim).unsqueeze(-1)   # [N_resample, 1]
    samples = k_pos + k_vel * dt

    frozen = (k_pos.abs() < zero_tol) & (k_vel.abs() < zero_tol)
    samples = torch.where(frozen, torch.zeros_like(samples), samples)

    return samples
