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

---

The four *_chunked_torch functions below are ADDITIONS ONLY -- the four
functions above are never modified. They resample directly from a
sequence of chunk .pt files (as written by fast_grid_sticky_zigzag.py /
fast_grid_sticky_boomerang.py's chunked sample()) without ever
materializing the full [N, D] trajectory in memory, per the chunked-
checkpointing plan (radiant-finding-church.md).

CLAMP BOUND, the one place these functions genuinely differ from the four
above, not just restructure them: within a single chunk, the clamp is
`indices.clamp(0, tim.shape[0] - 1)`, NOT `- 2`. The `-2` bound above is
correct for a WHOLE trajectory (the final skeleton row can never be a left
endpoint -- there's no successor interval past it). Applied per-chunk,
that's wrong: a chunk's last row legitimately IS a valid left endpoint for
any sample time up to the next chunk's first row. Using `-2` per-chunk
would silently exclude it, anchoring those draws to the second-to-last row
and producing an over-long, incorrect dt -- see the plan's review history
for how this was caught.

No carry-over/prefix row is needed at chunk boundaries: the interpolation
formulas below (like the four functions above) are functions of the LEFT
endpoint and dt only -- there is no right-endpoint term anywhere in the
math, so a sample time in [chunk_k's last row, chunk_{k+1}'s first row)
anchors correctly to chunk k's own last row, already present in chunk k.
"""

import bisect
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor


def _burnin_cut(positions: Tensor, velocities: Tensor, times: Tensor,
                 burnin_frac: float) -> tuple[Tensor, Tensor, Tensor]:
    """
    Burn in by ELAPSED TIME, not skeleton-row count: skeleton density isn't
    uniform in time (e.g. sticky freeze/thaw churn is denser early in a run
    as sparsity climbs from 0), so cutting the first `burnin_frac * N` ROWS
    discards a different (often smaller) fraction of simulated time than
    burnin_frac names -- silently biasing resampled draws toward the early,
    less-equilibrated part of the trajectory.
    """
    t_cut = times[0] + burnin_frac * (times[-1] - times[0])
    n_burn = int(torch.searchsorted(times, t_cut).clamp(max=times.shape[0] - 2))
    return positions[n_burn:], velocities[n_burn:], times[n_burn:]


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
    pos, vel, tim = _burnin_cut(positions, velocities, times, burnin_frac)

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
    pos, vel, tim = _burnin_cut(positions, velocities, times, burnin_frac)

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

    burnin_frac cuts by ELAPSED TIME (see _burnin_cut), not skeleton index.
    """
    device = positions.device
    dtype = positions.dtype
    pos, vel, tim = _burnin_cut(positions, velocities, times, burnin_frac)

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
    pos, vel, tim = _burnin_cut(positions, velocities, times, burnin_frac)

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


# ===========================================================================
# Chunk-aware resamplers — new additions, see module docstring.
# ===========================================================================

def _load_manifest(chunk_files: list[Path] | list[str], manifest_path: Optional[Path | str]) -> list[dict]:
    """
    Returns the manifest's per-chunk metadata list, sorted by chunk_idx.
    Prefers manifest_path (a single small file written once per flush by
    sample() -- see fast_grid_sticky_zigzag.py's _write_manifest) so this
    never has to open a chunk file (which contains the full [rows, D]
    tensors) just to learn its time range. Falls back to opening every
    chunk file's metadata fields directly (still loads the tensor payload
    in the process, since torch.load has no partial-read mode -- slower,
    but correct) only if no manifest_path is given.
    """
    if manifest_path is not None:
        manifest = torch.load(manifest_path, weights_only=False)
        return sorted(manifest["chunks"], key=lambda e: e["chunk_idx"])

    entries = []
    for cf in chunk_files:
        chunk = torch.load(cf, weights_only=False, map_location="cpu")
        entries.append({
            "chunk_idx": chunk["chunk_idx"],
            "row_count": chunk["row_count"],
            "t_start": chunk["t_start"],
            "t_end": chunk["t_end"],
            "global_row_start": chunk["global_row_start"],
            "path": str(cf),
        })
    return sorted(entries, key=lambda e: e["chunk_idx"])


def _find_burnin_t_start(chunk_entries: list[dict], n_burn: int) -> float:
    """
    Returns times[n_burn] (the global skeleton row index n_burn's time
    value) by loading ONLY the one chunk file containing that row --
    never the full trajectory. n_burn falls in chunk k iff
    chunk_k.global_row_start <= n_burn < chunk_k.global_row_start + row_count.
    """
    for entry in chunk_entries:
        start = entry["global_row_start"]
        end = start + entry["row_count"]
        if start <= n_burn < end:
            chunk = torch.load(entry["path"], weights_only=False, map_location="cpu")
            local_idx = n_burn - start
            return float(chunk["times"][local_idx])
    # n_burn >= total row count (e.g. burnin_frac close to/at 1.0, or a
    # single-row final chunk) -- fall back to the last chunk's last row.
    last = chunk_entries[-1]
    chunk = torch.load(last["path"], weights_only=False, map_location="cpu")
    return float(chunk["times"][-1])


def _resample_chunked_core(
    chunk_files: list[Path] | list[str], N_resample: int, burnin_frac: float,
    dtype: torch.dtype, device: torch.device | str, manifest_path: Optional[Path | str],
    interpolate_fn,
) -> Tensor:
    """
    Shared core for all four *_chunked_torch functions below -- mirrors
    the four in-memory resamplers' shared searchsorted-and-gather skeleton
    (see module docstring), just batched per-chunk instead of once over a
    full [N, D] tensor. `interpolate_fn(k_pos, k_vel, k_tim, sample_times)
    -> samples` supplies the one piece that differs between ZigZag/
    Boomerang and sticky/non-sticky -- identical in spirit to the shared
    dt/k_pos/k_vel gather these four functions already do, just factored
    out so this core doesn't need to know which dynamics it's serving.
    """
    chunk_entries = _load_manifest(chunk_files, manifest_path)
    if not chunk_entries:
        raise ValueError("No chunks found -- chunk_files/manifest_path is empty.")

    N = sum(e["row_count"] for e in chunk_entries)
    n_burn = int(burnin_frac * N)

    T_start = _find_burnin_t_start(chunk_entries, n_burn)
    T_end = chunk_entries[-1]["t_end"]
    if T_end <= T_start:
        raise ValueError("Skeleton times are not increasing after burnin.")

    sample_times = torch.rand(N_resample, dtype=dtype, device=device) * (T_end - T_start) + T_start
    # NOTE: the four in-memory resamplers above do
    # `sample_times, _ = torch.sort(sample_times)` and DISCARD the sort
    # index -- their output row k is the k-th SMALLEST drawn time, not the
    # k-th DRAWN time. This function must match that exact convention
    # (output in sorted-time order), not "unsort" back to draw order, or
    # its output would silently permute relative to the in-memory
    # resamplers' for the same draws.
    sample_times, _ = torch.sort(sample_times)

    # Chunk lookup: searchsorted against chunk START times (NOT t_end --
    # see module docstring's off-by-one warning), right=True then -1,
    # matching the exact convention the per-chunk searchsorted below
    # applies to skeleton times within a chunk. Vectorized over every
    # sample_time in one call.
    starts = torch.tensor([e["t_start"] for e in chunk_entries], dtype=dtype, device=device)
    chunk_id = torch.searchsorted(starts, sample_times, right=True) - 1
    chunk_id = chunk_id.clamp_min(0)
    chunk_id_list = chunk_id.tolist()

    D = None
    output: Optional[Tensor] = None

    # sample_times (and therefore chunk_id) came out of torch.sort already
    # sorted, and chunk time-ranges are monotonic/non-overlapping, so
    # chunk_id is itself non-decreasing -- grouping into contiguous runs
    # is a single linear pass, not a full groupby.
    i = 0
    n_samples = sample_times.shape[0]
    while i < n_samples:
        cid = chunk_id_list[i]
        j = i
        while j < n_samples and chunk_id_list[j] == cid:
            j += 1
        # sample_times[i:j] all belong to chunk cid -- load it once.
        entry = chunk_entries[cid]
        chunk = torch.load(entry["path"], weights_only=False, map_location="cpu")
        pos = chunk["positions"].to(dtype=dtype, device=device)
        vel = chunk["velocities"].to(dtype=dtype, device=device)
        tim = chunk["times"].to(dtype=dtype, device=device)

        if D is None:
            D = pos.shape[1]
            output = torch.zeros(N_resample, D, dtype=dtype, device=device)

        sub_times = sample_times[i:j]

        # tim[idx-1] <= t < tim[idx]; -1 clamp bound (NOT -2 -- see module
        # docstring), since within a single chunk the last row IS a valid
        # left endpoint for times up to the next chunk's first row.
        indices = torch.searchsorted(tim, sub_times, right=True) - 1
        indices = indices.clamp(0, tim.shape[0] - 1)

        k_pos = pos[indices]
        k_vel = vel[indices]
        k_tim = tim[indices]

        sub_samples = interpolate_fn(k_pos, k_vel, k_tim, sub_times)

        # Write directly at [i:j] -- sample_times (and therefore this
        # loop's positions) are already in the sorted-time order the
        # in-memory resamplers return their output in (see the sort note
        # above); no unsort/scatter needed.
        output[i:j] = sub_samples

        i = j

    return output


def resample_zigzag_path_chunked_torch(
    chunk_files: list[Path] | list[str], N_resample: int, burnin_frac: float = 0.1,
    manifest_path: Optional[Path | str] = None,
    dtype: torch.dtype = torch.float64, device: torch.device | str = "cpu",
) -> Tensor:
    """
    Chunk-aware equivalent of resample_zigzag_path_torch -- reads directly
    from chunk_files (as written by a chunked sample() call) instead of a
    single in-memory [N, D] trajectory. dtype/device are explicit required-
    in-spirit parameters (defaulted, not inherited) since no full tensor
    is in hand up front to infer them from -- see module docstring.
    """
    def interpolate_fn(k_pos, k_vel, k_tim, sample_times):
        dt = (sample_times - k_tim).unsqueeze(-1)
        return k_pos + k_vel * dt

    return _resample_chunked_core(
        chunk_files, N_resample, burnin_frac, dtype, device, manifest_path, interpolate_fn,
    )


def resample_zigzag_path_sticky_chunked_torch(
    chunk_files: list[Path] | list[str], N_resample: int, burnin_frac: float = 0.1,
    zero_tol: float = 1e-12, manifest_path: Optional[Path | str] = None,
    dtype: torch.dtype = torch.float64, device: torch.device | str = "cpu",
) -> Tensor:
    """
    Sticky, chunk-aware equivalent of resample_zigzag_path_sticky_torch --
    same left-endpoint frozen-detection convention (see that function's
    docstring), applied per-chunk.
    """
    def interpolate_fn(k_pos, k_vel, k_tim, sample_times):
        dt = (sample_times - k_tim).unsqueeze(-1)
        samples = k_pos + k_vel * dt
        frozen = (k_pos.abs() < zero_tol) & (k_vel.abs() < zero_tol)
        return torch.where(frozen, torch.zeros_like(samples), samples)

    return _resample_chunked_core(
        chunk_files, N_resample, burnin_frac, dtype, device, manifest_path, interpolate_fn,
    )


def resample_boomerang_path_chunked_torch(
    chunk_files: list[Path] | list[str], x_ref: Tensor, N_resample: int, burnin_frac: float = 0.1,
    manifest_path: Optional[Path | str] = None,
    dtype: torch.dtype = torch.float64, device: torch.device | str = "cpu",
) -> Tensor:
    """
    Chunk-aware equivalent of resample_boomerang_path_torch.
    """
    def interpolate_fn(k_pos, k_vel, k_tim, sample_times):
        dt = (sample_times - k_tim).unsqueeze(-1)
        dx = k_pos - x_ref
        return x_ref + dx * torch.cos(dt) + k_vel * torch.sin(dt)

    return _resample_chunked_core(
        chunk_files, N_resample, burnin_frac, dtype, device, manifest_path, interpolate_fn,
    )


def resample_boomerang_path_sticky_chunked_torch(
    chunk_files: list[Path] | list[str], x_ref: Tensor, N_resample: int, burnin_frac: float = 0.1,
    zero_tol: float = 1e-12, manifest_path: Optional[Path | str] = None,
    dtype: torch.dtype = torch.float64, device: torch.device | str = "cpu",
) -> Tensor:
    """
    Sticky, chunk-aware equivalent of resample_boomerang_path_sticky_torch.
    """
    def interpolate_fn(k_pos, k_vel, k_tim, sample_times):
        dt = (sample_times - k_tim).unsqueeze(-1)
        dx = k_pos - x_ref
        samples = x_ref + dx * torch.cos(dt) + k_vel * torch.sin(dt)
        frozen = (k_pos.abs() < zero_tol) & (k_vel.abs() < zero_tol)
        return torch.where(frozen, torch.zeros_like(samples), samples)

    return _resample_chunked_core(
        chunk_files, N_resample, burnin_frac, dtype, device, manifest_path, interpolate_fn,
    )
