"""Posterior draws from a PDMP trajectory, uniform in time.

The skeleton arrives in consecutive chunks (the last row of one chunk is the
first row of the next). A reservoir of independent slots keeps, for every
slot, one point uniform over the trajectory seen so far. When a chunk covering
[a, b] arrives, each slot moves to a fresh uniform point in [a, b] with
probability (b - a) / (b - t0). At the end, slots in the first burnin_frac of
the time are dropped and n_out of the rest are kept, which gives iid draws
uniform over the post-burn-in trajectory without storing the skeleton.
"""

import math
import time
from typing import Callable, Optional

import torch
from torch import Tensor


def path_at(positions: Tensor, velocities: Tensor, times: Tensor, t: Tensor,
            trajectory: Callable) -> Tensor:
    """Evaluate the piecewise-deterministic path at times t (sorted, within
    [times[0], times[-1]]). Coordinates frozen at the left skeleton point
    (x = v = 0) stay at exactly 0."""
    idx = (torch.searchsorted(times, t, right=True) - 1).clamp(0, times.shape[0] - 2)
    x, v = positions[idx], velocities[idx]
    dt = (t - times[idx]).to(x.dtype).unsqueeze(-1)
    out, _ = trajectory(dt, x, v)
    frozen = (x == 0) & (v == 0)
    return torch.where(frozen, torch.zeros_like(out), out)


class UniformTimeReservoir:
    """See the module docstring. n_slots = ceil(slack * n_out / (1 - burnin))."""

    def __init__(self, n_out: int, burnin_frac: float = 0.2, slack: float = 1.2):
        self.n_out, self.burnin_frac = n_out, burnin_frac
        self.n_slots = math.ceil(slack * n_out / (1.0 - burnin_frac))
        self.times = torch.full((self.n_slots,), math.nan, dtype=torch.float64)
        self.draws: Optional[Tensor] = None
        self.t0: Optional[float] = None
        self.t_end: Optional[float] = None

    def add_chunk(self, positions: Tensor, velocities: Tensor, times: Tensor,
                  trajectory: Callable, max_batch: int = 2000) -> None:
        a, b = float(times[0]), float(times[-1])
        if b <= a:
            return
        if self.t0 is None:
            self.t0, p = a, 1.0
        else:
            p = (b - a) / (b - self.t0)
        self.t_end = b
        idx = torch.nonzero(torch.rand(self.n_slots, dtype=torch.float64) < p).squeeze(-1)
        for lo in range(0, idx.numel(), max_batch):
            slots = idx[lo:lo + max_batch]
            t = torch.sort(a + (b - a) * torch.rand(slots.numel(), dtype=torch.float64,
                                                     device=times.device)).values
            x = path_at(positions, velocities, times, t, trajectory).cpu()
            if self.draws is None:
                self.draws = torch.zeros(self.n_slots, x.shape[1], dtype=x.dtype)
            self.draws[slots] = x
            self.times[slots] = t.cpu()

    def finalize(self) -> Tensor:
        """n_out draws sorted by time (fewer, with a warning, if too few slots
        survive the burn-in, which happens with negligible probability)."""
        t_cut = self.t0 + self.burnin_frac * (self.t_end - self.t0)
        keep = torch.nonzero(self.times >= t_cut).squeeze(-1)
        if keep.numel() < self.n_out:
            print(f"  warning: only {keep.numel()} post-burn-in draws")
        keep = keep[torch.randperm(keep.numel())[: self.n_out]]
        return self.draws[keep[torch.argsort(self.times[keep])]]


def run_and_resample(sampler, x0: Tensor, n_out: int = 4000, burnin_frac: float = 0.2,
                     chunk_size: int = 10_000, keep_skeleton: bool = False, **budget) -> dict:
    """Run sampler.sample(x0, **budget) streaming chunks into a reservoir.
    Returns {"samples", "elapsed_sec", **sampler summary} (+ "skeleton")."""
    reservoir = UniformTimeReservoir(n_out, burnin_frac)
    skeleton = []

    def on_chunk(pos, vel, times):
        reservoir.add_chunk(pos, vel, times, sampler.trajectory)
        if keep_skeleton:
            skeleton.append((pos.cpu().clone(), vel.cpu().clone(), times.cpu().clone()))

    t0 = time.perf_counter()
    out = sampler.sample(x0, chunk_size=chunk_size, on_chunk=on_chunk, **budget)
    out["elapsed_sec"] = time.perf_counter() - t0
    out["samples"] = reservoir.finalize()
    if keep_skeleton:
        out["skeleton"] = {
            "positions": torch.cat([skeleton[0][0]] + [c[0][1:] for c in skeleton[1:]]),
            "velocities": torch.cat([skeleton[0][1]] + [c[1][1:] for c in skeleton[1:]]),
            "times": torch.cat([skeleton[0][2]] + [c[2][1:] for c in skeleton[1:]]),
        }
    return out
