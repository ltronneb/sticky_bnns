"""
Minibatch variant of GridBoomerangSampler -- see mb_grid_zigzag.py's module
docstring for the full rationale of the `resample_grad_batch` hook.

`_grid_bound` is the per-loop-iteration chokepoint (one Poisson-thinning
proposal per call), so the minibatch swap goes there, before delegating to
the unchanged base implementation. The Boomerang reference measure
(x_ref/Sigma_inv, set via preprocess()) is NOT resampled -- it is a fixed
Gaussian anchor, independent of which data minibatch the excess-gradient is
evaluated against; only grad_target's data changes between episodes.

resample_grad_batch=None reproduces GridBoomerangSampler bit-for-bit.
"""

from typing import Callable, Optional

from torch import Tensor

from .grid_boomerang import GridBoomerangSampler


class MbGridBoomerangSampler(GridBoomerangSampler):
    def __init__(
        self,
        *args,
        resample_grad_batch: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._resample_grad_batch = resample_grad_batch

    def _grid_bound(
        self, pos: Tensor, vel: Tensor, dt_refresh: float,
    ) -> tuple[float, dict]:
        if self._resample_grad_batch is not None:
            self._resample_grad_batch()
        return super()._grid_bound(pos, vel, dt_refresh)
