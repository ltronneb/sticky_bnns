"""
Minibatch variant of GridStickyBoomerangSampler -- see mb_grid_zigzag.py's
module docstring for the full rationale of the `resample_grad_batch` hook.

`_grid_bound` is the per-loop-iteration chokepoint (called once per
Poisson-thinning proposal, AFTER dt_hit/dt_thaw, which are closed-form and
never touch grad_target), so the minibatch swap goes there, before
delegating to the unchanged base implementation. As in the non-sticky
Boomerang case, the reference measure (x_ref/Sigma_inv) is a fixed Gaussian
anchor and is NOT resampled; freeze/thaw scheduling and the sticky
trajectory are inherited untouched.

resample_grad_batch=None reproduces GridStickyBoomerangSampler bit-for-bit.
"""

from typing import Callable, Optional

from torch import Tensor

from .grid_sticky_boomerang import GridStickyBoomerangSampler


class MbGridStickyBoomerangSampler(GridStickyBoomerangSampler):
    def __init__(
        self,
        *args,
        resample_grad_batch: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._resample_grad_batch = resample_grad_batch

    def _grid_bound(
        self, pos: Tensor, vel: Tensor, dt_refresh: float, dt_hit: float, dt_thaw: float,
    ) -> tuple[float, dict]:
        if self._resample_grad_batch is not None:
            self._resample_grad_batch()
        return super()._grid_bound(pos, vel, dt_refresh, dt_hit, dt_thaw)
