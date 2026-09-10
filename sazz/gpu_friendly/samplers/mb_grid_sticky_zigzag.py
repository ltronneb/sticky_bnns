"""
Minibatch variant of GridStickyZigZagSampler -- see mb_grid_zigzag.py's
module docstring for the full rationale of the `resample_grad_batch` hook.

Same story as the non-sticky case: `_grid_bound` is the single per-loop-
iteration chokepoint (called once per Poisson-thinning proposal, AFTER the
closed-form dt_hit/dt_thaw computations, which never touch grad_target), so
resampling the minibatch there -- then delegating to the unchanged base
_grid_bound -- swaps the batch at exactly the fast-sampler cadence without
duplicating sample(). freeze/thaw scheduling, the sticky trajectory, and
all diagnostics are inherited untouched.

resample_grad_batch=None reproduces GridStickyZigZagSampler bit-for-bit.
"""

from typing import Callable, Optional

from torch import Tensor

from .grid_sticky_zigzag import GridStickyZigZagSampler


class MbGridStickyZigZagSampler(GridStickyZigZagSampler):
    def __init__(
        self,
        *args,
        resample_grad_batch: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._resample_grad_batch = resample_grad_batch

    def _grid_bound(
        self, pos: Tensor, vel: Tensor, dt_hit: float, dt_thaw: float,
    ) -> tuple[float, dict]:
        if self._resample_grad_batch is not None:
            self._resample_grad_batch()
        return super()._grid_bound(pos, vel, dt_hit, dt_thaw)
