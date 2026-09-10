"""
Minibatch variant of GridZigZagSampler -- identical dynamics, bound
construction, thinning and diagnostics, with ONE addition: an optional
`resample_grad_batch` callback that is invoked exactly once per sample()
loop iteration, immediately before the grid bound is built.

Rationale (see sazz/gpu_friendly/scripts/fast_mnist_cnn.py::build_minibatch_grad_target
and FastGridStickyZigZagSampler for the original of this hook):

  * `grad_target` must stay a FIXED function of x for the whole duration of
    one _grid_bound episode -- every rate evaluation inside grid_thinning
    (the eager accept/reject checks and the vmapped/jvp'd bound builder
    alike) has to see the same rate function, or the grid upper bound is
    not a valid bound. So the minibatch may only be swapped BETWEEN
    episodes, never inside one.
  * `_grid_bound` is the single chokepoint called once per outer loop
    iteration (one Poisson-thinning proposal), so overriding it to call
    the resample hook first -- then delegating to the unchanged base
    implementation -- puts the swap at exactly the right cadence without
    duplicating the ~200-line sample() loop.
  * The hook itself is a plain, non-vmapped call (torch.randint cannot run
    inside torch.func.vmap: "called random operation while in randomness
    error mode"), which is why the draw lives in the callback and not in
    grad_target's body.

resample_grad_batch=None (the default) reproduces GridZigZagSampler
bit-for-bit -- the override becomes a no-op passthrough.
"""

from typing import Callable, Optional

from torch import Tensor

from .grid_zigzag import GridZigZagSampler


class MbGridZigZagSampler(GridZigZagSampler):
    def __init__(
        self,
        *args,
        resample_grad_batch: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._resample_grad_batch = resample_grad_batch

    def _grid_bound(self, pos: Tensor, vel: Tensor) -> tuple[float, dict]:
        if self._resample_grad_batch is not None:
            self._resample_grad_batch()
        return super()._grid_bound(pos, vel)
