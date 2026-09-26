"""
Staged sticky PDMP runs with draws uniform in simulated time.

A run of n_skeleton events is produced in stages of stage_size events on one
_Cheap sampler, chained through its resume_state. Each stage is written to
disk in chunks, fed to a UniformTimeReservoir (resample.py) and deleted
before the next stage starts, so peak disk is one stage while the final
draws are iid uniform in time over the whole trajectory, after dropping the
first burnin_frac of its TIME. This is the loop of
deep_wide_uci._run_staged_sticky, shared by the image drivers
(fast_cheap_ffn_mnist.py, fast_cheap_mnist_cnn.py, fast_cheap_cifar_resnet.py).
"""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Optional

import torch

from sazz.gpu_friendly.utils.resample import stage_time_range, UniformTimeReservoir


def bound_kwargs(bound_mode: str, adapt_rule: str, t_max_init: float, spacing: float,
                 single_segment_t_max_init: Optional[float] = None) -> dict:
    """Sampler kwargs for the bound mode. "grid" passes exactly what the
    builders passed before --bound-mode existed. "single_segment" starts t_max
    at the old grid spacing unless single_segment_t_max_init overrides it."""
    if bound_mode == "grid":
        return {"grid_t_max_init": t_max_init}
    if single_segment_t_max_init is not None:
        t_max_init = single_segment_t_max_init
    else:
        t_max_init = spacing
    return {"grid_t_max_init": t_max_init, "bound_mode": bound_mode, "adapt_rule": adapt_rule}


def run_staged_uniform_time(sampler, x0, *, n_skeleton: int, stage_size: int, stage_dir: Path,
                            n_out: int, burnin_frac: float, resample_stage_fn,
                            pool_per_stage: int = 500) -> dict:
    """
    resample_stage_fn(chunk_files, manifest_path, n) -> (draws [n, D], times [n])
    with the n times iid uniform over the whole stage, i.e. a chunked sticky
    resampler called with burnin_frac=0.0 and return_times=True.

    Returns the n_out draws sorted by time, their times, and the run totals.
    """
    reservoir = UniformTimeReservoir(n_out, burnin_frac)
    n_stages = math.ceil(n_skeleton / stage_size)
    stage_spans: list[float] = []
    total_grad_evals = 0
    total_bound_violations = 0
    all_tmax_log: list[float] = []
    frozen_mask_final = None
    resume_state = None

    t0 = time.perf_counter()
    for stage in range(n_stages):
        # sample()'s N counts row 0 (x0, or a copy of the previous stage's
        # last row) plus N - 1 new events
        stage_new_events = min(stage_size, n_skeleton - stage * stage_size)
        this_dir = stage_dir / f"stage_{stage:04d}"
        print(f"      [stage {stage + 1}/{n_stages}] sampling {stage_new_events} skeleton points "
              f"({'cold start' if stage == 0 else 'resumed'}) -> {this_dir}")

        result = sampler.sample(
            N=stage_new_events + 1,
            x0=(x0 if stage == 0 else None),
            resume_state=resume_state,
            diagnostics=True,
            chunk_size=stage_size,
            chunk_dir=this_dir,
        )
        chunk_files, manifest_path = result["chunk_files"], result["manifest_path"]
        t_start, t_end = stage_time_range(chunk_files, manifest_path)

        def draw_fn(n: int):
            # At most pool_per_stage draws per resampler call, which bounds the
            # transient [n, D] allocation on the device
            parts, times = [], []
            for lo in range(0, n, pool_per_stage):
                d, t = resample_stage_fn(chunk_files, manifest_path, min(pool_per_stage, n - lo))
                parts.append(d.cpu())
                times.append(t.cpu())
            return torch.cat(parts), torch.cat(times)

        n_switched = reservoir.add_stage(t_start, t_end, draw_fn)
        stage_spans.append(t_end - t_start)
        print(f"      [stage {stage + 1}/{n_stages}] sim-time [{t_start:.6g}, {t_end:.6g}], "
              f"{n_switched}/{reservoir.n_slots} reservoir slots moved here")

        total_grad_evals += result["gradient_evals"]
        total_bound_violations += result["bound_violations"]
        all_tmax_log.extend(result["grid_t_max_log"])
        frozen_mask_final = result["frozen_mask_final"]
        resume_state = result["resume_state"]

        if this_dir.exists():
            shutil.rmtree(this_dir)

    elapsed = time.perf_counter() - t0
    samples, times, info = reservoir.finalize()
    order = torch.argsort(times)
    samples, times = samples[order], times[order]
    print(f"      uniform-in-time resample: total sim-time {info['t_end'] - info['t0']:.6g} across "
          f"{n_stages} stages, burn-in cut at t={info['burnin_t_cut']:.6g} "
          f"({burnin_frac:.0%} of the time), {info['n_survivors']}/{info['n_slots']} slots after "
          f"burn-in, kept {samples.shape[0]}")

    return {
        "stage_size": stage_size,
        "burnin_frac": burnin_frac,
        "samples": samples,
        "sample_times": times,
        "reservoir": info,
        "stage_time_spans": stage_spans,
        "n_stages": n_stages,
        "gradient_evals": total_grad_evals,
        "bound_violations": total_bound_violations,
        "grid_t_max_log": all_tmax_log,
        "frozen_mask_final": frozen_mask_final,
        "elapsed_sec": elapsed,
    }


def run_provenance(run: dict, thin, **settings) -> dict:
    """Extra fields for a saved run. thin(tensor) is the same thinning the
    caller applies to the samples, so sample_times[i] stays the time of
    samples[i]. settings are recorded as given (bound_mode, adapt_rule, ...)."""
    return {
        "resample_scheme": "uniform_time_reservoir",
        "sample_times": thin(run["sample_times"].unsqueeze(-1)).squeeze(-1),
        "reservoir": run["reservoir"],
        "stage_time_spans": run["stage_time_spans"],
        "stage_size": run["stage_size"],
        "n_stages": run["n_stages"],
        "burnin_frac": run["burnin_frac"],
        **settings,
    }
