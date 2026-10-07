"""SG-PDMPs on the 1-D toy regressions of sazz.paper_ready.scripts.toy_bnn, the same
network (one hidden layer of 100 tanh units, known noise), prior and MAP. The
default budget is the paper's G = 2e5 full-data gradients per run.

    python -m sazz.sg_pdmp.scripts.toy_bnn
    python -m sazz.sg_pdmp.scripts.toy_bnn --batch-size 1 --step-sizes 1e-3 1e-4 1e-5
    python -m sazz.sg_pdmp.scripts.toy_bnn --x-ref-from results/paper_v2/toy_bnns

The MAP is fitted as in paper_ready (same seed, so the same MAP), or read from a
result file with an x_ref under --x-ref-from/<data>/split_00.
"""

import argparse
from pathlib import Path

import torch

from sazz.paper_ready.common import DEVICE, DTYPE, save, seed_all
from sazz.paper_ready.data import toy_data
from sazz.paper_ready.models.bnn import BNN, prior_std
from sazz.paper_ready.models.networks import FFN
from sazz.paper_ready.samplers import run_and_resample
from sazz.paper_ready.scripts.toy_bnn import ACT, DATASETS, INCLUSION, LAYERS, PRIOR_STD, SIGMA_INV_SCALE
from sazz.paper_ready.utils.reference import fit_map, laplace_precision

from ..common import SG_PDMPS, make_sg_pdmp, nan_samples, run_name, summary
from .uci_bnn import x_ref_from_results

BATCH = 4


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--samplers", nargs="+", default=list(SG_PDMPS), choices=SG_PDMPS)
    p.add_argument("--step-sizes", nargs="+", type=float, default=[1e-3, 1e-4])
    p.add_argument("--batch-size", type=int, default=BATCH, help="data points per iteration")
    p.add_argument("--grad-budget", type=float, default=200_000, help="full-data gradients per run")
    p.add_argument("--n-iters", type=int, default=None, help="iterations per run, overrides the budget")
    p.add_argument("--max-minutes", type=float, default=None,
                   help="also stop each run after this much wall-clock time")
    p.add_argument("--precondition", action="store_true",
                   help="SG-BPS velocities N(0, Sigma) with the Boomerang's Laplace covariance")
    p.add_argument("--refresh-rate", type=float, default=1.0, help="SG-BPS refreshment rate")
    p.add_argument("--no-cv", action="store_true", help="plain minibatch gradients, no control variate")
    p.add_argument("--n-draws", type=int, default=4000)
    p.add_argument("--x-ref-from", type=Path, default=None,
                   help="folder with <data>/split_00/*.pt result files holding x_ref, e.g. the paper's")
    p.add_argument("--out", type=Path, default=Path("results/sg_pdmp/toy_bnns"))
    p.add_argument("--resume", action="store_true", help="skip runs whose file exists")
    p.add_argument("--smoke", action="store_true", help="tiny MAP fit, to check that the script runs")
    args = p.parse_args()
    batch_tag = None if args.batch_size == BATCH else args.batch_size

    for ds in args.datasets:
        data = toy_data(ds)
        sd = args.out / ds / "split_00"
        todo = [(s, h) for s in args.samplers for h in args.step_sizes
                if not (args.resume and (sd / f"{run_name(s, h, batch_tag)}.pt").exists())]
        if not todo:
            continue
        seed_all(42)
        module = FFN(LAYERS, ACT)
        bm = BNN.build(module, "gaussian", data["X_train"], data["y_train"],
                       prior_std(module, PRIOR_STD, PRIOR_STD), noise_std=data["noise_std"],
                       dtype=DTYPE, device=DEVICE)
        if args.x_ref_from is not None:
            x_ref = x_ref_from_results(args.x_ref_from / ds / "split_00")
        else:
            x_ref = fit_map(bm, 500 if args.smoke else 10_000)
        metric = 1.0 / (SIGMA_INV_SCALE * laplace_precision(bm, x_ref)) if args.precondition else None
        meta = dict(dataset=ds, layer_sizes=LAYERS, y_std=data["y_std"], x_ref=x_ref,
                    batch_size=args.batch_size, cv=not args.no_cv, precondition=args.precondition)
        for name, h in todo:
            print(f"\n[{ds}] {name} h={h:g} n={args.batch_size}")
            seed_all(42)
            sampler = make_sg_pdmp(name, bm, x_ref, step_size=h, batch_size=args.batch_size,
                                   cv=not args.no_cv, std_weight=PRIOR_STD, inclusion=INCLUSION,
                                   refresh_rate=args.refresh_rate, metric=metric)
            n_iters = args.n_iters or sampler.grad.iterations_for(args.grad_budget)
            out = run_and_resample(sampler, x_ref, n_out=args.n_draws, n_iters=n_iters,
                                   max_seconds=args.max_minutes * 60 if args.max_minutes else None)
            print("  " + summary(out))
            if out["diverged"]:
                out["samples"] = nan_samples(args.n_draws, bm.D)
            save(sd / f"{run_name(name, h, batch_tag)}.pt", **out, **meta, piw=INCLUSION)


if __name__ == "__main__":
    main()
