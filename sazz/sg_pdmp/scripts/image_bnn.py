"""SG-PDMPs on the image BNNs of sazz.paper_ready (FFN and LeNet-5 on MNIST,
ResNet-20 on CIFAR-10), started from the pruned MAP of
paper_ready.scripts.image_reference, which is also the control variate centre.
SG-SZZ starts with the pruned weights frozen, like the paper's sticky samplers.

    python -m sazz.sg_pdmp.scripts.image_bnn --model lenet
    python -m sazz.sg_pdmp.scripts.image_bnn --model ffn --samplers sg_sticky_zigzag --step-sizes 1e-5

The minibatch is the paper's (1024 for MNIST, 128 for CIFAR-10). The per-point
gradients at the MAP do not fit in memory here, so an iteration costs two
minibatch gradients. Default budget 1e6 iterations.
"""

import argparse
from pathlib import Path

import torch

from sazz.paper_ready.common import DEVICE, save, seed_all
from sazz.paper_ready.data import image_data
from sazz.paper_ready.samplers import run_and_resample
from sazz.paper_ready.scripts.image_bnn import bma_accuracy
from sazz.paper_ready.scripts.image_reference import MODELS, PRIOR_STD, SMOKE_DATA, build_target

from ..common import SG_PDMPS, make_sg_pdmp, nan_samples, run_name, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--samplers", nargs="+", default=list(SG_PDMPS), choices=SG_PDMPS)
    p.add_argument("--step-sizes", nargs="+", type=float, default=[1e-4, 1e-5])
    p.add_argument("--batch-size", type=int, default=None, help="default: the paper's for the model")
    p.add_argument("--n-iters", type=int, default=1_000_000)
    p.add_argument("--max-minutes", type=float, default=None,
                   help="also stop each run after this much wall-clock time")
    p.add_argument("--precondition", action="store_true",
                   help="SG-BPS velocities N(0, Sigma) with the Laplace covariance of the reference")
    p.add_argument("--refresh-rate", type=float, default=1.0, help="SG-BPS refreshment rate")
    p.add_argument("--no-cv", action="store_true", help="plain minibatch gradients, no control variate")
    p.add_argument("--n-draws", type=int, default=4000)
    p.add_argument("--chunk-size", type=int, default=None, help="skeleton rows held on the device")
    p.add_argument("--references", type=Path, default=Path("results/images/references"))
    p.add_argument("--out", type=Path, default=Path("results/sg_pdmp/images"))
    p.add_argument("--resume", action="store_true", help="skip runs whose file exists")
    args = p.parse_args()

    cfg = MODELS[args.model]
    seed_all(0)
    ref = torch.load(args.references / f"{args.model}_map.pt", map_location=DEVICE)
    sizes = SMOKE_DATA if ref.get("smoke") else dict(n_val=2000)   # same data as the reference
    data = image_data(cfg["data"], flatten=cfg["flatten"], dtype=ref["x_ref"].dtype, device=DEVICE, **sizes)
    bm = build_target(args.model, data, ref["module_state_dict"])
    x_ref = ref["x_ref"].to(bm.X.dtype)
    metric = 1.0 / ref["Sigma_inv"].to(bm.X.dtype) if args.precondition else None
    default_batch = 32 if ref.get("smoke") else cfg["batch"]   # smoke: fast on a CPU
    batch = args.batch_size or default_batch
    batch_tag = None if batch == default_batch else batch

    for name in args.samplers:
        for h in args.step_sizes:
            path = args.out / args.model / f"{run_name(name, h, batch_tag)}.pt"
            if args.resume and path.exists():
                continue
            print(f"\n[{args.model}] {name} h={h:g} n={batch}")
            seed_all(42)
            sampler = make_sg_pdmp(name, bm, x_ref, step_size=h, batch_size=batch, cv=not args.no_cv,
                                   std_weight=PRIOR_STD, inclusion=cfg["piw"],
                                   cold_start=ref["cold_start_mask"], refresh_rate=args.refresh_rate,
                                   metric=metric)
            out = run_and_resample(sampler, x_ref, n_out=args.n_draws, n_iters=args.n_iters,
                                   chunk_size=args.chunk_size or cfg["chunk"],
                                   max_seconds=args.max_minutes * 60 if args.max_minutes else None)
            if out["diverged"]:
                out["samples"] = nan_samples(args.n_draws, bm.D)
                acc = float("nan")
            else:
                acc = bma_accuracy(bm, out["samples"], data["X_test"], data["y_test"])
            print(f"  {summary(out)}, test accuracy {acc:.4f}")
            save(path, **out, x_ref=x_ref, test_accuracy=acc, piw=cfg["piw"], batch_size=batch,
                 cv=not args.no_cv, precondition=args.precondition)


if __name__ == "__main__":
    main()
