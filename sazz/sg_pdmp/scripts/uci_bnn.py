"""SG-PDMPs on the UCI regression BNNs of sazz.paper_ready.scripts.uci_bnn, the same
networks, priors, splits and MAP. The samplers start at the MAP, which is also
the control variate centre.

    python -m sazz.sg_pdmp.scripts.uci_bnn --variant small --datasets boston --splits 0
    python -m sazz.sg_pdmp.scripts.uci_bnn --variant medium --step-sizes 1e-3 1e-4 1e-5
    python -m sazz.sg_pdmp.scripts.uci_bnn --variant small --x-ref-from results/paper_v2/shallow

Budget. On the small and medium networks the default is the paper's G = 1e6,
counted in full-data gradients (N per-point gradients each, the pass at the
MAP for the control variate included), so the SG-PDMPs cost the same as the
paper's PDMPs. --n-iters sets the number of iterations directly. The large
network defaults to 1e6 iterations. --max-minutes also stops a run after that
much wall-clock time, to compare at equal run time.

MAP. Read from --references (paper_ready layout, <variant>/<data>/split_XX/map.pt),
else taken from any result file with an x_ref under --x-ref-from/<data>/split_XX
(for example the paper's runs), else fitted exactly as in paper_ready and cached
under --out.

One result file per sampler and step size, <sampler>_h<step>.pt, with
_b<batch> appended when --batch-size is not the default. The files have the
fields of the paper's runs (samples, grad_evals, n_events, frozen_final, ...)
plus the SG settings. A diverged run has NaN samples and diverged = True.
"""

import argparse
from pathlib import Path

import torch

from sazz.paper_ready.common import DEVICE, DTYPE, save, seed_all
from sazz.paper_ready.data import uci_split
from sazz.paper_ready.models.bnn import BNN, prior_std
from sazz.paper_ready.models.networks import FFN
from sazz.paper_ready.samplers import run_and_resample
from sazz.paper_ready.scripts.uci_bnn import (DATASETS, NOISE_PRIOR_SCALE, VARIANTS, reference,
                                              sigma_inv_scale)
from sazz.paper_ready.utils.reference import laplace_precision

from ..common import SG_PDMPS, make_sg_pdmp, nan_samples, run_name, summary

BATCH = {"small": 32, "medium": 32, "large": 128}
BUDGET = {"small": dict(grad_budget=1_000_000), "medium": dict(grad_budget=1_000_000),
          "large": dict(n_iters=1_000_000)}


def x_ref_from_results(folder: Path):
    for f in sorted(folder.glob("*.pt")):
        r = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(r, dict) and "x_ref" in r:
            print(f"  x_ref from {f}")
            return r["x_ref"].to(DTYPE).to(DEVICE)
    raise FileNotFoundError(f"no result file with an x_ref in {folder}")


def get_reference(args, bm, ds: str, split: int):
    sub = Path(args.variant) / ds / f"split_{split:02d}"
    if (args.references / sub / "map.pt").exists():
        return reference(bm, args.references / sub / "map.pt")
    if args.x_ref_from is not None:
        x_ref = x_ref_from_results(args.x_ref_from / ds / f"split_{split:02d}")
        return x_ref, laplace_precision(bm, x_ref)
    return reference(bm, args.out / sub / "map.pt", steps=args.map_steps)


def run_split(args, ds: str, split: int):
    cfg = VARIANTS[args.variant]
    data = uci_split(ds, split, DTYPE, DEVICE)
    layers = [data["X_train"].shape[1], *cfg["hidden"], 1]
    seed_all(42 + split)  # before the network is built, as in paper_ready
    module = FFN(layers, "tanh")
    bm = BNN.build(module, "gaussian", data["X_train"], data["y_train"], prior_std(module, 1.0, 1.0),
                   prior_sigma_scale=NOISE_PRIOR_SCALE[ds], dtype=DTYPE, device=DEVICE)
    x_ref, Sigma_inv = get_reference(args, bm, ds, split)
    scale = torch.full_like(Sigma_inv, sigma_inv_scale(args.variant, ds))
    scale[-1] = 1.0
    metric = 1.0 / (scale * Sigma_inv) if args.precondition else None
    meta = dict(dataset=ds, split=split, layer_sizes=layers, y_std=data["y_std"], x_ref=x_ref,
                batch_size=args.batch_size, cv=not args.no_cv, precondition=args.precondition)
    budget = ({"n_iters": args.n_iters} if args.n_iters else
              {"grad_budget": args.grad_budget} if args.grad_budget else BUDGET[args.variant])

    for piw in args.piw or [cfg["piw"]]:
        tag = args.variant if piw == cfg["piw"] else f"{args.variant}_piw{piw:g}"
        sd = args.out / tag / ds / f"split_{split:02d}"
        for name in args.samplers:
            if piw != cfg["piw"] and name != "sg_sticky_zigzag":
                continue
            for h in args.step_sizes:
                path = sd / f"{run_name(name, h, args.batch_tag)}.pt"
                if args.resume and path.exists():
                    continue
                print(f"\n[{tag} {ds} split {split}] {name} h={h:g} n={args.batch_size}")
                seed_all(42 + split)
                sampler = make_sg_pdmp(name, bm, x_ref, step_size=h, batch_size=args.batch_size,
                                       cv=not args.no_cv, std_weight=1.0, inclusion=piw,
                                       refresh_rate=args.refresh_rate, metric=metric)
                n_iters = budget.get("n_iters") or sampler.grad.iterations_for(budget["grad_budget"])
                out = run_and_resample(sampler, x_ref, n_out=args.n_draws, n_iters=n_iters,
                                       chunk_size=cfg["chunk"], max_seconds=args.max_seconds)
                print("  " + summary(out))
                if out["diverged"]:
                    out["samples"] = nan_samples(args.n_draws, bm.D)
                save(path, **out, **meta, piw=piw)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=list(VARIANTS), default="small")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--samplers", nargs="+", default=list(SG_PDMPS), choices=SG_PDMPS)
    p.add_argument("--step-sizes", nargs="+", type=float, default=[1e-3, 1e-4])
    p.add_argument("--batch-size", type=int, default=None, help="data points per iteration (default: "
                   + ", ".join(f"{k} {v}" for k, v in BATCH.items()) + ")")
    p.add_argument("--grad-budget", type=float, default=None, help="full-data gradients per run")
    p.add_argument("--n-iters", type=int, default=None, help="iterations per run, overrides the budget")
    p.add_argument("--max-minutes", type=float, default=None,
                   help="also stop each run after this much wall-clock time")
    p.add_argument("--piw", nargs="+", type=float, default=None,
                   help="prior inclusion probabilities for SG-SZZ (default: variant's)")
    p.add_argument("--precondition", action="store_true",
                   help="SG-BPS velocities N(0, Sigma) with the Boomerang's Laplace covariance")
    p.add_argument("--refresh-rate", type=float, default=1.0, help="SG-BPS refreshment rate")
    p.add_argument("--no-cv", action="store_true", help="plain minibatch gradients, no control variate")
    p.add_argument("--n-draws", type=int, default=4000)
    p.add_argument("--references", type=Path, default=Path("results/uci"))
    p.add_argument("--x-ref-from", type=Path, default=None,
                   help="folder with <data>/split_XX/*.pt result files holding x_ref, e.g. the paper's")
    p.add_argument("--out", type=Path, default=Path("results/sg_pdmp/uci"))
    p.add_argument("--resume", action="store_true", help="skip runs whose file exists")
    p.add_argument("--smoke", action="store_true", help="tiny MAP fit, to check that the script runs")
    args = p.parse_args()
    args.batch_tag = args.batch_size if args.batch_size not in (None, BATCH[args.variant]) else None
    args.batch_size = args.batch_size or BATCH[args.variant]
    args.max_seconds = args.max_minutes * 60 if args.max_minutes else None
    args.map_steps = 500 if args.smoke else 20_000
    for ds in args.datasets:
        for split in args.splits:
            run_split(args, ds, split)


if __name__ == "__main__":
    main()
