"""1-D toy regression with a BNN with one hidden layer of 100 tanh units and known
noise. The PDMPs get G = 2e5 gradient evaluations, NUTS 4 x (1000 + 1000),
LBBNN 2e4 epochs.

    python -m sazz.paper_ready.scripts.toy_bnn --out results/toy_bnns
"""

import argparse
from pathlib import Path

import torch

from ..baselines import lbbnn, nuts
from ..common import DEVICE, DTYPE, PDMPS, make_pdmp, save, seed_all, summary
from ..data import toy_data
from ..models.bnn import BNN, prior_std
from ..models.networks import FFN
from ..samplers import run_and_resample
from ..utils.reference import fit_map, laplace_precision

DATASETS = ("hernandez", "gap", "sharp", "multiscale")
LAYERS, ACT, PRIOR_STD = [1, 100, 1], "tanh", 3.0
INCLUSION = 0.1          # sticky prior inclusion probability
SIGMA_INV_SCALE = 0.1    # Boomerang reference precision = scale * Laplace precision
T_MAX_INIT = {"zigzag": 2e-4, "boomerang": 3e-3}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--samplers", nargs="+", default=[*PDMPS, "nuts", "lbbnn"])
    p.add_argument("--grad-budget", type=int, default=200_000)
    p.add_argument("--n-draws", type=int, default=4000)
    p.add_argument("--save-skeleton", action="store_true", help="also save the PDMP skeletons")
    p.add_argument("--out", type=Path, default=Path("results/toy_bnns"))
    p.add_argument("--resume", action="store_true", help="skip runs whose file exists")
    args = p.parse_args()

    for ds in args.datasets:
        data = toy_data(ds)
        sd = args.out / ds / "split_00"
        todo = [s for s in args.samplers if not (args.resume and (sd / f"{s}.pt").exists())]
        if not todo:
            continue
        seed_all(42)
        module = FFN(LAYERS, ACT)
        bm = BNN.build(module, "gaussian", data["X_train"], data["y_train"],
                       prior_std(module, PRIOR_STD, PRIOR_STD), noise_std=data["noise_std"],
                       dtype=DTYPE, device=DEVICE)
        x_ref = fit_map(bm, 10_000)
        Sigma_inv = laplace_precision(bm, x_ref)
        for name in todo:
            print(f"\n[{ds}] {name}")
            seed_all(42)
            meta = dict(dataset=ds, layer_sizes=LAYERS, y_std=data["y_std"], x_ref=x_ref)
            if name in PDMPS:
                sampler = make_pdmp(name, bm, x_ref, SIGMA_INV_SCALE * Sigma_inv,
                                    t_max_init=T_MAX_INIT[name.split("_")[-1]],
                                    std_weight=PRIOR_STD, inclusion=INCLUSION)
                out = run_and_resample(sampler, x_ref, n_out=args.n_draws,
                                       grad_budget=args.grad_budget, keep_skeleton=args.save_skeleton)
                print("  " + summary(out))
                if args.save_skeleton:
                    save(args.out / "skeletons" / ds / f"{name}.pt", **out.pop("skeleton"))
                save(sd / f"{name}.pt", **out, **meta)
            elif name == "nuts":
                draws, sec, evals = nuts(data["X_train"], data["y_train"], LAYERS, ACT, PRIOR_STD,
                                         PRIOR_STD, noise_std=data["noise_std"])
                save(sd / "nuts.pt", samples=draws, elapsed_sec=sec, grad_evals=evals, **meta)
            elif name == "lbbnn":
                draws, sec, evals, alpha = lbbnn(data, LAYERS, ACT, PRIOR_STD, PRIOR_STD,
                                                 noise_std=data["noise_std"], epochs=20_000,
                                                 temper=0.5, n_draws=args.n_draws)
                save(sd / "lbbnn.pt", samples=draws, elapsed_sec=sec, grad_evals=evals,
                     inclusion_probabilities=alpha, **meta)


if __name__ == "__main__":
    main()
