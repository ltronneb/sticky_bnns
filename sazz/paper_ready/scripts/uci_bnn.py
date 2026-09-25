"""UCI regression BNNs (tanh, learned noise), five 90/10 splits per dataset.

The networks are small [50], medium [50, 50, 50] and large [256, 128, 64]. On
the small and medium networks the PDMPs get G = 1e6 gradient evaluations. On the
large one only the sticky PDMPs run, with K = 1e6 events and minibatch gradients.

    python -m sazz.paper_ready.scripts.uci_bnn --variant small --datasets boston
    python -m sazz.paper_ready.scripts.uci_bnn --variant medium --piw 0.1 \\
        --samplers sticky_zigzag sticky_boomerang
    python -m sazz.paper_ready.scripts.uci_bnn --variant large \\
        --samplers sticky_zigzag sticky_boomerang
    python -m sazz.paper_ready.scripts.uci_bnn --variant large --splits 0 --chains 0 1 2 3 \\
        --samplers sticky_zigzag sticky_boomerang        # convergence check

--piw with several values runs the prior inclusion sweep. --chains c starts
chain c from its own MAP (seed 90000 + 1000 split + c) with chain 0's Sigma_inv.
"""

import argparse
from pathlib import Path

import torch

from ..baselines import lbbnn, nuts
from ..common import DEVICE, DTYPE, PDMPS, make_pdmp, save, seed_all, summary
from ..data import uci_split
from ..models.bnn import BNN, prior_std
from ..models.networks import FFN
from ..samplers import run_and_resample
from ..utils.reference import fit_map, laplace_precision

DATASETS = ("boston", "energy", "concrete", "yacht")
NOISE_PRIOR_SCALE = {"boston": 0.3, "energy": 0.03, "concrete": 0.2, "yacht": 0.01}
MEDIUM_SIGMA_INV_SCALE = {"boston": 0.75, "energy": 0.5, "concrete": 10.0, "yacht": 0.5}
VARIANTS = {
    "small": dict(hidden=[50], piw=0.3, budget=dict(grad_budget=1_000_000), gamma=0.01,
                  refresh=1.0, t0={"zigzag": 2e-4, "boomerang": 3e-3}, batch=None,
                  alpha_violation=2.0, chunk=10_000),
    "medium": dict(hidden=[50, 50, 50], piw=0.3, budget=dict(grad_budget=1_000_000), gamma=0.01,
                   refresh=1.0, t0={"zigzag": 2e-4, "boomerang": 3e-3}, batch=None,
                   alpha_violation=2.0, chunk=10_000),
    "large": dict(hidden=[256, 128, 64], piw=0.05, budget=dict(n_events=1_000_000), gamma=1e-6,
                  refresh=500.0, t0={"zigzag": 1e-5, "boomerang": 1e-3}, batch=128,
                  alpha_violation=1.1, chunk=25_000),
}


def sigma_inv_scale(variant: str, ds: str) -> float:
    return {"small": 0.1, "medium": MEDIUM_SIGMA_INV_SCALE[ds], "large": 1.0}[variant]


def reference(bm, path: Path, seed: int):
    """MAP (Adam from N(0, I), 2e4 steps) and Laplace precision, cached at path."""
    if path.exists():
        r = torch.load(path, map_location=DEVICE)
        return r["x_ref"].to(DTYPE), r["Sigma_inv"].to(DTYPE)
    seed_all(seed)
    x_ref = fit_map(bm, 20_000)
    Sigma_inv = laplace_precision(bm, x_ref)
    save(path, x_ref=x_ref, Sigma_inv=Sigma_inv)
    return x_ref, Sigma_inv


def run_split(args, ds: str, split: int, chain):
    cfg = VARIANTS[args.variant]
    data = uci_split(ds, split, DTYPE, DEVICE)
    layers = [data["X_train"].shape[1], *cfg["hidden"], 1]
    module = FFN(layers, "tanh")
    bm = BNN.build(module, "gaussian", data["X_train"], data["y_train"], prior_std(module, 1.0, 1.0),
                   prior_sigma_scale=NOISE_PRIOR_SCALE[ds], dtype=DTYPE, device=DEVICE)
    base = args.out / args.variant / ds / f"split_{split:02d}"
    if chain is None:
        x_ref, Sigma_inv = reference(bm, base / "map.pt", 42 + split)
        seed = 42 + split
    else:
        x_ref, _ = reference(bm, base / "maps" / f"map_{chain}.pt", 90_000 + 1000 * split + chain)
        _, Sigma_inv = reference(bm, base / "maps" / "map_0.pt", 90_000 + 1000 * split)
        seed = 42 + split + 10_000 * chain
    scale = torch.full_like(Sigma_inv, sigma_inv_scale(args.variant, ds))
    scale[-1] = 1.0  # log_sigma keeps its prior-only precision
    meta = dict(dataset=ds, split=split, layer_sizes=layers, y_std=data["y_std"], x_ref=x_ref)

    for piw in args.piw or [cfg["piw"]]:
        tag = args.variant if piw == cfg["piw"] else f"{args.variant}_piw{piw:g}"
        sd = args.out / tag / ds / f"split_{split:02d}" / ("" if chain is None else f"chain_{chain}")
        for name in args.samplers:
            path = sd / f"{name}.pt"
            if (args.resume and path.exists()) or (piw != cfg["piw"] and not name.startswith("sticky")):
                continue
            print(f"\n[{tag} {ds} split {split}{'' if chain is None else f' chain {chain}'}] {name}")
            seed_all(seed)
            if name in PDMPS:
                sampler = make_pdmp(name, bm, x_ref, scale * Sigma_inv, gamma=cfg["gamma"],
                                    refresh_rate=cfg["refresh"], t_max_init=cfg["t0"][name.split("_")[-1]],
                                    std_weight=1.0, inclusion=piw, batch_size=cfg["batch"],
                                    alpha_violation=cfg["alpha_violation"])
                budget = {k: args.budget or v for k, v in cfg["budget"].items()}
                out = run_and_resample(sampler, x_ref, n_out=args.n_draws, chunk_size=cfg["chunk"],
                                       **budget)
                print("  " + summary(out))
                save(path, **out, **meta, piw=piw)
            elif name == "nuts":
                draws, sec, evals = nuts(data["X_train"], data["y_train"], layers, "tanh", 1.0, 1.0,
                                         prior_sigma_scale=NOISE_PRIOR_SCALE[ds], x_init=x_ref, seed=seed)
                save(path, samples=draws, elapsed_sec=sec, grad_evals=evals, **meta)
            elif name == "lbbnn":
                draws, sec, evals, alpha = lbbnn(data, layers, "tanh", 1.0, 1.0,
                                                 prior_sigma_scale=NOISE_PRIOR_SCALE[ds],
                                                 batch_size=cfg["batch"] or 10_000,
                                                 learn_model_prior=False, n_draws=args.n_draws,
                                                 seed=seed, device=DEVICE, dtype=DTYPE)
                save(path, samples=draws, elapsed_sec=sec, grad_evals=evals,
                     inclusion_probabilities=alpha, **meta)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=list(VARIANTS), default="small")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--samplers", nargs="+", default=[*PDMPS, "nuts", "lbbnn"])
    p.add_argument("--piw", nargs="+", type=float, default=None,
                   help="prior inclusion probabilities for the sticky PDMPs (default: variant's)")
    p.add_argument("--chains", nargs="+", type=int, default=None)
    p.add_argument("--n-draws", type=int, default=4000)
    p.add_argument("--budget", type=int, default=None,
                   help="override G (small, medium) or K (large), e.g. for a quick test")
    p.add_argument("--out", type=Path, default=Path("results/uci"))
    p.add_argument("--resume", action="store_true", help="skip runs whose file exists")
    args = p.parse_args()
    for ds in args.datasets:
        for split in args.splits:
            for chain in args.chains or [None]:
                run_split(args, ds, split, chain)


if __name__ == "__main__":
    main()
