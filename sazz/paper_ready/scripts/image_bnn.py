"""Sticky ZigZag and sticky Boomerang on the image BNNs, started from the
pruned MAP of image_reference.py with the pruned weights frozen. K = 1e6
events with minibatch gradients.

    python -m sazz.paper_ready.scripts.image_bnn --model lenet
"""

import argparse
from pathlib import Path

import torch

from ..common import DEVICE, make_pdmp, save, seed_all, summary
from ..data import image_data
from ..samplers import run_and_resample
from .image_reference import MODELS, PRIOR_STD, build_target


@torch.no_grad()
def bma_accuracy(bm, samples, X, y, n: int = 300) -> float:
    idx = torch.randperm(samples.shape[0])[:n]
    probs = sum(torch.softmax(bm.predict(samples[i], X), -1) for i in idx) / len(idx)
    return float((probs.argmax(-1) == y).float().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--samplers", nargs="+", default=["sticky_zigzag", "sticky_boomerang"])
    p.add_argument("--n-events", type=int, default=1_000_000)
    p.add_argument("--n-draws", type=int, default=4000)
    p.add_argument("--chunk-size", type=int, default=None, help="skeleton rows held on the device")
    p.add_argument("--references", type=Path, default=Path("results/images/references"))
    p.add_argument("--out", type=Path, default=Path("results/images"))
    p.add_argument("--resume", action="store_true", help="skip runs whose file exists")
    args = p.parse_args()

    cfg = MODELS[args.model]
    seed_all(0)
    ref = torch.load(args.references / f"{args.model}_map.pt", map_location=DEVICE)
    data = image_data(cfg["data"], n_val=2000, flatten=cfg["flatten"], dtype=ref["x_ref"].dtype,
                      device=DEVICE)
    bm = build_target(args.model, data, ref["module_state_dict"])
    x_ref, Sigma_inv = ref["x_ref"].to(bm.X.dtype), ref["Sigma_inv"].to(bm.X.dtype)

    for name in args.samplers:
        path = args.out / args.model / f"{name}.pt"
        if args.resume and path.exists():
            continue
        print(f"\n[{args.model}] {name}")
        seed_all(42)
        sampler = make_pdmp(name, bm, x_ref, Sigma_inv, t_max_init=cfg["t0"][name.split("_")[-1]],
                            gamma=1e-6, refresh_rate=1.0, std_weight=PRIOR_STD, inclusion=cfg["piw"],
                            cold_start=ref["cold_start_mask"], batch_size=cfg["batch"],
                            alpha=1.02, alpha_violation=1.1)
        out = run_and_resample(sampler, x_ref, n_out=args.n_draws, n_events=args.n_events,
                               chunk_size=args.chunk_size or cfg["chunk"])
        acc = bma_accuracy(bm, out["samples"], data["X_test"], data["y_test"])
        print(f"  {summary(out)}, test accuracy {acc:.4f}")
        save(path, **out, x_ref=x_ref, test_accuracy=acc, piw=cfg["piw"])


if __name__ == "__main__":
    main()
