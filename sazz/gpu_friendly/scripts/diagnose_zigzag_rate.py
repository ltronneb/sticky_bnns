"""
Minimal diagnostic: sample the true ZigZag rate lambda(t) at a handful of
points along one velocity direction from x_ref, plot it. No vmap, no grid
bound reconstruction -- just a plain Python loop, one gradient eval at a
time, so memory stays flat regardless of n_points.

Run with:
    python -m sazz.gpu_friendly.scripts.diagnose_zigzag_rate \\
        --checkpoint results/grid/mnist_cnn/split_00/grid_sticky_zigzag.pt \\
        --map-path results/maps/lenet_reference_N60000_steps10000.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from sazz.gpu_friendly.scripts.fast_mnist_cnn import (
    CNNConfig, load_mnist_subset, build_target, DATA_DIR,
    N_TRAIN, N_TEST, BASE_SEED, DTYPE, DEVICE,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                    default="results/grid/mnist_cnn/split_00/grid_sticky_zigzag.pt")
    p.add_argument("--map-path", type=str,
                    default="results/maps/lenet_reference_N60000_steps10000.pt")
    p.add_argument("--architecture", type=str, default="lenet5")
    p.add_argument("--t-max", type=float, default=0.002)
    p.add_argument("--n-points", type=int, default=20)
    p.add_argument("--gamma", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/grid/mnist_cnn/zigzag_rate_diagnostic.png")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    x_ref = ckpt["x_ref"].to(dtype=DTYPE, device=DEVICE)
    active_mask = ~ckpt["cold_start_mask"]
    D = x_ref.numel()
    print(f"  D={D}  active={int(active_mask.sum())}")

    print("Rebuilding target ...")
    data = load_mnist_subset(N_TRAIN, N_TEST, BASE_SEED, DATA_DIR, DTYPE, DEVICE, n_sweep=0)
    cfg = CNNConfig(activation="tanh", pool="avg")
    bm, _, _, _ = build_target(
        data, cfg, dtype=DTYPE, device=DEVICE,
        map_path=Path(args.map_path), architecture=args.architecture,
    )
    grad_target = torch.func.grad(bm.energy)
    print("  target ready.")

    v = torch.zeros(D, dtype=DTYPE, device=DEVICE)
    signs = torch.randint(0, 2, (D,), device=DEVICE, dtype=torch.int64)
    v[active_mask] = (2 * signs[active_mask] - 1).to(DTYPE)

    ts = [i * args.t_max / (args.n_points - 1) for i in range(args.n_points)]
    rates = []
    with torch.no_grad():
        for i, t in enumerate(ts):
            x_t = x_ref + t * v
            grad = grad_target(x_t)
            rate = torch.clamp(v * grad, min=0.0).sum().item() + D * args.gamma
            rates.append(rate)
            print(f"  t={t:.6f}  rate={rate:.4e}  ({i+1}/{len(ts)})", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, rates, marker="o")
    ax.set_xlabel("t")
    ax.set_ylabel("true rate lambda(t)")
    ax.set_title(f"ZigZag true rate along one direction from x_ref\n{Path(args.checkpoint).name}")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved plot -> {args.out}")

    print(f"\nrate(t=0)   = {rates[0]:.4e}")
    print(f"rate(t_max) = {rates[-1]:.4e}")
    print(f"max rate    = {max(rates):.4e}")


if __name__ == "__main__":
    main()
