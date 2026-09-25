"""
One-off memory profiler: measures peak CUDA memory for a single
torch.func.grad(bm.energy) call on the real MNIST CNN target, at a range
of train-set sizes -- to find where full-batch (N=60_000) actually lands
relative to the GPU's ~23 GiB, and whether it's close/feasible or needs
mini-batching.

Usage (on the GPU machine):
    CUDA_VISIBLE_DEVICES=3 python -m sazz.gpu_friendly.scripts.diagnose_grad_memory \
        --map-path results/maps/lenet_reference_N60000_pruned_refit_N60k.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sazz.gpu_friendly.scripts.fast_mnist_cnn import (
    CNNConfig, DATA_DIR, DTYPE, DEVICE, load_mnist_subset, build_target,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-path", type=Path, required=True)
    parser.add_argument("--architecture", default="cnn")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-list", type=int, nargs="+",
                         default=[2_000, 5_000, 10_000, 20_000, 30_000, 45_000, 60_000])
    args = parser.parse_args()

    cfg = CNNConfig()

    for n_train in args.n_list:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        data = load_mnist_subset(n_train, 500, args.seed, args.data_dir, DTYPE, DEVICE, n_sweep=0)
        bm, x_ref, Sigma_inv, cold_start_mask = build_target(
            data, cfg, map_path=args.map_path, architecture=args.architecture,
        )

        grad_fn = torch.func.grad(bm.energy)
        try:
            g = grad_fn(x_ref)
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_allocated() / 2**30
            print(f"n_train={n_train:>7d}  peak_allocated={peak_gb:6.2f} GiB  OK")
        except torch.cuda.OutOfMemoryError as e:
            print(f"n_train={n_train:>7d}  OOM: {e}")
            break

        del bm, data, g
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
