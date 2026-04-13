"""
Run ONE (sampler, target, seed) combination and save samples + diagnostics.

Pipeline:
    1. build target          (target block in config)
    2. build kappa if needed (kappa block, for sticky samplers)
    3. build sampler         (sampler block)
    4. preprocess            (preprocess block)
    5. sample + save

Usage:
    python run_experiment.py --config configs/logreg_sparse_sticky.yaml
    python run_experiment.py --config configs/bnn_sticky.yaml --seed 7
"""
import argparse
import time
import pickle
from pathlib import Path
import numpy as np
import yaml

from targets import build_target
from samplers import build_sampler, apply_preprocess, build_kappa


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    return ap.parse_args()


def load_config(path, args):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:    cfg["seed"] = args.seed
    if args.N is not None:       cfg["N"] = args.N
    if args.out_dir is not None: cfg["out_dir"] = args.out_dir
    cfg.setdefault("out_dir", "results")
    cfg.setdefault("seed", 0)
    return cfg


def run(cfg, tag=""):
    np.random.seed(cfg["seed"])

    # 1. Target
    target = build_target(cfg["target"]["name"], **cfg["target"].get("kwargs", {}))

    # 2. Kappa (only if the config has a kappa block; non-sticky samplers ignore it)
    sampler_kwargs = dict(cfg["sampler"].get("kwargs", {}))
    if "kappa" in cfg:
        sampler_kwargs["kappa"] = build_kappa(cfg["kappa"], target)

    # 3. Sampler
    sampler = build_sampler(cfg["sampler"]["name"], target, N=cfg["N"], **sampler_kwargs)

    # 4. Preprocess
    apply_preprocess(sampler, target, cfg["preprocess"])

    # 5. Sample
    t0 = time.perf_counter()
    sampler.sample_auto(diagnostics=False)
    wall = time.perf_counter() - t0

    # Pack artifact
    artifact = {
        "config": cfg,
        "tag": tag,
        "wall_seconds": wall,
        "samples": {
            "position": np.asarray(sampler.Position),
            "velocity": np.asarray(sampler.Velocity),
            "times": np.asarray(getattr(sampler, "Times", [])),
        },
        "target": {
            "name": target.name,
            "D": target.D,
            "task_type": target.task_type,
            "true_params": target.true_params,
            "data": target.data,
            "meta": target.meta,
        },
        "diagnostics": _collect_diagnostics(sampler),
    }

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{target.name}__{cfg['sampler']['name']}__seed{cfg['seed']}"
    if tag:
        stem += f"__{tag}"
    out_path = out_dir / f"{stem}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved -> {out_path}  ({wall:.2f}s)")
    return out_path


def _collect_diagnostics(sampler):
    attrs = [
        "n_grad_evals", "event_counts", "event_horizons",
        "accept_count", "reject_count", "rate_evals_per_brent",
        "freeze_count", "thaw_count", "max_frozen",
    ]
    return {a: getattr(sampler, a, None) for a in attrs}


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config, args)
    run(cfg, tag=args.tag)
