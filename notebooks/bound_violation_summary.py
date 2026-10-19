"""Bound violations, gradient evaluations and draws per experiment folder of
results/paper_v2 (the gpu_friendly grid_*.pt files). Run from the repo root."""
import glob, torch
from collections import defaultdict
agg = defaultdict(lambda: [0, 0, 0, 0])
for f in glob.glob("results/paper_v2/**/grid_*.pt", recursive=True):
    if "skeletons" in f:
        continue
    r = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
    key = f.split("/")[2] + ("/piw_0.1" if "piw_0.1/" in f and "shallow" not in f else "")
    ev = r.get("gradient_evals") or 0
    a = agg[key]
    a[0] += 1
    a[1] += r.get("bound_violations") or 0
    a[2] = ev if a[0] == 1 else min(a[2], ev)
    a[3] = max(a[3], r["samples"].shape[0])
for k, (n, viol, min_ev, s) in sorted(agg.items()):
    print(f"{k:28s} runs {n:3d}   bound violations {viol:4d}   min grad evals {min_ev:,}   S {s}")