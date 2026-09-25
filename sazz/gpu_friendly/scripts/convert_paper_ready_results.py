"""Copy UCI results written by sazz/paper_ready/scripts/uci_bnn.py into the
results/paper_v2 layout and key names the analysis notebooks read.

    <src>/medium/<ds>/split_XX/zigzag.pt           -> <dst>/deep_narrow/<ds>/split_XX/grid_zigzag.pt
    <src>/medium_piw0.1/<ds>/split_XX/sticky_*.pt  -> <dst>/deep_narrow/piw_0.1/<ds>/split_XX/grid_sticky_*.pt
    <src>/small_piw0.1/...                          -> <dst>/shallow_piw_0.1/...

Never overwrites an existing destination file.

    python -m sazz.gpu_friendly.scripts.convert_paper_ready_results \\
        --src results/paper_v2_medium --dst results/paper_v2
"""

import argparse
from pathlib import Path

import torch

VARIANT_DIR = {"small": "shallow", "medium": "deep_narrow", "large": "deep_wide_v2/deep_wide"}
NOISE_PRIOR_SCALE = {"boston": 0.3, "energy": 0.03, "concrete": 0.2, "yacht": 0.01}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, default=Path("results/paper_v2"))
    args = p.parse_args()

    n_new = n_skip = 0
    for f in sorted(args.src.glob("*/*/split_*/*.pt")):
        tag, ds, split, name = f.parts[-4], f.parts[-3], f.parts[-2], f.stem
        if name == "map" or f.parent.name.startswith("chain_"):
            continue
        variant, _, piw = tag.partition("_piw")
        base = VARIANT_DIR[variant]
        if piw:  # v1 layout: shallow_piw_0.1/ next to shallow/, deep_narrow/piw_0.1/ inside it
            base = f"{base}_piw_{piw}" if variant == "small" else f"{base}/piw_{piw}"
        out_dir = args.dst / base / ds / split
        out_name = name if name in ("nuts", "lbbnn") else f"grid_{name}"
        out = out_dir / f"{out_name}.pt"
        if out.exists():
            n_skip += 1
            continue
        r = torch.load(f, map_location="cpu", weights_only=False)
        r.update(sampler=out_name, split_id=int(split.split("_")[1]), activation="tanh",
                 learned_noise=True, prior_sigma_scale=NOISE_PRIOR_SCALE[ds],
                 gradient_evals=r.get("grad_evals"), source=str(f))
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(r, out)
        n_new += 1
        print(f"  {f} -> {out}")
    print(f"converted {n_new}, skipped {n_skip} existing")


if __name__ == "__main__":
    main()
