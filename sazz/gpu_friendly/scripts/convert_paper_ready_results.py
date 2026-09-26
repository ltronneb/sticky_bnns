"""Copy UCI results written by sazz/paper_ready/scripts/uci_bnn.py into the
results/paper_v2 layout and key names the analysis notebooks read.

    <src>/medium/<ds>/split_XX/zigzag.pt           -> <dst>/deep_narrow/<ds>/split_XX/zigzag.pt
    <src>/medium_piw0.1/<ds>/split_XX/sticky_*.pt  -> <dst>/deep_narrow/piw_0.1/<ds>/split_XX/sticky_*.pt
    <src>/small_piw0.1/...                          -> <dst>/shallow_piw_0.1/...

Never overwrites an existing destination file.

    python -m sazz.gpu_friendly.scripts.convert_paper_ready_results \\
        --src results/paper_v2_medium --dst results/paper_v2

--ablation converts a PIW sweep instead (no folder suffix means the variant's
default PIW, 0.3). Sticky results go to the layout uci_sparsity_ablation.ipynb
reads, and skeleton files are moved (not loaded) next to the other skeletons.
Plain zigzag/boomerang results are skipped, they are reruns of the main ones.

    <src>/medium[_piwX]/<ds>/split_XX/sticky_*.pt
        -> <dst>/uci_sparsity_ablation/deep_narrow/<ds>/split_XX/piw_X/sticky_*.pt
    <src>/medium[_piwX]/<ds>/split_XX/*_skeleton.pt
        -> <dst>/skeletons/deep_narrow/piw_X/<ds>/split_XX/*_skeleton.pt

File names stay as paper_ready writes them (zigzag.pt, sticky_zigzag.pt, ...),
the paper_v2 convention since 26.09 (no grid_ prefix).

    python -m sazz.gpu_friendly.scripts.convert_paper_ready_results \\
        --src results/paper_v2_ablation --dst results/paper_v2 --ablation
"""

import argparse
import os
from pathlib import Path

import torch

VARIANT_DIR = {"small": "shallow", "medium": "deep_narrow", "large": "deep_wide_v2/deep_wide"}
DEFAULT_PIW = {"small": 0.3, "medium": 0.3, "large": 0.05}
NOISE_PRIOR_SCALE = {"boston": 0.3, "energy": 0.03, "concrete": 0.2, "yacht": 0.01}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, default=Path("results/paper_v2"))
    p.add_argument("--ablation", action="store_true", help="convert a PIW sweep, see above")
    args = p.parse_args()

    n_new = n_skip = 0
    for f in sorted(args.src.glob("*/*/split_*/*.pt")):
        tag, ds, split, name = f.parts[-4], f.parts[-3], f.parts[-2], f.stem
        if name == "map" or f.parent.name.startswith("chain_"):
            continue
        if name.endswith("_skeleton") and not args.ablation:
            continue
        variant, _, piw = tag.partition("_piw")
        base = VARIANT_DIR[variant]
        if args.ablation:
            w = f"{float(piw) if piw else DEFAULT_PIW[variant]:g}"
            if name.endswith("_skeleton"):
                out = args.dst / "skeletons" / base / f"piw_{w}" / ds / split / f"{name}.pt"
                if out.exists():
                    n_skip += 1
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                os.replace(f, out)
                n_new += 1
                print(f"  moved {f} -> {out}")
                continue
            if not name.startswith("sticky"):
                continue
            out_dir = args.dst / "uci_sparsity_ablation" / base / ds / split / f"piw_{w}"
        elif piw:  # v1 layout: shallow_piw_0.1/ next to shallow/, deep_narrow/piw_0.1/ inside it
            base = f"{base}_piw_{piw}" if variant == "small" else f"{base}/piw_{piw}"
        if not args.ablation:
            out_dir = args.dst / base / ds / split
        out_name = name
        out = out_dir / f"{out_name}.pt"
        if out.exists():
            n_skip += 1
            continue
        r = torch.load(f, map_location="cpu", weights_only=False)
        if args.ablation:
            r["prior_inclusion_weight"] = float(w)
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
