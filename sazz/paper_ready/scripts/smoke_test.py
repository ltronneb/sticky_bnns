"""Smoke test. Runs every experiment script of the main results with tiny budgets
(--smoke), then the analysis on their output, and checks that every run wrote its
result file and that every metric is finite. Takes a few minutes on a laptop CPU.

    python -m sazz.paper_ready.scripts.smoke_test
    python -m sazz.paper_ready.scripts.smoke_test --skip-images   # no MNIST download
    python -m sazz.paper_ready.scripts.smoke_test --with-cifar    # also ResNet-20, slow on a CPU

The numbers it prints are not meaningful, the budgets are far too small.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PDMPS = ["zigzag", "sticky_zigzag", "boomerang", "sticky_boomerang"]


def run(name: str, *args: str) -> str:
    cmd = [sys.executable, "-m", f"sazz.paper_ready.scripts.{name}", *args]
    print(f"\n$ {' '.join(cmd[1:])}", flush=True)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-3000:], res.stderr[-3000:], sep="\n")
        sys.exit(f"FAILED: {name} {' '.join(args)}")
    print(f"  ok, {time.perf_counter() - t0:.0f} s")
    return res.stdout


def expect(*paths: Path):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        sys.exit("FAILED, missing result files:\n  " + "\n  ".join(missing))


def check_finite(table_text: str, what: str):
    if re.search(r"(?<![\w.])-?(nan|inf)\b", table_text, re.IGNORECASE):
        print(table_text)
        sys.exit(f"FAILED: non-finite metric in the {what} analysis")
    print(table_text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("results/smoke_test"))
    p.add_argument("--skip-images", action="store_true")
    p.add_argument("--with-cifar", action="store_true", help="also ResNet-20 on CIFAR-10")
    args = p.parse_args()
    if args.out.exists():
        shutil.rmtree(args.out)
    t0 = time.perf_counter()

    # toys, all samplers on one data set
    toy = args.out / "toy_bnns"
    run("toy_bnn", "--datasets", "hernandez", "--grad-budget", "2000", "--n-draws", "200", "--smoke",
        "--out", str(toy))
    expect(*[toy / "hernandez" / "split_00" / f"{s}.pt" for s in PDMPS + ["nuts", "lbbnn"]])

    # UCI, small network with all samplers, medium with the sticky samplers at w = 0.1
    uci = args.out / "uci"
    run("uci_bnn", "--variant", "small", "--datasets", "boston", "--splits", "0", "--budget", "2000",
        "--n-draws", "200", "--smoke", "--out", str(uci))
    run("uci_bnn", "--variant", "medium", "--datasets", "boston", "--splits", "0", "--budget", "2000",
        "--n-draws", "200", "--smoke", "--piw", "0.1", "--samplers", "sticky_zigzag", "sticky_boomerang",
        "--out", str(uci))
    expect(*[uci / "small" / "boston" / "split_00" / f"{s}.pt" for s in PDMPS + ["nuts", "lbbnn"]],
           *[uci / "medium_piw0.1" / "boston" / "split_00" / f"{s}.pt" for s in ("sticky_zigzag", "sticky_boomerang")])

    check_finite(run("analyse", "toy", "--results", str(toy)), "toy")
    check_finite(run("analyse", "uci", "--results", str(uci)), "UCI")

    # images, SGD baseline, pruned MAP and both sticky samplers for every model
    if not args.skip_images:
        img, refs = args.out / "images", args.out / "images" / "references"
        for m in ("ffn", "lenet") + (("resnet20",) if args.with_cifar else ()):
            run("image_reference", "sgd", "--model", m, "--smoke", "--out", str(refs))
            run("image_reference", "map", "--model", m, "--smoke", "--out", str(refs))
            run("image_bnn", "--model", m, "--n-events", "30" if m == "resnet20" else "100", "--n-draws", "50",
                "--chunk-size", "200",
                "--references", str(refs), "--out", str(img))
            expect(refs / f"{m}_sgd.pt", refs / f"{m}_map.pt",
                   *[img / m / f"{s}.pt" for s in ("sticky_zigzag", "sticky_boomerang")])
        check_finite(run("analyse", "images", "--results", str(img), "--n-draws", "50"), "image")

    print(f"\nSMOKE TEST PASSED in {(time.perf_counter() - t0) / 60:.1f} min, outputs in {args.out}")


if __name__ == "__main__":
    main()
