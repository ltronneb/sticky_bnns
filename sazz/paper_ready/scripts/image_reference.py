"""SGD baselines and MAP references for the image BNNs, an FFN [784, 256, 256,
10] and LeNet-5 on MNIST and ResNet-20 on CIFAR-10.

`sgd` trains a plain network with SGD (the SGD baseline, which for ResNet-20
also provides the BatchNorm statistics). `map` fits the MAP with Adam from the
network initialization (ResNet-20: from the SGD network, in eval mode), adds
the Laplace precision, then prunes and refits it (see utils/reference.py).
The pruned coordinates start frozen in image_bnn.py.

    python -m sazz.paper_ready.scripts.image_reference sgd --model lenet
    python -m sazz.paper_ready.scripts.image_reference map --model lenet
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from ..common import DEVICE, seed_all
from ..data import image_data
from ..models.bnn import BNN, freezable, prior_std
from ..models.networks import FFN, LeNet5, ResNet20
from ..utils.reference import accuracy, fit_map, laplace_precision, prune_and_refit

DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64
MODELS = {
    "ffn": dict(data="mnist", net=lambda: FFN([784, 256, 256, 10], "relu"), flatten=True,
                sgd=dict(epochs=40, lr=0.05, wd=0.0, augment=False),
                map=dict(steps=10_000, batch=None, n_fisher=1024, tol=0.03, refit=2000),
                piw=0.05, batch=1024, t0={"zigzag": 1e-5, "boomerang": 1e-3}, chunk=10_000),
    "lenet": dict(data="mnist", net=lambda: LeNet5("tanh"), flatten=False,
                  sgd=dict(epochs=40, lr=0.05, wd=0.0, augment=False),
                  map=dict(steps=10_000, batch=512, n_fisher=1024, tol=0.01, refit=200),
                  piw=0.05, batch=1024, t0={"zigzag": 1e-5, "boomerang": 1e-3}, chunk=10_000),
    "resnet20": dict(data="cifar10", net=lambda: ResNet20("relu"), flatten=False,
                     sgd=dict(epochs=180, lr=0.1, wd=1e-4, augment=True),
                     map=dict(steps=2000, batch=128, n_fisher=128, tol=0.1, refit=200),
                     piw=0.01, batch=128, t0={"zigzag": 1e-5, "boomerang": 1e-4}, chunk=2_000),
}
PRIOR_STD = 2.0


def build_target(model: str, data: dict, state_dict=None):
    module = MODELS[model]["net"]()
    if state_dict is not None:
        module.load_state_dict(state_dict)
    module.eval()  # BatchNorm uses its stored statistics
    bm = BNN.build(module, "categorical", data["X_train"], data["y_train"],
                   prior_std(module, PRIOR_STD, PRIOR_STD), dtype=DTYPE, device=DEVICE)
    return bm


def augment(X):
    """Random 32x32 crop of the 4-pixel reflect-padded image and random flip."""
    B, _, H, W = X.shape
    Xp = F.pad(X, [4, 4, 4, 4], mode="reflect")
    i, j = torch.randint(0, 9, (2, B), device=X.device)
    rows = (i[:, None] + torch.arange(H, device=X.device))[:, None, :, None]
    cols = (j[:, None] + torch.arange(W, device=X.device))[:, None, None, :]
    out = Xp[torch.arange(B, device=X.device)[:, None, None, None],
             torch.arange(X.shape[1], device=X.device)[None, :, None, None], rows, cols]
    flip = torch.rand(B, device=X.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def sgd(args, cfg, data):
    s = cfg["sgd"]
    module = cfg["net"]().to(dtype=DTYPE, device=DEVICE)
    opt = torch.optim.SGD(module.parameters(), lr=s["lr"], momentum=0.9, nesterov=True,
                          weight_decay=s["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, s["epochs"], s["lr"] * (0 if s["augment"] else 0.01))
    X, y = data["X_train"], data["y_train"]
    for epoch in range(s["epochs"]):
        module.train()
        perm = torch.randperm(X.shape[0], device=DEVICE)
        for k in range(0, X.shape[0], 128):
            idx = perm[k:k + 128]
            opt.zero_grad()
            F.cross_entropy(module(augment(X[idx]) if s["augment"] else X[idx]), y[idx]).backward()
            opt.step()
        sched.step()
        module.eval()
        with torch.no_grad():
            acc = float((module(data["X_test"][:2000]).argmax(-1) == data["y_test"][:2000]).float().mean())
        print(f"  epoch {epoch + 1}/{s['epochs']}  test acc {acc:.4f}")
    return {"state_dict": module.state_dict(),
            "beta": torch.cat([p.detach().flatten() for p in module.parameters()])}


def map_reference(args, cfg, data):
    m = cfg["map"]
    state = None
    if args.model == "resnet20":
        state = torch.load(args.out / "resnet20_sgd.pt", map_location=DEVICE)["state_dict"]
    bm = build_target(args.model, data, state)
    init = torch.cat([p.detach().flatten() for p in bm.module.parameters()])
    x = fit_map(bm, m["steps"], 2e-3, init=init, batch_size=m["batch"], cosine=True, log_every=500)
    Sigma_inv = laplace_precision(bm, x, m["n_fisher"])
    print(f"  MAP test acc {accuracy(bm, x, data['X_test'], data['y_test']):.4f}")
    mask = freezable(bm.module).to(DEVICE)
    x, pruned = prune_and_refit(bm, x, mask, data["X_val"], data["y_val"], tol=m["tol"],
                                refit_steps=m["refit"], batch_size=m["batch"])
    print(f"  pruned MAP test acc {accuracy(bm, x, data['X_test'], data['y_test']):.4f}")
    return {"x_ref": x, "Sigma_inv": Sigma_inv, "cold_start_mask": pruned,
            "module_state_dict": bm.module.state_dict()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["sgd", "map"])
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--out", type=Path, default=Path("results/images/references"))
    args = p.parse_args()
    cfg = MODELS[args.model]
    seed_all(0)
    data = image_data(cfg["data"], n_val=2000, flatten=cfg["flatten"], dtype=DTYPE, device=DEVICE)
    result = (sgd if args.command == "sgd" else map_reference)(args, cfg, data)
    path = args.out / f"{args.model}_{args.command}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in result.items()}, path)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
