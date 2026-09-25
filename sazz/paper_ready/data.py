"""Datasets for the experiments, 1-D toy regression, UCI regression, MNIST and CIFAR-10."""

from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path("datasets")

# ------------------------------------------------------------------ toys
TOY_SEED = 42


def _toy(f, x_train, noise, rng):
    y_train = f(x_train) + rng.normal(0, noise, size=x_train.shape)
    x_test = np.linspace(-4.0, 4.0, 300)
    y_clean = f(x_test)
    y_test = y_clean + rng.normal(0, noise, size=300)
    xm, xs, ym, ys = x_train.mean(), x_train.std(), y_train.mean(), y_train.std()
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    return {"X_train": t((x_train - xm) / xs).unsqueeze(1), "y_train": t((y_train - ym) / ys),
            "X_test": t((x_test - xm) / xs).unsqueeze(1), "y_test": t((y_test - ym) / ys),
            "y_test_clean": t((y_clean - ym) / ys), "y_std": float(ys), "noise_std": noise / ys}


def toy_data(name: str) -> dict:
    """The four 1-D benchmarks, standardized. noise_std is the true noise."""
    rng = np.random.default_rng(TOY_SEED)
    u = rng.uniform
    if name == "hernandez":
        return _toy(lambda x: x ** 3, u(-4, 4, 20), 3.0, rng)
    if name == "gap":
        return _toy(lambda x: np.sin(1.5 * x) + 0.3 * x,
                    np.concatenate([u(-3, -1.5, 30), u(1.5, 3, 30)]), 0.1, rng)
    if name == "sharp":
        return _toy(lambda x: 0.3 * np.tanh(x) + 1.5 * np.exp(-((x - 0.5) ** 2) / 0.05),
                    np.concatenate([u(-3, 3, 40), u(0, 1, 20)]), 0.1, rng)
    if name == "multiscale":
        return _toy(lambda x: np.sin(0.5 * x) + 0.3 * np.sin(4 * x), u(-3, 3, 80), 0.1, rng)
    raise ValueError(name)


# ------------------------------------------------------------------- UCI
def uci_raw(name: str):
    import pandas as pd
    if name == "boston":
        from sklearn.datasets import fetch_openml
        b = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
        return b.data.values.astype(float), b.target.values.astype(float)
    if name == "energy":
        df = pd.read_excel(DATA_DIR / "energy_data.xlsx")
        return df.iloc[:, :8].values.astype(float), df.iloc[:, 8].values.astype(float)
    if name == "yacht":
        df = pd.read_csv(DATA_DIR / "yacht_hydrodynamics.data", sep=r"\s+", header=None)
        return df.iloc[:, :6].values.astype(float), df.iloc[:, 6].values.astype(float)
    if name == "concrete":
        df = pd.read_excel(DATA_DIR / "concrete+compressive+strength 3/Concrete_Data.xls",
                           engine="xlrd")
        return df.iloc[:, :8].values.astype(float), df.iloc[:, 8].values.astype(float)
    raise ValueError(name)


def uci_split(name: str, split: int, dtype=torch.float64, device="cpu", seed: int = 42) -> dict:
    """90/10 train/test split with seed + split, standardized on the train set."""
    from sklearn.model_selection import train_test_split
    X, y = uci_raw(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=seed + split)
    xm, xs = Xtr.mean(0), Xtr.std(0)
    xs[xs == 0] = 1.0
    ym, ys = ytr.mean(), ytr.std()
    t = lambda a: torch.tensor(a, dtype=dtype, device=device)
    return {"X_train": t((Xtr - xm) / xs), "y_train": t((ytr - ym) / ys),
            "X_test": t((Xte - xm) / xs), "y_test": t((yte - ym) / ys), "y_std": float(ys)}


# ---------------------------------------------------------------- images
MNIST_NORM = ((0.1307,), (0.3081,))
CIFAR_NORM = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


def image_data(name: str, n_train=None, n_test=None, n_val: int = 0, seed: int = 42,
               flatten: bool = False, dtype=torch.float32, device="cpu") -> dict:
    """name in {"mnist", "cifar10"}. Random subsets of the train and test sets
    (None = all). X_val/y_val (for pruning) come from the test pool, disjoint
    from X_test."""
    from torchvision import datasets, transforms
    ds_cls, norm = (datasets.MNIST, MNIST_NORM) if name == "mnist" else (datasets.CIFAR10, CIFAR_NORM)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*norm)])
    train = ds_cls(DATA_DIR, train=True, download=True, transform=tf)
    test = ds_cls(DATA_DIR, train=False, download=True, transform=tf)
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(train), size=n_train or len(train), replace=False)
    te_pool = rng.permutation(len(test))
    n_test = n_test or len(test) - n_val

    def stack(ds, idx):
        X = torch.stack([ds[int(i)][0] for i in idx]).to(dtype=dtype, device=device)
        y = torch.tensor([ds[int(i)][1] for i in idx], dtype=torch.long, device=device)
        return (X.flatten(1) if flatten else X), y

    out = dict(zip(["X_train", "y_train"], stack(train, tr_idx)))
    out.update(zip(["X_test", "y_test"], stack(test, te_pool[:n_test])))
    if n_val:
        out.update(zip(["X_val", "y_val"], stack(test, te_pool[n_test:n_test + n_val])))
    return out
