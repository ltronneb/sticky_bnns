"""LBBNN tuning on the UCI data sets (Status step 15).

One run = one config on one (network, dataset, split), with exactly the call of
sazz/paper_ready/scripts/uci_bnn.py (same priors, noise prior, full batch, fixed
Beta-Binomial(1, 1) model prior, seed 42 + split) plus the config's overrides.
Each run writes one JSON line to results/lbbnn_tuning/<config>__<net>__<ds>__<split>.json.

    python notes/lbbnn_tuning/tune.py run base small energy 0
    python notes/lbbnn_tuning/tune.py jobs > jobs.txt      # every (config, net, ds, split)
    python notes/lbbnn_tuning/tune.py summary
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path("results/lbbnn_tuning")
CONFIGS = {
    "base":            {},
    "alpha_hi":        dict(lam_init=(2.0, 3.0)),
    "long":            dict(epochs=50_000),
    "long_cos":        dict(epochs=50_000, cosine=True),
    "hi_long_cos":     dict(lam_init=(2.0, 3.0), epochs=50_000, cosine=True),
    "hi_long_cos_t05": dict(lam_init=(2.0, 3.0), epochs=50_000, cosine=True, temper=0.5),
    "hi_long_cos_rho": dict(lam_init=(2.0, 3.0), epochs=50_000, cosine=True, rho_init=(-3.0, -2.0)),
    "hi_long_cos_mc4": dict(lam_init=(2.0, 3.0), epochs=50_000, cosine=True, mc_samples=4),
    # round 2 (energy, yacht): inclusion probabilities started near 1, learned model prior
    "a98_long_cos":    dict(lam_init=(3.5, 4.5), epochs=50_000, cosine=True),
    "hi_long_cos_mp":  dict(lam_init=(2.0, 3.0), epochs=50_000, cosine=True, learn_model_prior=True),
    "a98_long_cos_mp": dict(lam_init=(3.5, 4.5), epochs=50_000, cosine=True, learn_model_prior=True),
    # round 3, the original author's tempering (KL x tau with sigma = 1 on standardized y), which is
    # the same as fixing the noise variance at tau. Same optimiser settings as a98_long_cos.
    **{f"tau{t:g}": dict(lam_init=(3.5, 4.5), epochs=50_000, cosine=True, noise_var=t)
       for t in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)},
}
NETS = {"small": [50], "medium": [50, 50, 50]}
DATASETS = ["energy", "yacht", "concrete", "boston"]
SPLITS = [0, 1]


def run(config, net, ds, split):
    import torch
    from sazz.paper_ready.baselines import lbbnn
    from sazz.paper_ready.data import uci_split
    from sazz.paper_ready.metrics import regression_metrics, sparsity
    from sazz.paper_ready.models.bnn import freezable
    from sazz.paper_ready.models.networks import FFN
    from sazz.paper_ready.scripts.analyse import ffn_predict
    from sazz.paper_ready.scripts.uci_bnn import NOISE_PRIOR_SCALE

    path = OUT / f"{config}__{net}__{ds}__{split}.json"
    if path.exists():
        return
    torch.set_num_threads(1)
    data = uci_split(ds, split)
    layers = [data["X_train"].shape[1], *NETS[net], 1]
    kw = dict(CONFIGS[config])
    learn_mp = kw.pop("learn_model_prior", False)
    noise_var = kw.pop("noise_var", None)
    noise_std = None if noise_var is None else noise_var ** 0.5
    t0 = time.perf_counter()
    draws, sec, evals, alpha = lbbnn(data, layers, "tanh", 1.0, 1.0, noise_std=noise_std,
                                     prior_sigma_scale=NOISE_PRIOR_SCALE[ds], batch_size=10_000,
                                     learn_model_prior=learn_mp, n_draws=4000, seed=42 + split, **kw)
    z = draws.double()
    beta = z if noise_std is not None else z[:, :-1]
    sig = (torch.full((z.shape[0],), noise_std, dtype=torch.float64) if noise_std is not None
           else z[:, -1].exp())
    preds = ffn_predict(beta, layers, data["X_test"].double())
    m = regression_metrics(data["y_test"].double(), preds, sig, data["y_std"])
    row = dict(config=config, net=net, dataset=ds, split=split, **m,
               sparsity=sparsity(beta, freezable(FFN(layers))), sigma=float(sig[0]),
               seconds=time.perf_counter() - t0, settings={k: str(v) for k, v in CONFIGS[config].items()})
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))
    print(json.dumps(row))


def summary():
    import pandas as pd
    rows = [json.loads(p.read_text()) for p in OUT.glob("*.json")]
    if not rows:
        print("no results yet")
        return
    df = pd.DataFrame(rows)
    g = df.groupby(["net", "dataset", "config"])[["RMSE", "NLL", "CRPS", "Cov", "sparsity", "seconds"]].mean()
    n = df.groupby(["net", "dataset", "config"]).size().rename("splits")
    print(pd.concat([n, g], axis=1).round(3).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif cmd == "jobs":
        for c in CONFIGS:
            for net in NETS:
                for ds in DATASETS:
                    for s in SPLITS:
                        print(c, net, ds, s)
    elif cmd == "summary":
        summary()
