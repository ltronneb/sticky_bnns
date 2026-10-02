# Sticky PDMP samplers for Bayesian neural networks

Code for the experiments in the paper. The samplers are the ZigZag and Boomerang
processes and their sticky versions, which put a spike-and-slab prior on the
network weights by freezing a weight at zero when it hits zero.

## Layout

```
samplers/   event loop (pdmp.py), ZigZag, Boomerang, the Poisson bound (bound.py)
            and uniform-in-time resampling of the trajectory (resample.py)
models/     BNN target with a pure energy function (bnn.py), networks
utils/      MAP, Laplace reference precision, pruning (reference.py)
baselines.py, lbbnn.py    NUTS and LBBNN
metrics.py                RMSE, NLL, CRPS, coverage, sparsity, Acc, ECE, Brier
data.py, common.py        datasets, sampler construction, result files
scripts/    one script per experiment, analyse.py for the tables, smoke_test.py
```

Events are simulated by thinning a Poisson process with one tangent-line bound
per window of length t_max, which adapts during the run. Every PDMP run has a
budget of K events or G gradient evaluations. Posterior samples are S = 4000
points drawn uniformly in time over the trajectory after discarding the first
20% of the time.

## Setup

```
pip install -r sazz/paper_ready/requirements.txt
```

Run every command from the directory that contains `sazz/`. A GPU is used when
present (float32), otherwise the CPU (float64). Add `--resume` to skip finished
runs.

Data. The toy data sets are generated in `data.py`. Boston is fetched from
OpenML, MNIST and CIFAR-10 are downloaded by torchvision to `datasets/`. The
other UCI data sets come from the UCI Machine Learning Repository and go in
`datasets/` under these names:

```
datasets/energy_data.xlsx                                        Energy efficiency (ENB2012_data.xlsx)
datasets/yacht_hydrodynamics.data                                Yacht hydrodynamics
datasets/concrete+compressive+strength 3/Concrete_Data.xls       Concrete compressive strength
```

## Smoke test

Runs the scripts of the main results with tiny budgets (`--smoke`): toys, UCI
small and medium, and the FFN and LeNet-5 on MNIST. Then it runs the analysis
and checks that all result files exist and all metrics are finite. About 2
minutes on a laptop CPU. ResNet-20 on CIFAR-10 is slow on a CPU and optional.

```
python -m sazz.paper_ready.scripts.smoke_test
python -m sazz.paper_ready.scripts.smoke_test --skip-images   # no MNIST download
python -m sazz.paper_ready.scripts.smoke_test --with-cifar    # also ResNet-20
```

## Main results

Toy regression, four data sets, G = 2e5 gradient evaluations per PDMP, NUTS
4 x (1000 + 1000), LBBNN 1.5e4 epochs.

```
python -m sazz.paper_ready.scripts.toy_bnn --out results/toy_bnns
python -m sazz.paper_ready.scripts.analyse toy --results results/toy_bnns
```

UCI regression, four data sets and five 90/10 splits, the small [50] and medium
[50, 50, 50] networks, G = 1e6 per PDMP. The sticky samplers run with prior
inclusion probabilities w = 0.3 (default) and w = 0.1.

```
python -m sazz.paper_ready.scripts.uci_bnn --variant small --out results/uci
python -m sazz.paper_ready.scripts.uci_bnn --variant medium --out results/uci
python -m sazz.paper_ready.scripts.uci_bnn --variant small --piw 0.1 --samplers sticky_zigzag sticky_boomerang --out results/uci
python -m sazz.paper_ready.scripts.uci_bnn --variant medium --piw 0.1 --samplers sticky_zigzag sticky_boomerang --out results/uci
python -m sazz.paper_ready.scripts.analyse uci --results results/uci
```

Images, an FFN and LeNet-5 on MNIST and ResNet-20 on CIFAR-10, K = 1e6 events
with minibatch gradients. For each model, first the SGD baseline (for ResNet-20
also the BatchNorm statistics), then the pruned MAP reference, then the sticky
samplers.

```
for m in ffn lenet resnet20; do
  python -m sazz.paper_ready.scripts.image_reference sgd --model $m
  python -m sazz.paper_ready.scripts.image_reference map --model $m
  python -m sazz.paper_ready.scripts.image_bnn --model $m
done
python -m sazz.paper_ready.scripts.analyse images --results results/images
```

`analyse` prints one table per data set (mean and standard error over the
splits for UCI), `--latex` also as LaTeX. The metrics are defined in `metrics.py`.

## Further experiments

Prior inclusion sweep and convergence chains (chain c starts from its own MAP
and Laplace precision).

```
python -m sazz.paper_ready.scripts.uci_bnn --variant small --datasets boston --splits 0 \
    --piw 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --samplers sticky_zigzag sticky_boomerang
python -m sazz.paper_ready.scripts.uci_bnn --variant medium --datasets boston --splits 0 \
    --chains 1 2 3 --budget 10000000
```

Sparse linear regression against the exact posterior (collapsed Gibbs), K = 2e5.

```
python -m sazz.paper_ready.scripts.linreg_exactness run
python -m sazz.paper_ready.scripts.linreg_exactness summary
```

Banana, Boomerang over refresh rates and reference scales, K = 5e4.

```
python -m sazz.paper_ready.scripts.banana
```

## Notes

Each run saves a `.pt` file with the posterior samples (`samples`), the
gradient evaluations, the number of events, the number of bound violations and
the frozen coordinates at the end of the run.

Gradient count. The bounce of an accepted event reuses the gradient already
computed at the proposed time. The paper's runs recomputed it, one extra
gradient per bounce, so at the same budget G these scripts simulate somewhat
more events (about 1.2 to 1.3 times for ZigZag). The paths are otherwise the
same.

Compute. The paper's runs used an Apple M3 laptop (8 cores, 16 GB) for the toy
and small UCI runs, a server with two Intel Xeon Gold 6226R CPUs (64 threads,
753 GB RAM) for the medium UCI runs and NUTS, and four NVIDIA A10 GPUs (24 GB) for
the image models. Rough run times per run: a toy PDMP about 6 minutes, a small
UCI PDMP about 30 minutes, a medium UCI PDMP 2.5 to 4.5 hours on one core, the
MNIST models about 16 hours and ResNet-20 about 2 days on one GPU.
