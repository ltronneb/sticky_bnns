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
data.py, common.py        datasets, sampler construction, result files
scripts/    one script per experiment
```

Events are simulated by thinning a Poisson process with one tangent-line bound
per window of length t_max, which adapts during the run. Every PDMP run has a
budget of K events or G gradient evaluations. Posterior samples are S = 4000
points drawn uniformly in time over the trajectory after discarding the first
20% of the time.

## Setup

```
pip install torch torchvision numpy scipy scikit-learn pandas openpyxl xlrd matplotlib tqdm jax numpyro
```

Run every command from the directory that contains `sazz/`. The UCI files
(energy, yacht, concrete) go in `datasets/`, Boston is fetched from OpenML, and
MNIST and CIFAR-10 are downloaded to `datasets/`. A GPU is used when present
(float32), otherwise the CPU (float64). Add `--resume` to skip finished runs.

## Experiments

Toy regression, four data sets, G = 2e5.

```
python -m sazz.paper_ready.scripts.toy_bnn
```

UCI regression, four data sets and five splits. Small and medium networks with
G = 1e6, the large network with the sticky samplers only, K = 1e6 and
minibatches of 128.

```
python -m sazz.paper_ready.scripts.uci_bnn --variant small
python -m sazz.paper_ready.scripts.uci_bnn --variant medium
python -m sazz.paper_ready.scripts.uci_bnn --variant small --piw 0.1 --samplers sticky_zigzag sticky_boomerang
python -m sazz.paper_ready.scripts.uci_bnn --variant medium --piw 0.1 --samplers sticky_zigzag sticky_boomerang
python -m sazz.paper_ready.scripts.uci_bnn --variant large --samplers sticky_zigzag sticky_boomerang lbbnn
```

Prior inclusion sweep and multi-chain convergence check.

```
python -m sazz.paper_ready.scripts.uci_bnn --variant small --datasets boston --splits 0 \
    --piw 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --samplers sticky_zigzag sticky_boomerang
python -m sazz.paper_ready.scripts.uci_bnn --variant large --datasets boston --splits 0 \
    --chains 0 1 2 3 --samplers sticky_zigzag sticky_boomerang
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

Images, an FFN and LeNet-5 on MNIST and ResNet-20 on CIFAR-10, K = 1e6. For
each model, first the SGD baseline (for ResNet-20 also the BatchNorm
statistics), then the pruned MAP reference, then the sticky samplers.

```
for m in ffn lenet resnet20; do
  python -m sazz.paper_ready.scripts.image_reference sgd --model $m
  python -m sazz.paper_ready.scripts.image_reference map --model $m
  python -m sazz.paper_ready.scripts.image_bnn --model $m
done
```

Each run saves a `.pt` file with the posterior samples (`samples`), the
gradient evaluations, the number of events, the number of bound violations and
the frozen coordinates at the end of the run.
