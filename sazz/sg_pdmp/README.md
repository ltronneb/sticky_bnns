# Stochastic gradient PDMPs (Fearnhead et al., 2024)

SG-ZZ, SG-BPS and the sticky SG-SZZ of Fearnhead, Grazzi, Nemeth and Roberts
(2024), Algorithms 2 to 4, on the same BNN targets as `sazz/paper_ready`. The
models, data, priors, MAP references, uniform-in-time posterior draws and
metrics all come from `paper_ready`, which is not changed.

```
gradients.py   control variate minibatch gradient (their eq. 2), centred at the MAP
samplers.py    SG-ZZ (sticky optional) and SG-BPS
common.py      sampler construction, result names
scripts/       toy_bnn, uci_bnn, image_bnn, analyse, exactness
```

Every iteration draws a fresh minibatch, holds the event rates fixed and moves
to the first event or to the end of the interval of length h. After an event
the rest of the interval is simulated with a new minibatch (their Appendix C).
Posterior draws are 4000 points uniform in time after 20% burn-in, as for the
paper's samplers.

## Runs

Run from the directory that contains `sazz/`.

```
python -m sazz.sg_pdmp.scripts.toy_bnn
python -m sazz.sg_pdmp.scripts.uci_bnn --variant small
python -m sazz.sg_pdmp.scripts.uci_bnn --variant medium
python -m sazz.sg_pdmp.scripts.image_bnn --model lenet
python -m sazz.sg_pdmp.scripts.analyse uci --results results/sg_pdmp/uci --paper results/uci
```

Useful options are `--step-sizes` (a sweep, default 1e-3 1e-4), `--batch-size`,
`--grad-budget` (full-data gradients, default the paper's G), `--n-iters`,
`--max-minutes` (equal wall-clock comparisons), `--precondition` (SG-BPS with
the Boomerang's Laplace covariance), `--no-cv` and `--x-ref-from` (start from
the MAP stored in the paper's result files, e.g. `results/paper_v2/shallow`).

## Checks

```
python -m sazz.sg_pdmp.scripts.exactness gaussian   # Bayesian linear regression, exact posterior
python -m sazz.sg_pdmp.scripts.exactness sparse     # SG-SZZ against collapsed Gibbs
```

The bias shrinks with h as the paper predicts. With 1e5 iterations the SG-ZZ
posterior sd ratio goes 2.1, 1.47, 1.2, 1.06 as h goes from 1 to 0.03 (in
units of the posterior sd), with the means right throughout. The SG-SZZ mean
inclusion probability error goes 0.070, 0.035, 0.019, 0.015.
