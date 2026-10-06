"""Checks of the SG-PDMP samplers against exact posteriors, as the step size shrinks.

gaussian  Bayesian linear regression with known noise and a N(0, tau^2) prior, so
          the posterior is Gaussian. SG-ZZ and SG-BPS with one data point per
          iteration and control variates, error of the posterior means (in
          posterior standard deviations) and ratio of the standard deviations.
sparse    the sparse linear regression of paper_ready's linreg_exactness, SG-SZZ
          against collapsed Gibbs, mean absolute error of the inclusion
          probabilities and of the posterior means.

Both errors should shrink with the step size h (the SG-PDMP bias is O(h)).

    python -m sazz.sg_pdmp.scripts.exactness gaussian
    python -m sazz.sg_pdmp.scripts.exactness sparse
"""

import argparse
import math

import torch
import torch.nn as nn

from sazz.paper_ready.models.bnn import BNN
from sazz.paper_ready.samplers import run_and_resample
from sazz.paper_ready.scripts.linreg_exactness import collapsed_gibbs, make_data

from ..gradients import ControlVariateGradient
from ..samplers import SGBPS, SGZigZag

DT = torch.float64


def linear_target(X, y, sigma, tau):
    module = nn.Linear(X.shape[1], 1, bias=False)
    return BNN.build(module, "gaussian", X, y, torch.full((X.shape[1],), tau, dtype=DT),
                     noise_std=sigma, dtype=DT)


def exact_gaussian(X, y, sigma, tau):
    prec = X.T @ X / sigma ** 2 + torch.eye(X.shape[1], dtype=DT) / tau ** 2
    cov = torch.linalg.inv(prec)
    return cov @ X.T @ y / sigma ** 2, cov.diagonal().sqrt()


def gaussian(args):
    g = torch.Generator().manual_seed(0)
    N, d, sigma, tau = args.N, args.D, 1.0, 10.0
    X = torch.randn(N, d, generator=g, dtype=DT)
    y = X @ torch.randn(d, generator=g, dtype=DT) + sigma * torch.randn(N, generator=g, dtype=DT)
    mean, sd = exact_gaussian(X, y, sigma, tau)
    bm = linear_target(X, y, sigma, tau)
    print(f"N={N}, d={d}, posterior sd {sd.mean():.4f}, batch 1, {args.n_iters:,} iterations")
    print(f"{'sampler':<8}{'h':>9}{'T':>9}{'events':>10}{'max |mean err|/sd':>19}{'sd ratio range':>18}")
    for h in args.step_sizes:
        for name, cls in (("SG-ZZ", SGZigZag), ("SG-BPS", SGBPS)):
            torch.manual_seed(1)
            grad = ControlVariateGradient(bm, 1, mean)
            # h in units of the time to cross one posterior sd, unit speed ZZ and BPS with v ~ N(0, Sigma)
            kw = dict(metric=sd ** 2) if cls is SGBPS else {}
            s = cls(grad, d, h if cls is SGBPS else h * float(sd.mean()), **kw)
            out = run_and_resample(s, mean, n_out=args.n_draws, n_iters=args.n_iters, progress=False)
            z = out["samples"]
            err = ((z.mean(0) - mean) / sd).abs().max()
            ratio = z.std(0) / sd
            print(f"{name:<8}{h:>9.0e}{out['final_time']:>9.3g}{out['n_events']:>10,}{err:>19.3f}"
                  f"{f'{ratio.min():.3f}-{ratio.max():.3f}':>18}")


def sparse(args):
    X, y, beta = make_data(args.N, args.D, 5, seed=0)
    sigma, tau, w = 1.0, 1.0, 0.2
    gibbs = collapsed_gibbs(X, y, sigma, tau, w, 20_000, 2_000, seed=0)
    p_ref, m_ref = (gibbs != 0).double().mean(0), gibbs.mean(0)
    bm = linear_target(X, y, sigma, tau)
    kappa = w / (1 - w) / (tau * math.sqrt(2 * math.pi))
    x_hat = torch.linalg.solve(X.T @ X / sigma ** 2 + torch.eye(args.D, dtype=DT) / tau ** 2,
                               X.T @ y / sigma ** 2)
    sd = (gibbs.std(0)[p_ref > 0.5]).mean()
    print(f"N={args.N}, D={args.D}, w={w}, batch 1, {args.n_iters:,} iterations, h in units of "
          f"posterior sd {sd:.3f}")
    print(f"{'h':>9}{'T':>9}{'freezes':>10}{'mean |incl err|':>17}{'max |incl err|':>16}"
          f"{'mean |mean err|':>17}")
    for h in args.step_sizes:
        torch.manual_seed(1)
        s = SGZigZag(ControlVariateGradient(bm, 1, x_hat), args.D, h * float(sd), kappa=kappa)
        out = run_and_resample(s, x_hat, n_out=args.n_draws, n_iters=args.n_iters, progress=False)
        z = out["samples"]
        de = ((z != 0).double().mean(0) - p_ref).abs()
        print(f"{h:>9.0e}{out['final_time']:>9.3g}{out['freeze']:>10,}{de.mean():>17.3f}{de.max():>16.3f}"
              f"{(z.mean(0) - m_ref).abs().mean():>17.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("check", choices=["gaussian", "sparse"])
    p.add_argument("--step-sizes", nargs="+", type=float, default=[1.0, 0.3, 0.1, 0.03])
    p.add_argument("--n-iters", type=int, default=200_000)
    p.add_argument("--n-draws", type=int, default=20_000)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--D", type=int, default=None)
    args = p.parse_args()
    if args.check == "gaussian":
        args.N, args.D = args.N or 10_000, args.D or 5
        gaussian(args)
    else:
        args.N, args.D = args.N or 200, args.D or 20
        sparse(args)


if __name__ == "__main__":
    main()
