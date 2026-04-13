"""
Factories construct samplers. They do NOT preprocess. They do NOT compute kappa.
Preprocessing is handled by the runner via target/config.
Kappa is supplied by the config (possibly via a helper).
"""
from sazz.samplers.boomerang_sampler.BoomerangSampler import BoomerangSampler
from sazz.samplers.boomerang_sampler.BoomerangSampler_PLI import BoomerangSampler_PLI
from sazz.samplers.boomerang_sampler.FactorizedBoomerangSampler import FactorizedBoomerangSampler
from sazz.samplers.boomerang_sampler.StickyBoomerangSampler import StickyBoomerangSampler
from sazz.samplers.boomerang_sampler.StickyBoomerangSampler_PLI import StickyBoomerangSampler_PLI


def boomerang(target, N, **kwargs):
    return BoomerangSampler(
        E=target.E, N=N, D=target.D, grad_target=target.gradE, **kwargs
        )

def boomerang_pli(target, N, **kwargs):
    return BoomerangSampler_PLI(
        E=target.E, N=N, D=target.D, grad_target=target.gradE, **kwargs
        )


def factorized_boomerang(target, N, **kwargs):
    return FactorizedBoomerangSampler(
        E=target.E, N=N, D=target.D, grad_target=target.gradE, **kwargs
        )


def sticky_boomerang(target, N, kappa, **kwargs):
    return StickyBoomerangSampler(
        E=target.E, N=N, D=target.D, grad_target=target.gradE, kappa=kappa, **kwargs
    )

def sticky_boomerang_PLI(target, N, kappa, **kwargs):
    return StickyBoomerangSampler_PLI(
        E=target.E, N=N, D=target.D, grad_target=target.gradE, kappa=kappa, **kwargs
    )


SAMPLERS = {
    "boomerang": boomerang,
    "boomerang_pli": boomerang_pli,
    "factorized_boomerang": factorized_boomerang,
    "sticky_boomerang": sticky_boomerang,
    "sticky_boomerang_pli": sticky_boomerang_PLI,
}


def build_sampler(name: str, target, N, **kwargs):
    return SAMPLERS[name](target, N, **kwargs)
