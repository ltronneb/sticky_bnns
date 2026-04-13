from . import logreg as _logreg
from . import bnn as _bnn
from .base import Target
from .priors import build_prior, PRIORS

TARGETS = {
    "logreg":             _logreg.logreg,
    "logreg_synthetic":   _logreg.logreg_synthetic,
    "bnn_classification": _bnn.bnn_classification,
}


def build_target(name: str, **kwargs):
    return TARGETS[name](**kwargs)
