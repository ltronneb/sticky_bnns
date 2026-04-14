from . import logreg as _logreg
from . import bnn_classification as _bnn_classification
from . import bnn_regression as _bnn_regression
from .base import Target
from .priors import build_prior, PRIORS

TARGETS = {
    "logreg":             _logreg.logreg,
    "logreg_synthetic":   _logreg.logreg_synthetic,
    "bnn_classification": _bnn_classification.bnn_classification,
    "bnn_regression": _bnn_regression.bnn_regression,
}


def build_target(name: str, **kwargs):
    return TARGETS[name](**kwargs)
