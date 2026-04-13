from dataclasses import dataclass, field
from typing import Callable
import numpy as np


@dataclass
class Target:
    """
    Minimal contract: name, task_type, D, E, gradE.
    Everything else is optional and carried for downstream use.

    Optional fields:
      - true_params : parameter recovery metrics
      - x_ref, Sigma_inv : for manual preprocessing (BNN warmstart, etc.)
      - data : free-form (X, y, tau, ...)
      - meta : free-form metadata
    """
    name: str
    task_type: str
    D: int
    E: Callable
    gradE: Callable
    true_params: np.ndarray | None = None
    x_ref: np.ndarray | None = None
    Sigma_inv: np.ndarray | None = None
    data: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_callables(cls, name, task_type, D, E, gradE, **kwargs):
        """Escape hatch: wrap raw E/gradE without writing a dedicated factory."""
        return cls(name=name, task_type=task_type, D=D, E=E, gradE=gradE, **kwargs)
