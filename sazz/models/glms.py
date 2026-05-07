
import torch.nn as nn
from torch import Tensor

class LinearRegressionNet(nn.Module):
    """Single linear layer with bias — the nn.Module form of linear regression.
    Output is a scalar prediction per input row."""
    def __init__(self, in_features: int):
        super().__init__()
        self.layer = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x).squeeze(-1)


class LogisticRegressionNet(nn.Module):
    """Single linear layer with bias — the nn.Module form of logistic regression.
    Returns raw logits (no sigmoid); pair with BernoulliLikelihood or BCEWithLogitsLoss."""
    def __init__(self, in_features: int):
        super().__init__()
        self.layer = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x).squeeze(-1)

