"""기존 890 전처리와 256→128 구조를 보존한 bounded residual MLP."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from torch import nn

from contract import ARCHITECTURE, EPS, NUMERIC, ONEHOT, PREDICTION_FEATURES


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("qt", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("qt", QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                                     subsample=200_000, random_state=0))]), NUMERIC),
        ("na", MissingIndicator(features="missing-only"), NUMERIC),
        ("season", StandardScaler(), ["season"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONEHOT),
        ("prediction", StandardScaler(), PREDICTION_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, ARCHITECTURE[0]), nn.ReLU(),
            nn.Linear(ARCHITECTURE[0], ARCHITECTURE[1]), nn.ReLU(),
        )
        self.output = nn.Linear(ARCHITECTURE[1], 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(values)).squeeze(-1)


def final_probability(base: torch.Tensor, raw: torch.Tensor, cap: float) -> torch.Tensor:
    anchor = base.detach().clamp(EPS, 1 - EPS)
    return torch.sigmoid(torch.logit(anchor) + float(cap) * torch.tanh(raw))


def numpy_probability(base: np.ndarray, raw: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray]:
    correction = float(cap) * np.tanh(np.asarray(raw, dtype=np.float64))
    anchor = np.clip(np.asarray(base, dtype=np.float64), EPS, 1 - EPS)
    probability = 1.0 / (1.0 + np.exp(-(np.log(anchor / (1-anchor)) + correction)))
    return probability, correction
