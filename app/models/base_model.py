"""
base_model.py - Abstract base class all forecasting models must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np


class BaseForecaster(ABC):
    """Contract every model must satisfy."""

    def __init__(self, state: str, config: Optional[Dict[str, Any]] = None):
        self.state = state
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.feature_columns: list = []
        self.metadata: Dict[str, Any] = {}

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseForecaster":
        """Train on *train_df* (chronologically ordered, no leakage)."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Forecast *horizon* steps ahead. Returns 1-D numpy array."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artefacts to *path*."""

    @abstractmethod
    def load(self, path: Path) -> "BaseForecaster":
        """Restore model from *path*."""

    # ── Concrete helpers ──────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} for state '{self.state}' is not fitted yet.")
