"""
sarima_model.py - SARIMA / SARIMAX forecaster with automatic order selection.
"""

import warnings
import itertools
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from app.models.base_model import BaseForecaster
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


class SARIMAForecaster(BaseForecaster):
    """SARIMA model with grid-search for (p,d,q)(P,D,Q,s) orders."""

    def __init__(self, state: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(state, config)
        self.order: Tuple[int, int, int] = (1, 1, 1)
        self.seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 52)
        self._history: Optional[pd.Series] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> "SARIMAForecaster":
        ts = train_df.set_index(settings.DATE_COL)[settings.TARGET_COL].sort_index()
        self._history = ts.copy()

        d = self._determine_d(ts)
        self.order, self.seasonal_order = self._grid_search(ts, d)
        logger.info("[SARIMA][%s] Best order=%s seasonal=%s", self.state, self.order, self.seasonal_order)

        self.model = SARIMAX(
            ts,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        self.is_fitted = True
        self.metadata = {"order": self.order, "seasonal_order": self.seasonal_order}
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, horizon: int = settings.FORECAST_HORIZON) -> np.ndarray:
        self._assert_fitted()
        forecast = self.model.forecast(steps=horizon)
        return np.maximum(forecast.values, 0)   # clip negatives

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "sarima_model.pkl", "wb") as f:
            pickle.dump({"model_result": self.model, "metadata": self.metadata,
                         "order": self.order, "seasonal_order": self.seasonal_order,
                         "history": self._history}, f)
        logger.info("[SARIMA][%s] Saved to %s", self.state, path)

    def load(self, path: Path) -> "SARIMAForecaster":
        with open(path / "sarima_model.pkl", "rb") as f:
            data = pickle.load(f)
        self.model = data["model_result"]
        self.metadata = data["metadata"]
        self.order = data["order"]
        self.seasonal_order = data["seasonal_order"]
        self._history = data["history"]
        self.is_fitted = True
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _determine_d(self, ts: pd.Series) -> int:
        try:
            p_val = adfuller(ts.dropna())[1]
            return 0 if p_val < 0.05 else 1
        except Exception:
            return 1

    def _grid_search(self, ts: pd.Series, d: int) -> Tuple[Tuple, Tuple]:
        """Lightweight grid search — keeps search space small for speed."""
        best_aic = np.inf
        best_order = (1, d, 1)
        best_seasonal = (0, 0, 0, 0)   # default: no seasonal component

        p_range = self.config.get("p_range", [0, 1, 2])
        q_range = self.config.get("q_range", [0, 1, 2])

        for p, q in itertools.product(p_range, q_range):
            try:
                res = SARIMAX(ts, order=(p, d, q), seasonal_order=(0, 0, 0, 0),
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
            except Exception:
                continue

        return best_order, best_seasonal
