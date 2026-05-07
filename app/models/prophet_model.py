"""
prophet_model.py - Facebook Prophet forecaster with holiday effects.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from app.models.base_model import BaseForecaster
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class ProphetForecaster(BaseForecaster):
    """Wraps Facebook Prophet for weekly sales forecasting."""

    def __init__(self, state: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(state, config)
        self._last_date: Optional[pd.Timestamp] = None
        self._freq: str = settings.FREQ

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> "ProphetForecaster":
        from prophet import Prophet   # lazy import — optional dependency

        prophet_df = train_df[[settings.DATE_COL, settings.TARGET_COL]].rename(
            columns={settings.DATE_COL: "ds", settings.TARGET_COL: "y"}
        )
        prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)
        self._last_date = prophet_df["ds"].max()

        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=self.config.get(
                "changepoint_prior_scale", settings.PROPHET_CHANGEPOINT_PRIOR_SCALE
            ),
            seasonality_prior_scale=self.config.get(
                "seasonality_prior_scale", settings.PROPHET_SEASONALITY_PRIOR_SCALE
            ),
        )
        self.model.add_country_holidays(country_name="US")

        self.model.fit(prophet_df)
        self.is_fitted = True
        self.metadata = {"last_date": str(self._last_date)}
        logger.info("[Prophet][%s] Fitted on %d rows", self.state, len(prophet_df))
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, horizon: int = settings.FORECAST_HORIZON) -> np.ndarray:
        self._assert_fitted()
        future = self.model.make_future_dataframe(periods=horizon, freq=self._freq)
        forecast = self.model.predict(future)
        return np.maximum(forecast["yhat"].tail(horizon).values, 0)

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "prophet_model.pkl", "wb") as f:
            pickle.dump({"model": self.model, "metadata": self.metadata,
                         "last_date": self._last_date, "freq": self._freq}, f)
        logger.info("[Prophet][%s] Saved to %s", self.state, path)

    def load(self, path: Path) -> "ProphetForecaster":
        with open(path / "prophet_model.pkl", "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.metadata = data["metadata"]
        self._last_date = data["last_date"]
        self._freq = data.get("freq", settings.FREQ)
        self.is_fitted = True
        return self
