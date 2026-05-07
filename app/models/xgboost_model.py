"""
xgboost_model.py - XGBoost forecaster using supervised lag-feature approach
with recursive multi-step forecasting.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

from app.models.base_model import BaseForecaster
from app.data.feature_engineering import FeatureEngineer
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class XGBoostForecaster(BaseForecaster):
    """XGBoost trained on lag features; uses recursive forecasting for future
    steps (each prediction is fed back as a lag for the next step)."""

    def __init__(self, state: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(state, config)
        self._feature_engineer = FeatureEngineer()
        self._feature_cols: List[str] = []
        self._last_known: Optional[pd.Series] = None   # last window of actuals
        self._last_date: Optional[pd.Timestamp] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> "XGBoostForecaster":
        df_feat = self._feature_engineer.fit_transform(train_df)
        self._feature_cols = self._feature_engineer.get_feature_columns(df_feat)

        X = df_feat[self._feature_cols]
        y = df_feat[settings.TARGET_COL]

        self.model = XGBRegressor(
            n_estimators=self.config.get("n_estimators", settings.XGBOOST_N_ESTIMATORS),
            max_depth=self.config.get("max_depth", settings.XGBOOST_MAX_DEPTH),
            learning_rate=self.config.get("learning_rate", settings.XGBOOST_LEARNING_RATE),
            subsample=self.config.get("subsample", settings.XGBOOST_SUBSAMPLE),
            colsample_bytree=self.config.get("colsample_bytree", settings.XGBOOST_COLSAMPLE_BYTREE),
            random_state=42,
            tree_method="hist",
        )
        self.model.fit(X, y)

        # Keep the raw history for recursive forecasting
        raw = train_df.sort_values(settings.DATE_COL).reset_index(drop=True)
        self._last_known = raw[settings.TARGET_COL].copy()
        self._last_date = raw[settings.DATE_COL].max()

        self.is_fitted = True
        self.feature_columns = self._feature_cols
        self.metadata = {"n_features": len(self._feature_cols), "last_date": str(self._last_date)}
        logger.info("[XGBoost][%s] Fitted | features=%d", self.state, len(self._feature_cols))
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, horizon: int = settings.FORECAST_HORIZON) -> np.ndarray:
        self._assert_fitted()
        history = list(self._last_known)
        preds = []
        current_date = self._last_date

        for _ in range(horizon):
            current_date = current_date + pd.tseries.frequencies.to_offset(settings.FREQ)
            feat = self._build_future_features(history, current_date)
            x = pd.DataFrame([feat])[self._feature_cols]
            pred = float(self.model.predict(x)[0])
            pred = max(pred, 0)
            preds.append(pred)
            history.append(pred)

        return np.array(preds)

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_cols": self._feature_cols,
            "last_known": self._last_known,
            "last_date": self._last_date,
            "metadata": self.metadata,
        }, path / "xgboost_model.joblib")
        logger.info("[XGBoost][%s] Saved to %s", self.state, path)

    def load(self, path: Path) -> "XGBoostForecaster":
        data = joblib.load(path / "xgboost_model.joblib")
        self.model = data["model"]
        self._feature_cols = data["feature_cols"]
        self._last_known = data["last_known"]
        self._last_date = data["last_date"]
        self.metadata = data["metadata"]
        self.feature_columns = self._feature_cols
        self.is_fitted = True
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_future_features(self, history: list, date: pd.Timestamp) -> dict:
        """Build one row of features from rolling history list."""
        feat: dict = {}
        h = np.array(history)

        # Lag features
        for lag in settings.LAG_WINDOWS:
            feat[f"lag_{lag}"] = h[-lag] if len(h) >= lag else np.nan

        # Rolling features (over past values, no current value)
        for w in settings.ROLLING_WINDOWS:
            window = h[-w:] if len(h) >= w else h
            feat[f"rolling_mean_{w}"] = np.mean(window)
            feat[f"rolling_std_{w}"] = np.std(window) if len(window) > 1 else 0
            feat[f"rolling_min_{w}"] = np.min(window)
            feat[f"rolling_max_{w}"] = np.max(window)

        # Date features
        feat["day_of_week"] = date.dayofweek
        feat["week_of_year"] = date.isocalendar()[1]
        feat["month"] = date.month
        feat["quarter"] = date.quarter
        feat["year"] = date.year
        feat["is_weekend"] = int(date.dayofweek >= 5)

        # Trend
        feat["trend_index"] = len(history)
        feat["expanding_mean"] = np.mean(h)

        # Holiday (simple flag)
        feat["holiday_flag"] = int(date.month in [1, 7, 11, 12])

        return feat
