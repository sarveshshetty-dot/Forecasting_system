"""
feature_engineering.py - Lag, rolling, date, and holiday features.
All features use ONLY past data (no leakage).
"""

import pandas as pd
import numpy as np
from typing import List

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# US federal holidays (approximate weekly dates - extend as needed)
US_HOLIDAYS = {
    "new_year": {"month": 1, "week": 1},
    "memorial_day": {"month": 5, "week": 4},
    "independence_day": {"month": 7, "week": 1},
    "labor_day": {"month": 9, "week": 1},
    "thanksgiving": {"month": 11, "week": 4},
    "christmas": {"month": 12, "week": 4},
}


class FeatureEngineer:
    """Generates time-series features for supervised ML models.

    All lag / rolling computations use .shift(1) or higher so there is
    **no data leakage** — each row can only see past observations.
    """

    def __init__(
        self,
        lag_windows: List[int] = settings.LAG_WINDOWS,
        rolling_windows: List[int] = settings.ROLLING_WINDOWS,
        target_col: str = settings.TARGET_COL,
        date_col: str = settings.DATE_COL,
    ):
        self.lag_windows = lag_windows
        self.rolling_windows = rolling_windows
        self.target_col = target_col
        self.date_col = date_col

    # ── Public ────────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all features to *df* and drop NaN rows caused by lags."""
        df = df.copy().sort_values(self.date_col).reset_index(drop=True)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_date_features(df)
        df = self._add_trend_features(df)
        df = self._add_holiday_features(df)
        df = df.dropna().reset_index(drop=True)
        logger.debug("Feature engineering produced %d rows with %d columns", len(df), len(df.columns))
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return the list of feature columns (everything except target and date)."""
        exclude = {self.target_col, self.date_col, settings.STATE_COL}
        return [c for c in df.columns if c not in exclude]

    # ── Private ───────────────────────────────────────────────────────────────

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for lag in self.lag_windows:
            df[f"lag_{lag}"] = df[self.target_col].shift(lag)
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Shift by 1 BEFORE rolling so the window never includes the current row
        shifted = df[self.target_col].shift(1)
        for w in self.rolling_windows:
            df[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
            df[f"rolling_std_{w}"] = shifted.rolling(w).std()
            df[f"rolling_min_{w}"] = shifted.rolling(w).min()
            df[f"rolling_max_{w}"] = shifted.rolling(w).max()
        return df

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = pd.DatetimeIndex(df[self.date_col])
        df["day_of_week"] = dt.dayofweek
        df["week_of_year"] = dt.isocalendar().week.astype(int)
        df["month"] = dt.month
        df["quarter"] = dt.quarter
        df["year"] = dt.year
        df["is_weekend"] = (dt.dayofweek >= 5).astype(int)
        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["trend_index"] = np.arange(len(df))
        df["expanding_mean"] = df[self.target_col].shift(1).expanding().mean()
        return df

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = pd.DatetimeIndex(df[self.date_col])
        holiday_flag = np.zeros(len(df), dtype=int)
        for _name, info in US_HOLIDAYS.items():
            mask = (dt.month == info["month"]) & (
                dt.isocalendar().week.astype(int) >= info["week"] * 4
            )
            holiday_flag = holiday_flag | mask.astype(int).values
        df["holiday_flag"] = holiday_flag
        return df
