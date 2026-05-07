"""
tests/test_pipeline.py - Core unit & integration tests.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data.preprocessing import Preprocessor
from app.data.feature_engineering import FeatureEngineer
from app.utils.metrics import mae, rmse, mape, compute_all_metrics
from app.training.model_selector import ModelSelector
from app.core.config import settings


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Generate a synthetic weekly state dataset."""
    dates = pd.date_range("2020-01-05", periods=100, freq="W")
    data = {
        settings.STATE_COL: ["TestState"] * 100,
        settings.DATE_COL: dates,
        settings.TARGET_COL: (
            10_000 + np.cumsum(np.random.randn(100) * 500)
        ).clip(min=1000),
    }
    return pd.DataFrame(data)


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_mae_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mae(a, a) == pytest.approx(0.0)

    def test_rmse_known(self):
        actual = np.array([3.0, 5.0])
        pred = np.array([4.0, 4.0])
        expected = np.sqrt(((1.0**2 + 1.0**2) / 2))
        assert rmse(actual, pred) == pytest.approx(expected)

    def test_mape_symmetry(self):
        a = np.array([100.0, 200.0])
        p = np.array([110.0, 190.0])
        result = mape(a, p)
        assert 0 < result < 20

    def test_compute_all_returns_all_keys(self):
        a = np.array([1.0, 2.0])
        metrics = compute_all_metrics(a, a)
        assert set(metrics.keys()) == {"mae", "rmse", "mape"}


# ── Preprocessor ──────────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_preprocess_fills_gaps(self, sample_df):
        # Drop a few rows to create gaps
        dropped = sample_df.drop(index=[5, 10, 20]).reset_index(drop=True)
        pp = Preprocessor()
        result = pp.preprocess(dropped)
        assert result[settings.TARGET_COL].isna().sum() == 0

    def test_train_val_split_is_chronological(self, sample_df):
        pp = Preprocessor(freq="W")
        clean = pp.preprocess(sample_df)
        state_df = clean[clean[settings.STATE_COL] == "TestState"]
        train, val = pp.train_val_split(state_df, val_weeks=10)
        assert train[settings.DATE_COL].max() < val[settings.DATE_COL].min()

    def test_no_leakage_in_split(self, sample_df):
        pp = Preprocessor(freq="W")
        clean = pp.preprocess(sample_df)
        state_df = clean[clean[settings.STATE_COL] == "TestState"]
        train, val = pp.train_val_split(state_df, val_weeks=10)
        assert set(train[settings.DATE_COL]) & set(val[settings.DATE_COL]) == set()


# ── Feature Engineering ───────────────────────────────────────────────────────

class TestFeatureEngineer:
    def test_lag_features_created(self, sample_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_df)
        for lag in settings.LAG_WINDOWS:
            assert f"lag_{lag}" in result.columns

    def test_no_nan_after_transform(self, sample_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert result.isnull().sum().sum() == 0

    def test_rolling_features_created(self, sample_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_df)
        for w in settings.ROLLING_WINDOWS:
            assert f"rolling_mean_{w}" in result.columns

    def test_date_features_created(self, sample_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_df)
        for col in ["month", "quarter", "year", "week_of_year"]:
            assert col in result.columns


# ── Model Selector ────────────────────────────────────────────────────────────

class TestModelSelector:
    def test_picks_lowest_rmse(self):
        metrics = {
            "xgboost": {"mae": 100, "rmse": 200, "mape": 5},
            "prophet": {"mae": 150, "rmse": 300, "mape": 7},
            "sarima":  {"mae": 120, "rmse": 250, "mape": 6},
        }
        selector = ModelSelector(metric="rmse")
        best, _ = selector.select(metrics)
        assert best == "xgboost"

    def test_excludes_inf_models(self):
        metrics = {
            "lstm":    {"mae": np.inf, "rmse": np.inf, "mape": np.inf},
            "xgboost": {"mae": 100, "rmse": 200, "mape": 5},
        }
        selector = ModelSelector(metric="rmse")
        best, _ = selector.select(metrics)
        assert best == "xgboost"

    def test_raises_when_all_fail(self):
        metrics = {
            "lstm": {"mae": np.inf, "rmse": np.inf, "mape": np.inf},
        }
        selector = ModelSelector(metric="rmse")
        with pytest.raises(RuntimeError):
            selector.select(metrics)
