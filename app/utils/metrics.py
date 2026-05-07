"""
metrics.py - Evaluation metrics for time-series forecasting.
"""

import numpy as np
from typing import Dict


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((actual - predicted) / (np.abs(actual) + eps))) * 100)


def compute_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
    }


def weighted_score(metrics: Dict[str, float]) -> float:
    """Lower is better. RMSE=50%, MAE=30%, MAPE=20%."""
    return 0.5 * metrics["rmse"] + 0.3 * metrics["mae"] + 0.2 * metrics["mape"]
