"""
evaluator.py - Evaluates a fitted model on a validation set.
"""

import numpy as np
import pandas as pd
from typing import Dict

from app.models.base_model import BaseForecaster
from app.core.config import settings
from app.utils.metrics import compute_all_metrics
from app.core.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Runs inference on the validation set and computes metrics."""

    @staticmethod
    def evaluate(model: BaseForecaster, val_df: pd.DataFrame) -> Dict[str, float]:
        """Forecast len(val_df) steps and compare against actuals."""
        horizon = len(val_df)
        try:
            preds = model.predict(horizon)
        except Exception as exc:
            logger.error("[Evaluator] %s prediction failed: %s", model.__class__.__name__, exc)
            return {"mae": np.inf, "rmse": np.inf, "mape": np.inf}

        actuals = val_df[settings.TARGET_COL].values
        min_len = min(len(actuals), len(preds))
        metrics = compute_all_metrics(actuals[:min_len], preds[:min_len])
        logger.info(
            "[Evaluator][%s][%s] MAE=%.2f RMSE=%.2f MAPE=%.2f%%",
            model.state, model.__class__.__name__,
            metrics["mae"], metrics["rmse"], metrics["mape"],
        )
        return metrics
