"""
trainer.py - Orchestrates fit → evaluate → select → save for one state.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

from app.models.base_model import BaseForecaster
from app.models.sarima_model import SARIMAForecaster
from app.models.prophet_model import ProphetForecaster
from app.models.xgboost_model import XGBoostForecaster
from app.models.lstm_model import LSTMForecaster
from app.training.evaluator import Evaluator
from app.training.model_selector import ModelSelector
from app.data.preprocessing import Preprocessor
from app.utils.helpers import state_model_dir
from app.utils.visualization import plot_forecast, plot_model_comparison, plot_residuals
from app.core.config import settings
from app.core.constants import MODEL_SARIMA, MODEL_PROPHET, MODEL_XGBOOST, MODEL_LSTM
from app.core.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY = {
    MODEL_SARIMA: SARIMAForecaster,
    MODEL_PROPHET: ProphetForecaster,
    MODEL_XGBOOST: XGBoostForecaster,
    MODEL_LSTM: LSTMForecaster,
}


class StateTrainer:
    """Trains and evaluates all models for a single US state."""

    def __init__(
        self,
        state: str,
        models_to_train: Optional[List[str]] = None,
        saved_models_dir: Optional[Path] = None,
    ):
        self.state = state
        self.models_to_train = models_to_train or settings.MODELS_TO_TRAIN
        self.saved_models_dir = saved_models_dir or settings.SAVED_MODELS_DIR
        self._preprocessor = Preprocessor()
        self._selector = ModelSelector()

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, state_df: pd.DataFrame) -> Dict[str, Any]:
        """Full pipeline: split → train all → evaluate → select best → persist."""
        train_df, val_df = self._preprocessor.train_val_split(state_df)
        logger.info("[%s] Train=%d | Val=%d", self.state, len(train_df), len(val_df))

        fitted_models: Dict[str, BaseForecaster] = {}
        all_metrics: Dict[str, Dict[str, float]] = {}

        for model_name in self.models_to_train:
            logger.info("[%s] Training %s …", self.state, model_name)
            try:
                model = self._fit_model(model_name, train_df)
                metrics = Evaluator.evaluate(model, val_df)
                fitted_models[model_name] = model
                all_metrics[model_name] = metrics
            except Exception as exc:
                logger.error("[%s][%s] Training failed: %s", self.state, model_name, exc, exc_info=True)
                all_metrics[model_name] = {"mae": float("inf"), "rmse": float("inf"), "mape": float("inf")}

        best_model_name, rankings = self._selector.select(all_metrics)

        # Save all fitted models; re-train best on FULL data
        for model_name, model in fitted_models.items():
            save_path = state_model_dir(self.saved_models_dir, self.state, model_name)
            model.save(save_path)

        # Re-train best model on full data for final forecasting
        best_model = self._fit_model(best_model_name, state_df)
        best_save_path = state_model_dir(self.saved_models_dir, self.state, f"{best_model_name}_best")
        best_model.save(best_save_path)

        # Generate plots
        self._generate_plots(state_df, fitted_models, all_metrics, val_df)

        return {
            "state": self.state,
            "best_model": best_model_name,
            "metrics": all_metrics,
            "rankings": rankings,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _fit_model(self, model_name: str, train_df: pd.DataFrame) -> BaseForecaster:
        cls = MODEL_REGISTRY[model_name]
        model = cls(state=self.state)
        model.fit(train_df)
        return model

    def _generate_plots(self, full_df, fitted_models, all_metrics, val_df):
        try:
            history = full_df.sort_values(settings.DATE_COL).set_index(settings.DATE_COL)[settings.TARGET_COL]
            val_actual = val_df[settings.TARGET_COL].values

            for model_name, model in fitted_models.items():
                try:
                    forecast = model.predict(settings.FORECAST_HORIZON)
                    plot_forecast(self.state, history, forecast, model_name, val_actual=val_actual)

                    val_preds = model.predict(len(val_df))
                    min_len = min(len(val_actual), len(val_preds))
                    plot_residuals(self.state, model_name, val_actual[:min_len], val_preds[:min_len])
                except Exception as exc:
                    logger.warning("[%s] Plot failed for %s: %s", self.state, model_name, exc)

            plot_model_comparison(self.state, all_metrics)
        except Exception as exc:
            logger.warning("[%s] Visualization failed: %s", self.state, exc)
