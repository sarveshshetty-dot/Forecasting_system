"""
prediction_service.py - Generates forecasts for a given state via the registry.
"""

from typing import Dict, Any, List, Optional

import pandas as pd

from app.services.model_registry import ModelRegistry
from app.core.config import settings
from app.core.logger import get_logger
from app.utils.helpers import load_json

logger = get_logger(__name__)


class PredictionService:
    """Loads the best model for a state and returns structured forecasts."""

    def __init__(self):
        self._registry = ModelRegistry()

    def predict(self, state: str, horizon: int = settings.FORECAST_HORIZON) -> Dict[str, Any]:
        logger.info("Predicting %d weeks for state '%s'", horizon, state)
        model = self._registry.get_best_model(state)
        raw_preds = model.predict(horizon)

        forecast = [
            {"week": i + 1, "prediction": round(float(v), 2)}
            for i, v in enumerate(raw_preds)
        ]

        best_model_name = model.__class__.__name__.replace("Forecaster", "").lower()

        return {
            "state": state,
            "best_model": best_model_name,
            "forecast_horizon_weeks": horizon,
            "forecast": forecast,
        }

    def compare_models(self, state: Optional[str] = None) -> Dict[str, Any]:
        """Return training summary metrics (optionally filtered by state)."""
        summary_path = settings.SAVED_MODELS_DIR / "training_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError("No training summary found. Run /train first.")
        summary = load_json(summary_path)
        results = summary.get("results", {})
        if state:
            filtered = {k: v for k, v in results.items() if k == state}
            return {"state": state, "results": filtered}
        return {"results": results}
