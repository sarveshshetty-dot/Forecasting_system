"""
model_registry.py - Loads saved model artefacts from disk on demand.
"""

from pathlib import Path
from typing import Dict, Optional

from app.models.base_model import BaseForecaster
from app.models.sarima_model import SARIMAForecaster
from app.models.prophet_model import ProphetForecaster
from app.models.xgboost_model import XGBoostForecaster
from app.models.lstm_model import LSTMForecaster
from app.core.config import settings
from app.core.logger import get_logger
from app.utils.helpers import sanitize_state_name, load_json

logger = get_logger(__name__)

_MODEL_CLASSES = {
    "sarima": SARIMAForecaster,
    "prophet": ProphetForecaster,
    "xgboost": XGBoostForecaster,
    "lstm": LSTMForecaster,
}

# In-process cache: {state -> model}
_cache: Dict[str, BaseForecaster] = {}


class ModelRegistry:
    """Resolves which model to use for a given state and loads it."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or settings.SAVED_MODELS_DIR

    def get_best_model(self, state: str) -> BaseForecaster:
        """Return the best-model instance for *state* (cached after first load)."""
        if state in _cache:
            return _cache[state]

        best_name = self._resolve_best_model_name(state)
        model = self._load(state, best_name)
        _cache[state] = model
        return model

    def get_model(self, state: str, model_name: str) -> BaseForecaster:
        """Load a specific model for a state."""
        return self._load(state, model_name)

    def available_states(self) -> list:
        if not self.base_dir.exists():
            return []
        return [d.name for d in self.base_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]

    def clear_cache(self):
        _cache.clear()

    # ── Private ───────────────────────────────────────────────────────────────

    def _resolve_best_model_name(self, state: str) -> str:
        summary_path = self.base_dir / "training_summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            state_result = summary.get("results", {}).get(state, {})
            best = state_result.get("best_model")
            if best:
                return f"{best}_best"

        # Fallback: look for any *_best directory
        state_dir = self.base_dir / sanitize_state_name(state)
        if state_dir.exists():
            best_dirs = sorted(state_dir.glob("*_best"))
            if best_dirs:
                return best_dirs[0].name

        raise FileNotFoundError(f"No saved model found for state '{state}'. Train first.")

    def _load(self, state: str, model_dir_name: str) -> BaseForecaster:
        from app.utils.helpers import state_model_dir
        model_name_base = model_dir_name.replace("_best", "")
        cls = _MODEL_CLASSES.get(model_name_base)
        if cls is None:
            raise ValueError(f"Unknown model name: {model_name_base}")

        path = state_model_dir(self.base_dir, state, model_dir_name)
        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        model = cls(state=state)
        model.load(path)
        logger.info("Loaded %s for state '%s' from %s", cls.__name__, state, path)
        return model
