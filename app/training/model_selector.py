"""
model_selector.py - Picks the best model based on validation metrics.
"""

from typing import Dict, Tuple

from app.core.config import settings
from app.core.logger import get_logger
from app.utils.metrics import weighted_score

logger = get_logger(__name__)


class ModelSelector:
    """Ranks models by a configurable metric and returns the winner."""

    def __init__(self, metric: str = settings.METRIC_FOR_SELECTION):
        self.metric = metric

    def select(self, metrics_dict: Dict[str, Dict[str, float]]) -> Tuple[str, Dict]:
        """Return (best_model_name, all_ranked_metrics)."""
        if self.metric == "weighted":
            scores = {m: weighted_score(v) for m, v in metrics_dict.items()}
        else:
            scores = {m: v.get(self.metric, float("inf")) for m, v in metrics_dict.items()}

        # Filter out inf (model failed)
        valid = {m: s for m, s in scores.items() if s < float("inf")}
        if not valid:
            raise RuntimeError("All models failed evaluation.")

        best = min(valid, key=valid.get)
        logger.info("Best model: %s (score=%.4f by %s)", best, valid[best], self.metric)

        ranked = dict(sorted(scores.items(), key=lambda x: x[1]))
        return best, ranked
