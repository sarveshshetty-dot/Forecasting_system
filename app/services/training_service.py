"""
training_service.py - Service layer wrapping the forecasting pipeline for the API.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

from app.training.forecasting_pipeline import ForecastingPipeline
from app.services.model_registry import ModelRegistry
from app.core.logger import get_logger

logger = get_logger(__name__)


class TrainingService:
    """Thin service layer — validates inputs and delegates to the pipeline."""

    def __init__(self):
        self._registry = ModelRegistry()

    def train(
        self,
        data_path: Optional[Path] = None,
        models: Optional[List[str]] = None,
        parallel: bool = False,
    ) -> Dict[str, Any]:
        logger.info("TrainingService.train() called")
        pipeline = ForecastingPipeline(
            data_path=data_path,
            models_to_train=models,
            parallel=parallel,
        )
        result = pipeline.run()
        # Clear model cache so next /predict uses freshly trained models
        self._registry.clear_cache()
        return result
