"""
forecasting_pipeline.py - End-to-end pipeline: load → preprocess → train all states.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from app.data.loader import DataLoader
from app.data.preprocessing import Preprocessor
from app.training.trainer import StateTrainer
from app.core.config import settings
from app.core.logger import get_logger
from app.utils.helpers import save_json

logger = get_logger(__name__)


class ForecastingPipeline:
    """Top-level orchestrator used by the training API endpoint."""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        models_to_train: Optional[List[str]] = None,
        parallel: bool = False,
    ):
        self.data_path = data_path or settings.DATA_PATH
        self.models_to_train = models_to_train or settings.MODELS_TO_TRAIN
        self.parallel = parallel

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        logger.info("=== ForecastingPipeline START ===")

        # 1. Load
        raw_df = DataLoader(self.data_path).load()

        # 2. Preprocess
        preprocessor = Preprocessor()
        clean_df = preprocessor.preprocess(raw_df)

        # 3. Train per state
        states = clean_df[settings.STATE_COL].unique().tolist()
        logger.info("States to train: %d", len(states))

        results: Dict[str, Any] = {}

        if self.parallel:
            results = self._parallel_train(clean_df, states)
        else:
            for state in states:
                state_df = clean_df[clean_df[settings.STATE_COL] == state].copy()
                results[state] = self._train_state(state, state_df)

        # 4. Save summary
        summary = {"states_trained": list(results.keys()), "results": results}
        save_json(summary, settings.SAVED_MODELS_DIR / "training_summary.json")
        logger.info("=== ForecastingPipeline COMPLETE ===")
        return summary

    # ── Private ───────────────────────────────────────────────────────────────

    def _train_state(self, state: str, state_df: pd.DataFrame) -> Dict[str, Any]:
        try:
            trainer = StateTrainer(state=state, models_to_train=self.models_to_train)
            return trainer.run(state_df)
        except Exception as exc:
            logger.error("Pipeline failed for state %s: %s", state, exc, exc_info=True)
            return {"state": state, "error": str(exc)}

    def _parallel_train(self, clean_df: pd.DataFrame, states: List[str]) -> Dict[str, Any]:
        results = {}
        # Use max 4 workers to avoid OOM with LSTM/XGBoost
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    self._train_state,
                    state,
                    clean_df[clean_df[settings.STATE_COL] == state].copy(),
                ): state
                for state in states
            }
            for future in as_completed(futures):
                state = futures[future]
                results[state] = future.result()
        return results
