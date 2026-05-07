"""
preprocessing.py - Missing-value handling, date-gap filling, and resampling.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """Per-state preprocessing: fills date gaps, interpolates missing sales."""

    def __init__(self, freq: str = settings.FREQ):
        self.freq = freq

    # ── Public ────────────────────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a fully preprocessed DataFrame with one row per (state, week)."""
        processed_parts = []
        for state, group in df.groupby(settings.STATE_COL):
            processed = self._process_state(state, group)
            processed_parts.append(processed)
        result = pd.concat(processed_parts, ignore_index=True)
        logger.info("Preprocessing complete: %d rows across %d states",
                    len(result), result[settings.STATE_COL].nunique())
        return result

    def train_val_split(self, state_df: pd.DataFrame, val_weeks: int = settings.VALIDATION_WEEKS
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological split — NO data leakage."""
        state_df = state_df.sort_values(settings.DATE_COL).reset_index(drop=True)
        split_idx = len(state_df) - val_weeks
        if split_idx <= 0:
            raise ValueError(
                f"Not enough data for {val_weeks} validation weeks "
                f"(only {len(state_df)} rows available)."
            )
        return state_df.iloc[:split_idx].copy(), state_df.iloc[split_idx:].copy()

    # ── Private ───────────────────────────────────────────────────────────────

    def _process_state(self, state: str, group: pd.DataFrame) -> pd.DataFrame:
        group = group.set_index(settings.DATE_COL)[[settings.TARGET_COL]].copy()

        # Resample to weekly frequency, sum multiple rows if any
        group = group.resample(self.freq).sum()

        # Replace 0-sales (from resampling gaps) with NaN so we can interpolate
        group[settings.TARGET_COL] = group[settings.TARGET_COL].replace(0, np.nan)

        # Fill missing values
        group[settings.TARGET_COL] = (
            group[settings.TARGET_COL]
            .interpolate(method="time")
            .ffill()
            .bfill()
        )

        n_missing = group[settings.TARGET_COL].isna().sum()
        if n_missing:
            logger.warning("State %s: %d NaNs remain after interpolation", state, n_missing)

        group = group.reset_index()
        group[settings.STATE_COL] = state
        return group
