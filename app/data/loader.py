"""
loader.py - Handles loading and initial validation of the sales Excel file.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Loads raw sales data from an Excel (or CSV) file and performs
    structural validation before returning a clean DataFrame."""

    REQUIRED_COLS = {settings.STATE_COL, settings.DATE_COL, settings.TARGET_COL}

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = Path(file_path) if file_path else settings.DATA_PATH

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Load data, validate columns, parse dates, and return raw DataFrame."""
        logger.info("Loading data from %s", self.file_path)
        df = self._read_file()
        df = self._validate_columns(df)
        df = self._parse_dates(df)
        df = self._basic_clean(df)
        logger.info("Loaded %d rows | states: %s", len(df), df[settings.STATE_COL].nunique())
        return df

    # ── Private ───────────────────────────────────────────────────────────────

    def _read_file(self) -> pd.DataFrame:
        suffix = self.file_path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(self.file_path)
        if suffix == ".csv":
            return pd.read_csv(self.file_path)
        raise ValueError(f"Unsupported file format: {suffix}")

    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df[settings.DATE_COL] = pd.to_datetime(df[settings.DATE_COL], infer_datetime_format=True)
        return df

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=[settings.STATE_COL, settings.DATE_COL])
        after = len(df)
        if before != after:
            logger.warning("Removed %d duplicate (state, date) rows", before - after)
        df = df.sort_values([settings.STATE_COL, settings.DATE_COL]).reset_index(drop=True)
        return df
