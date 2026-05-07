"""
config.py - Central configuration for the forecasting system.
All environment variables and tunable parameters are defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings:
    # ── Project ──────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "Time Series Forecasting System"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_PATH: Path = Path(os.getenv("DATA_PATH", str(BASE_DIR / "data" / "sales_data.xlsx")))
    SAVED_MODELS_DIR: Path = BASE_DIR / "saved_models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    PLOTS_DIR: Path = BASE_DIR / "plots"

    # ── Data Columns ──────────────────────────────────────────────────────────
    DATE_COL: str = "Date"
    TARGET_COL: str = "Total"
    STATE_COL: str = "State"
    CATEGORY_COL: str = "Category"

    # ── Forecasting ───────────────────────────────────────────────────────────
    FORECAST_HORIZON: int = int(os.getenv("FORECAST_HORIZON", "8"))   # weeks
    FREQ: str = "W"                                                     # weekly
    VALIDATION_WEEKS: int = int(os.getenv("VALIDATION_WEEKS", "12"))

    # ── Feature Engineering ───────────────────────────────────────────────────
    LAG_WINDOWS: list = [1, 2, 4, 8, 12, 26]   # weeks
    ROLLING_WINDOWS: list = [4, 8, 12]          # weeks
    SEASONALITY_PERIOD: int = 52                # annual for weekly data

    # ── Model Parameters ──────────────────────────────────────────────────────
    # SARIMA
    SARIMA_P_RANGE: list = [0, 1, 2]
    SARIMA_D_RANGE: list = [0, 1]
    SARIMA_Q_RANGE: list = [0, 1, 2]
    SARIMA_SEASONAL_PERIOD: int = 52

    # Prophet
    PROPHET_CHANGEPOINT_PRIOR_SCALE: float = 0.05
    PROPHET_SEASONALITY_PRIOR_SCALE: float = 10.0

    # XGBoost
    XGBOOST_N_ESTIMATORS: int = 500
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_LEARNING_RATE: float = 0.05
    XGBOOST_SUBSAMPLE: float = 0.8
    XGBOOST_COLSAMPLE_BYTREE: float = 0.8
    XGBOOST_EARLY_STOPPING_ROUNDS: int = 50

    # LSTM
    LSTM_LOOKBACK: int = 26          # weeks of history fed into LSTM
    LSTM_EPOCHS: int = 100
    LSTM_BATCH_SIZE: int = 32
    LSTM_UNITS: list = [64, 32]
    LSTM_DROPOUT: float = 0.2
    LSTM_PATIENCE: int = 15          # early-stopping patience

    # ── Model Selection ───────────────────────────────────────────────────────
    METRIC_FOR_SELECTION: str = "rmse"   # "rmse" | "mae" | "mape"
    MODELS_TO_TRAIN: list = ["sarima", "prophet", "xgboost", "lstm"]

    # ── API ───────────────────────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))

    def __init__(self):
        # Ensure directories exist at startup
        for d in [self.SAVED_MODELS_DIR, self.LOGS_DIR, self.PLOTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
