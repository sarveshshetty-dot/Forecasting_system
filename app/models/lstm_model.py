"""
lstm_model.py - LSTM deep learning forecaster using Keras/TensorFlow.
Uses a sliding-window approach and recursive multi-step prediction.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from app.models.base_model import BaseForecaster
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class LSTMForecaster(BaseForecaster):
    """LSTM network for time-series forecasting.

    Architecture: LSTM → Dropout → LSTM → Dropout → Dense(1)
    Training: Adam optimizer, EarlyStopping, ReduceLROnPlateau
    Inference: Recursive (one step at a time, appending predictions to window)
    """

    def __init__(self, state: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(state, config)
        self.lookback: int = self.config.get("lookback", settings.LSTM_LOOKBACK)
        self._scaler = None
        self._last_window: Optional[np.ndarray] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> "LSTMForecaster":
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler

        tf.random.set_seed(42)

        ts = train_df.sort_values(settings.DATE_COL)[settings.TARGET_COL].values.reshape(-1, 1)

        self._scaler = MinMaxScaler()
        ts_scaled = self._scaler.fit_transform(ts)

        X, y = self._create_sequences(ts_scaled)
        if len(X) == 0:
            raise ValueError(f"[LSTM][{self.state}] Not enough data (need > {self.lookback} rows).")

        self.model = self._build_model()
        callbacks = self._get_callbacks()

        self.model.fit(
            X, y,
            epochs=self.config.get("epochs", settings.LSTM_EPOCHS),
            batch_size=self.config.get("batch_size", settings.LSTM_BATCH_SIZE),
            validation_split=0.15,
            callbacks=callbacks,
            verbose=0,
        )

        # Keep last window for recursive prediction
        self._last_window = ts_scaled[-self.lookback:]

        self.is_fitted = True
        self.metadata = {"lookback": self.lookback}
        logger.info("[LSTM][%s] Trained on %d sequences", self.state, len(X))
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, horizon: int = settings.FORECAST_HORIZON) -> np.ndarray:
        self._assert_fitted()
        window = self._last_window.copy()
        preds_scaled = []

        for _ in range(horizon):
            x = window.reshape(1, self.lookback, 1)
            pred = self.model.predict(x, verbose=0)[0, 0]
            preds_scaled.append(pred)
            window = np.append(window[1:], [[pred]], axis=0)

        preds = self._scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        return np.maximum(preds, 0)

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path / "lstm_model.keras"))
        joblib.dump({
            "scaler": self._scaler,
            "last_window": self._last_window,
            "lookback": self.lookback,
            "metadata": self.metadata,
        }, path / "lstm_aux.joblib")
        logger.info("[LSTM][%s] Saved to %s", self.state, path)

    def load(self, path: Path) -> "LSTMForecaster":
        import tensorflow as tf
        self.model = tf.keras.models.load_model(str(path / "lstm_model.keras"))
        data = joblib.load(path / "lstm_aux.joblib")
        self._scaler = data["scaler"]
        self._last_window = data["last_window"]
        self.lookback = data["lookback"]
        self.metadata = data["metadata"]
        self.is_fitted = True
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def _build_model(self):
        import tensorflow as tf
        units = self.config.get("units", settings.LSTM_UNITS)
        dropout = self.config.get("dropout", settings.LSTM_DROPOUT)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.lookback, 1)),
            tf.keras.layers.LSTM(units[0], return_sequences=len(units) > 1),
            tf.keras.layers.Dropout(dropout),
            *([tf.keras.layers.LSTM(units[1])] if len(units) > 1 else []),
            *([tf.keras.layers.Dropout(dropout)] if len(units) > 1 else []),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def _get_callbacks(self) -> list:
        import tensorflow as tf
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.get("patience", settings.LSTM_PATIENCE),
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]
