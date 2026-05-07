"""
visualization.py - Forecast, residual, and comparison plots.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def plot_forecast(
    state: str,
    history: pd.Series,
    forecast: np.ndarray,
    model_name: str,
    val_actual: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
) -> Path:
    """Plot historical data + forecast (and optionally validation actuals)."""
    save_dir = save_dir or settings.PLOTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(history.index, history.values, label="Historical", color="steelblue")

    last_date = history.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(forecast) + 1, freq=settings.FREQ)[1:]

    ax.plot(future_dates, forecast, label=f"{model_name} Forecast", color="darkorange",
            linestyle="--", marker="o", markersize=4)

    if val_actual is not None and len(val_actual) > 0:
        val_dates = future_dates[:len(val_actual)]
        ax.plot(val_dates, val_actual, label="Actual (Val)", color="green",
                linestyle="-", marker="x", markersize=5)

    ax.set_title(f"{state} — {model_name} Forecast ({len(forecast)} weeks)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()

    out_path = save_dir / f"{state}_{model_name}_forecast.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.debug("Saved forecast plot: %s", out_path)
    return out_path


def plot_model_comparison(
    state: str,
    metrics_dict: Dict[str, Dict[str, float]],
    save_dir: Optional[Path] = None,
) -> Path:
    """Bar chart comparing RMSE across models."""
    save_dir = save_dir or settings.PLOTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    models = list(metrics_dict.keys())
    rmses = [metrics_dict[m]["rmse"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    bars = ax.bar(models, rmses, color=colors[:len(models)])
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_title(f"{state} — Model RMSE Comparison")
    ax.set_ylabel("RMSE")
    plt.tight_layout()

    out_path = save_dir / f"{state}_model_comparison.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_residuals(
    state: str,
    model_name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    save_dir: Optional[Path] = None,
) -> Path:
    save_dir = save_dir or settings.PLOTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    residuals = actual - predicted
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(residuals, marker="o", linestyle="-", color="coral")
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set_title("Residuals over time")

    axes[1].hist(residuals, bins=20, color="steelblue", edgecolor="white")
    axes[1].set_title("Residual distribution")

    fig.suptitle(f"{state} — {model_name} Residuals")
    plt.tight_layout()
    out_path = save_dir / f"{state}_{model_name}_residuals.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
