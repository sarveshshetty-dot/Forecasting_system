"""
prediction_schema.py - Pydantic models for prediction endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    state: str = Field(..., example="California")
    horizon: int = Field(default=8, ge=1, le=52, description="Weeks to forecast")


class WeeklyForecast(BaseModel):
    week: int
    prediction: float


class PredictResponse(BaseModel):
    state: str
    best_model: str
    forecast_horizon_weeks: int
    forecast: List[WeeklyForecast]


class ModelCompareRequest(BaseModel):
    state: Optional[str] = None


class ModelMetrics(BaseModel):
    mae: float
    rmse: float
    mape: float


class ModelCompareResponse(BaseModel):
    results: Dict[str, Any]
