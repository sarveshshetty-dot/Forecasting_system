"""
training_schema.py - Pydantic models for the training endpoint.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from app.core.constants import ALL_MODELS


class TrainRequest(BaseModel):
    data_path: Optional[str] = Field(default=None, description="Override default data path")
    models: Optional[List[str]] = Field(
        default=None,
        description=f"Subset of models to train. Options: {ALL_MODELS}",
        example=["xgboost", "prophet"],
    )
    parallel: bool = Field(default=False, description="Train states in parallel threads")


class TrainResponse(BaseModel):
    status: str
    states_trained: List[str]
    results: dict
