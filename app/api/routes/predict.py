"""
predict.py - /predict and /models/compare endpoints.
"""

from fastapi import APIRouter, HTTPException
from app.schemas.prediction_schema import (
    PredictRequest,
    PredictResponse,
    ModelCompareRequest,
    ModelCompareResponse,
)
from app.services.prediction_service import PredictionService
from app.core.logger import get_logger

router = APIRouter(tags=["Prediction"])
logger = get_logger(__name__)
_service = PredictionService()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Forecast the next N weeks of sales for the given state.

    Uses the best model selected during training.
    """
    logger.info("POST /predict: state=%s horizon=%d", request.state, request.horizon)
    try:
        result = _service.predict(state=request.state, horizon=request.horizon)
        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


@router.get("/models/compare")
async def compare_models(state: str = None):
    """
    Return validation metrics for all trained models.

    Optionally filter by state.
    """
    try:
        return _service.compare_models(state=state)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
