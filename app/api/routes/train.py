"""
train.py - /train endpoint: triggers full training pipeline.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.training_schema import TrainRequest, TrainResponse
from app.services.training_service import TrainingService
from app.core.logger import get_logger

router = APIRouter(tags=["Training"])
logger = get_logger(__name__)
_service = TrainingService()


@router.post("/train", response_model=TrainResponse)
async def train_models(request: TrainRequest):
    """
    Trigger the full training pipeline:

    1. Load dataset  
    2. Preprocess per state  
    3. Train all requested models  
    4. Evaluate on validation window  
    5. Select best model  
    6. Save artefacts  

    Returns training summary with metrics per state.
    """
    logger.info("POST /train received: %s", request.model_dump())
    try:
        data_path = Path(request.data_path) if request.data_path else None
        result = _service.train(
            data_path=data_path,
            models=request.models,
            parallel=request.parallel,
        )
        return TrainResponse(
            status="success",
            states_trained=result.get("states_trained", []),
            results=result.get("results", {}),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(exc)}")
