"""api.py - Registers all route modules."""

from fastapi import APIRouter
from app.api.routes import health, train, predict

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(train.router)
api_router.include_router(predict.router)
