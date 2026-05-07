"""
main.py - FastAPI application factory and entry point.
"""

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.api import api_router
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Production-grade Time Series Forecasting System. "
            "Trains SARIMA, Prophet, XGBoost, and LSTM models per US state "
            "and exposes predictions via REST API."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ────────────────────────────────────────────────────────────

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logger(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        logger.info("[%s] %s %s", request_id, request.method, request.url.path)
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info("[%s] %s completed in %.1f ms", request_id, request.url.path, elapsed)
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Exception handlers ────────────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # ── Routes ────────────────────────────────────────────────────────────────

    app.include_router(api_router)

    @app.on_event("startup")
    async def startup_event():
        logger.info("🚀 %s v%s starting …", settings.PROJECT_NAME, settings.VERSION)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.WORKERS,
        reload=settings.DEBUG,
        log_level="info",
    )
