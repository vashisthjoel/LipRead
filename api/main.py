"""Main application entrypoint for LipRead FastAPI REST service.

Configures application metadata, lifespan lifecycle management, middleware,
error handling, and routes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import model_manager
from api.routes import router as api_router
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler managing application startup and shutdown.

    Loads the PyTorch InferenceEngine and checkpoint into memory on startup
    so it is immediately available for requests without per-call loading latency.
    """
    logger.info("Initializing LipRead API service...")
    success, error = model_manager.load()
    if success:
        logger.info(
            "Service startup complete | Model loaded on device: %s | Checkpoint: %s",
            model_manager.device_name,
            model_manager.checkpoint_name,
        )
    else:
        logger.warning(
            "Service started with degraded model state: %s. "
            "Inference requests will return 503 until a valid checkpoint is provided.",
            error,
        )

    yield

    logger.info("Shutting down LipRead API service...")
    model_manager.unload()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Application factory building the configured FastAPI instance."""
    app = FastAPI(
        title="LipRead Inference API",
        description=(
            "Asynchronous REST API for visual speech recognition (lip-reading) "
            "using spatiotemporal 3D CNN, Bi-LSTM temporal sequence modeling, "
            "and Connectionist Temporal Classification (CTC) decoding."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global unhandled exception handler (prevents leaking internal stack traces)
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception processing %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred while processing the request.",
                "error_code": "INTERNAL_SERVER_ERROR",
            },
        )

    # Root service descriptor
    @app.get(
        "/",
        tags=["Meta"],
        summary="Service index and metadata",
        description="Returns basic service metadata and documentation links.",
    )
    async def root() -> dict[str, str]:
        return {
            "service": "LipRead Inference API",
            "version": "0.1.0",
            "status": "online",
            "docs_url": "/docs",
            "openapi_url": "/openapi.json",
            "health_url": "/health",
        }

    # Register inference routes
    app.include_router(api_router)

    return app


# Module-level instance for ASGI servers (e.g. uvicorn api.main:app)
app = create_app()
