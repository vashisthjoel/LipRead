"""API route definitions for LipRead FastAPI service.

Exposes:
  - GET  /health   : Service health status and model availability
  - POST /predict  : Video upload, validation, preprocessing, and CTC transcription
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from api.dependencies import (
    ModelManager,
    get_contextual_corrector,
    get_inference_engine,
    get_model_manager,
)
from api.schemas import ErrorResponse, HealthResponse, PredictionResponse
from src.inference.engine import InferenceEngine
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.api.routes")

router = APIRouter(tags=["Inference"])

# Supported video extensions
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}

# Max allowable file upload: 50 MiB (typical 3-second lip-reading videos are < 5 MiB)
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MiB chunk reading


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check service and model health",
    description="Returns the operational status of the API and indicates whether the model is loaded in memory.",
)
async def health(
    manager: ModelManager = Depends(get_model_manager),
) -> HealthResponse:
    """Service health probe endpoint."""
    is_loaded = manager.is_loaded
    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        service="LipRead",
        model_loaded=is_loaded,
        device=manager.device_name,
        checkpoint=manager.checkpoint_name,
        model_version="0.1.0",
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request (e.g. empty file or invalid method)"},
        413: {"model": ErrorResponse, "description": "Payload Too Large (exceeds 50 MiB)"},
        415: {"model": ErrorResponse, "description": "Unsupported Media Type"},
        500: {"model": ErrorResponse, "description": "Inference / Preprocessing Failure"},
        503: {"model": ErrorResponse, "description": "Model Not Loaded / Checkpoint Unavailable"},
    },
    summary="Transcribe video lip-reading sequence",
    description=(
        "Uploads a video file (.mp4, .avi, .mov), extracts mouth ROI frames, "
        "processes with 3D CNN + Bi-LSTM, and returns decoded CTC transcription."
    ),
)
async def predict(
    file: UploadFile = File(..., description="Video file to transcribe"),
    method: str = Query(
        default="greedy",
        description="CTC decoding algorithm ('greedy' or 'beam').",
    ),
    beam_width: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Beam width for prefix beam search (only used when method='beam').",
    ),
    use_correction: bool = Query(
        default=True,
        description="Whether to apply LLM contextual post-processing if GenAI is enabled.",
    ),
    engine: InferenceEngine = Depends(get_inference_engine),
    corrector=Depends(get_contextual_corrector),
) -> PredictionResponse:
    """Endpoint handling video upload and speech transcription."""
    # 1. Validate decode method
    norm_method = method.strip().lower()
    if norm_method not in ("greedy", "beam"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid decoding method '{method}'. Supported methods: 'greedy', 'beam'.",
        )

    # 2. Validate filename and extension
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if not suffix or suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported video format '{suffix or 'unknown'}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    # 3. Stream and write to safe temporary file with size limit
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=suffix, prefix="lipread_upload_")
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)

    try:
        total_bytes = 0
        with open(tmp_path, "wb") as out_file:
            while chunk := await file.read(CHUNK_SIZE_BYTES):
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=(
                            f"Uploaded file exceeds maximum limit of "
                            f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MiB."
                        ),
                    )
                out_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

        logger.info(
            "Received video upload: filename='%s' | size=%d bytes | method=%s (beam_width=%d, use_correction=%s)",
            filename,
            total_bytes,
            norm_method,
            beam_width,
            use_correction,
        )

        # 4. Invoke existing InferenceEngine
        result = engine.transcribe(
            video_path=tmp_path,
            decode_method=norm_method,
            beam_width=beam_width,
        )

        if not result.success:
            logger.error("InferenceEngine failed on %s: %s", filename, result.error)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference processing failed: {result.error}",
            )

        # 5. Apply GenAI contextual correction if enabled and requested
        raw_text = result.text
        if use_correction and corrector is not None and getattr(corrector, "is_active", False):
            c_result = corrector.correct(raw_text)
            final_transcription = c_result.effective_text
            corrected_text = c_result.corrected_transcription
            applied = c_result.correction_applied
            model_id = f"{c_result.provider}:{c_result.model}" if c_result.provider else c_result.model
        else:
            final_transcription = raw_text
            corrected_text = raw_text
            applied = False
            model_id = None

        # 6. Build clean, authentic response
        processing_sec = round(result.elapsed_ms / 1000.0, 4)
        confidence_val = (
            round(result.confidence, 4)
            if result.confidence != float("-inf")
            else None
        )

        return PredictionResponse(
            transcription=final_transcription,
            raw_transcription=raw_text,
            corrected_transcription=corrected_text,
            correction_applied=applied,
            correction_model=model_id,
            method=result.decode_method,
            confidence=confidence_val,
            processing_time_seconds=processing_sec,
            num_frames=result.num_frames,
        )

    finally:
        # 6. Guaranteed temporary file cleanup
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as cleanup_err:
                logger.warning(
                    "Failed to delete temporary upload file %s: %s",
                    tmp_path,
                    cleanup_err,
                )
