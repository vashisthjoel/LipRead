"""Unit tests for LipRead FastAPI inference service.

Covers:
  - Service metadata (/)
  - Health check endpoint (/health) with loaded and unloaded model states
  - Request validation (missing file, empty file, unsupported extension, payload too large)
  - Parameter validation (decoding method choices)
  - Transcription pipeline (/predict) with mocked InferenceEngine
  - Temporary file cleanup verification
  - Service unavailable (503) when model is not loaded
  - Internal server error (500) on inference failure
  - OpenAPI and Swagger documentation endpoints (/docs, /openapi.json)
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.dependencies import ModelManager, get_inference_engine, get_model_manager
from api.main import create_app
from src.inference.engine import InferenceEngine, InferenceResult


@pytest.fixture
def mock_engine() -> MagicMock:
    """Fixture providing a mock InferenceEngine."""
    engine = MagicMock(spec=InferenceEngine)
    engine.device = "cpu"
    engine.decode_method = "greedy"
    engine.beam_width = 10
    engine.transcribe.return_value = InferenceResult(
        text="bin blue at f two now",
        confidence=-0.4521,
        elapsed_ms=350.0,
        num_frames=75,
        decode_method="greedy",
        error=None,
    )
    return engine


@pytest.fixture
def mock_manager(mock_engine: MagicMock) -> ModelManager:
    """Fixture providing a ModelManager preloaded with mock_engine."""
    manager = ModelManager()
    manager._engine = mock_engine
    manager._checkpoint_path = Path("E:/LipRead_Data/checkpoints/best_model.pt")
    manager._device = "cpu"
    manager._error = None
    return manager


@pytest.fixture
def client(mock_manager: ModelManager, mock_engine: MagicMock) -> TestClient:
    """TestClient configured with dependency overrides for fast, deterministic unit testing."""
    app = create_app()
    # Override dependencies so tests do not hit real heavy GPU weights
    app.dependency_overrides[get_model_manager] = lambda: mock_manager
    app.dependency_overrides[get_inference_engine] = lambda: mock_engine
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Metadata & Documentation Tests
# ---------------------------------------------------------------------------


def test_root_endpoint(client: TestClient) -> None:
    """Tests the root metadata endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "LipRead Inference API"
    assert data["status"] == "online"
    assert "docs_url" in data
    assert "health_url" in data


def test_openapi_json(client: TestClient) -> None:
    """Tests that OpenAPI 3.x specification is generated correctly."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "/health" in data["paths"]
    assert "/predict" in data["paths"]


def test_swagger_docs(client: TestClient) -> None:
    """Tests that the Swagger UI HTML documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


def test_health_when_model_loaded(client: TestClient, mock_manager: ModelManager) -> None:
    """Tests /health when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "LipRead"
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"
    assert data["checkpoint"] == "best_model.pt"


def test_health_when_model_not_loaded(client: TestClient, mock_manager: ModelManager) -> None:
    """Tests /health when model is unloaded/unavailable."""
    mock_manager._engine = None
    mock_manager._checkpoint_path = None
    mock_manager._device = None
    mock_manager._error = "Checkpoint not found"

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False
    assert data["device"] is None
    assert data["checkpoint"] is None


# ---------------------------------------------------------------------------
# Request Validation Tests
# ---------------------------------------------------------------------------


def test_predict_missing_file_upload(client: TestClient) -> None:
    """Tests /predict returns 422 when no file is uploaded."""
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_unsupported_file_extension(client: TestClient) -> None:
    """Tests /predict returns 415 for non-video extensions."""
    files = {"file": ("notes.txt", b"plain text content", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 415
    assert "Unsupported video format" in response.json()["detail"]


def test_predict_unsupported_audio_extension(client: TestClient) -> None:
    """Tests /predict returns 415 for audio-only uploads."""
    files = {"file": ("audio.wav", b"fake wav audio", "audio/wav")}
    response = client.post("/predict", files=files)
    assert response.status_code == 415
    assert "Unsupported video format" in response.json()["detail"]


def test_predict_empty_file_upload(client: TestClient) -> None:
    """Tests /predict returns 400 for empty (0 bytes) file."""
    files = {"file": ("empty.mp4", b"", "video/mp4")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_invalid_method(client: TestClient) -> None:
    """Tests /predict returns 400 when an invalid decoding method is specified."""
    files = {"file": ("video.mp4", b"dummy video bytes", "video/mp4")}
    response = client.post("/predict?method=quantum", files=files)
    assert response.status_code == 400
    assert "Invalid decoding method" in response.json()["detail"]


def test_predict_payload_too_large(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests /predict returns 413 when uploaded file exceeds max size limit."""
    import api.routes as routes_module

    # Set threshold temporarily low for testing
    monkeypatch.setattr(routes_module, "MAX_FILE_SIZE_BYTES", 100)
    files = {"file": ("large.mp4", b"x" * 200, "video/mp4")}
    response = client.post("/predict", files=files)
    assert response.status_code == 413
    assert "exceeds maximum limit" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Prediction & Inference Pipeline Tests
# ---------------------------------------------------------------------------


def test_predict_greedy_success(client: TestClient, mock_engine: MagicMock) -> None:
    """Tests /predict with greedy decoding returns expected schema and fields."""
    files = {"file": ("speaker_video.mp4", b"dummy video stream", "video/mp4")}
    response = client.post("/predict?method=greedy", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["transcription"] == "bin blue at f two now"
    assert data["method"] == "greedy"
    assert data["confidence"] == -0.4521
    assert data["processing_time_seconds"] == 0.35
    assert data["num_frames"] == 75

    # Verify existing engine was invoked with correct arguments
    mock_engine.transcribe.assert_called_once()
    call_kwargs = mock_engine.transcribe.call_args.kwargs
    assert call_kwargs["decode_method"] == "greedy"
    assert call_kwargs["beam_width"] == 10


def test_predict_beam_search_success(client: TestClient, mock_engine: MagicMock) -> None:
    """Tests /predict with beam search decoding and custom beam width."""
    mock_engine.transcribe.return_value = InferenceResult(
        text="set white by p three please",
        confidence=-0.3120,
        elapsed_ms=520.0,
        num_frames=75,
        decode_method="beam",
        error=None,
    )
    files = {"file": ("test_clip.avi", b"dummy avi video", "video/avi")}
    response = client.post("/predict?method=beam&beam_width=25", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["transcription"] == "set white by p three please"
    assert data["method"] == "beam"
    assert data["confidence"] == -0.3120
    assert data["processing_time_seconds"] == 0.52
    assert data["num_frames"] == 75

    mock_engine.transcribe.assert_called_once()
    call_kwargs = mock_engine.transcribe.call_args.kwargs
    assert call_kwargs["decode_method"] == "beam"
    assert call_kwargs["beam_width"] == 25


def test_predict_model_not_loaded_503() -> None:
    """Tests /predict returns 503 Service Unavailable when the model is not loaded."""
    app = create_app()
    empty_manager = ModelManager()
    empty_manager._error = "Checkpoint best_model.pt not found on storage."
    app.dependency_overrides[get_model_manager] = lambda: empty_manager

    with TestClient(app) as test_client:
        files = {"file": ("video.mp4", b"dummy video bytes", "video/mp4")}
        response = test_client.post("/predict", files=files)
        assert response.status_code == 503
        assert "Inference service unavailable" in response.json()["detail"]


def test_predict_inference_failure_500(client: TestClient, mock_engine: MagicMock) -> None:
    """Tests /predict returns 500 when underlying InferenceEngine reports failure."""
    mock_engine.transcribe.return_value = InferenceResult(
        text="",
        confidence=float("-inf"),
        elapsed_ms=15.0,
        num_frames=0,
        decode_method="greedy",
        error="MediaPipe failed to detect facial landmarks across all frames",
    )
    files = {"file": ("unclear_video.mp4", b"dummy video bytes", "video/mp4")}
    response = client.post("/predict", files=files)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "Inference processing failed" in detail
    assert "MediaPipe" in detail


def test_predict_cleans_up_temporary_file(client: TestClient, mock_engine: MagicMock) -> None:
    """Verifies that uploaded temporary video file is deleted even if inference fails."""
    recorded_path: list[Path] = []

    def mock_transcribe_side_effect(video_path: Path, **kwargs):
        recorded_path.append(video_path)
        assert video_path.exists(), "Temporary file should exist during inference call"
        return InferenceResult(
            text="hello",
            confidence=-0.1,
            elapsed_ms=10.0,
            num_frames=75,
            decode_method="greedy",
            error=None,
        )

    mock_engine.transcribe.side_effect = mock_transcribe_side_effect

    files = {"file": ("cleanup_test.mp4", b"dummy video data", "video/mp4")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200

    assert len(recorded_path) == 1
    temp_file = recorded_path[0]
    assert not temp_file.exists(), f"Temporary file {temp_file} was not cleaned up after request!"
