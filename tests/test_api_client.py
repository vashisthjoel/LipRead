"""Unit tests for the LipRead Streamlit API Client and UI components.

Covers:
  - Client initialization and URL normalization
  - Health check endpoint queries (success, degraded, offline, timeout)
  - /predict request parameter construction (greedy and beam with beam_width)
  - Response parsing to PredictionResult dataclass
  - Typed error classification (400, 413, 415, 422, 500, 503)
  - Connection errors and timeout handling
  - Malformed JSON handling
  - Streamlit application module importability check
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.api_client import (
    APIConnectionError,
    APIResponseError,
    APITimeoutError,
    HealthStatus,
    LipReadAPIClient,
    LipReadAPIError,
    PredictionResult,
)


@pytest.fixture
def mock_session() -> MagicMock:
    """Fixture providing a mocked requests.Session."""
    return MagicMock(spec=requests.Session)


@pytest.fixture
def client(mock_session: MagicMock) -> LipReadAPIClient:
    """Fixture providing a LipReadAPIClient using mock_session."""
    return LipReadAPIClient(
        base_url="http://testserver:8000/",
        timeout=10.0,
        session=mock_session,
    )


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


def test_client_initialization_strips_trailing_slash() -> None:
    """Verifies that trailing slashes are removed from the base URL."""
    c = LipReadAPIClient(base_url="http://localhost:8000///")
    assert c.base_url == "http://localhost:8000"


def test_client_default_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies default base URL resolution from environment or fallback."""
    monkeypatch.delenv("LIPREAD_API_URL", raising=False)
    c = LipReadAPIClient()
    assert c.base_url == "http://localhost:8000"


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


def test_check_health_healthy(client: LipReadAPIClient, mock_session: MagicMock) -> None:
    """Tests health check when backend is online and model is loaded."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "ok",
        "service": "LipRead",
        "model_loaded": True,
        "device": "cuda:0",
        "checkpoint": "best_model.pt",
        "model_version": "0.1.0",
    }
    mock_session.get.return_value = mock_response

    status: HealthStatus = client.check_health()
    assert status.is_healthy is True
    assert status.reachable is True
    assert status.model_loaded is True
    assert status.device == "cuda:0"
    assert status.checkpoint == "best_model.pt"
    mock_session.get.assert_called_once_with("http://testserver:8000/health", timeout=5.0)


def test_check_health_degraded_model_unloaded(
    client: LipReadAPIClient, mock_session: MagicMock
) -> None:
    """Tests health check when backend is online but model is unloaded."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "degraded",
        "service": "LipRead",
        "model_loaded": False,
        "device": None,
        "checkpoint": None,
        "model_version": "0.1.0",
    }
    mock_session.get.return_value = mock_response

    status: HealthStatus = client.check_health()
    assert status.is_healthy is False
    assert status.reachable is True
    assert status.model_loaded is False
    assert status.status == "degraded"


def test_check_health_connection_error(
    client: LipReadAPIClient, mock_session: MagicMock
) -> None:
    """Tests health check when backend connection fails."""
    mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")

    status: HealthStatus = client.check_health()
    assert status.is_healthy is False
    assert status.reachable is False
    assert "Cannot connect to FastAPI server" in (status.error_message or "")


def test_check_health_timeout(client: LipReadAPIClient, mock_session: MagicMock) -> None:
    """Tests health check when request times out."""
    mock_session.get.side_effect = requests.exceptions.Timeout("Read timed out")

    status: HealthStatus = client.check_health()
    assert status.is_healthy is False
    assert status.reachable is False
    assert "timed out" in (status.error_message or "")


# ---------------------------------------------------------------------------
# /predict Tests
# ---------------------------------------------------------------------------


def test_predict_greedy_success(client: LipReadAPIClient, mock_session: MagicMock) -> None:
    """Tests successful prediction with greedy decoding."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "transcription": "bin blue at f two now",
        "method": "greedy",
        "confidence": -0.4125,
        "processing_time_seconds": 0.385,
        "num_frames": 75,
    }
    mock_session.post.return_value = mock_response

    result: PredictionResult = client.predict(
        video_bytes=b"dummy_video_bytes",
        filename="clip.mp4",
        method="greedy",
    )

    assert result.transcription == "bin blue at f two now"
    assert result.method == "greedy"
    assert result.confidence == -0.4125
    assert result.processing_time_seconds == 0.385
    assert result.num_frames == 75

    mock_session.post.assert_called_once()
    _, kwargs = mock_session.post.call_args
    assert kwargs["params"] == {"method": "greedy"}
    assert "file" in kwargs["files"]
    assert kwargs["files"]["file"][0] == "clip.mp4"
    assert kwargs["files"]["file"][1] == b"dummy_video_bytes"


def test_predict_beam_search_success(client: LipReadAPIClient, mock_session: MagicMock) -> None:
    """Tests prediction with beam search passing beam_width parameter."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "transcription": "lay green in zero soon",
        "method": "beam",
        "confidence": -0.3210,
        "processing_time_seconds": 0.512,
        "num_frames": 75,
    }
    mock_session.post.return_value = mock_response

    result = client.predict(
        video_bytes=b"sample_avi_bytes",
        filename="sequence.avi",
        method="beam",
        beam_width=25,
    )

    assert result.transcription == "lay green in zero soon"
    assert result.method == "beam"
    assert result.confidence == -0.3210

    _, kwargs = mock_session.post.call_args
    assert kwargs["params"] == {"method": "beam", "beam_width": 25}


def test_predict_connection_error(client: LipReadAPIClient, mock_session: MagicMock) -> None:
    """Tests that connection failure raises APIConnectionError."""
    mock_session.post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    with pytest.raises(APIConnectionError) as exc_info:
        client.predict(b"bytes", "video.mp4")
    assert "Failed to connect to LipRead API" in str(exc_info.value)


def test_predict_timeout(client: LipReadAPIClient, mock_session: MagicMock) -> None:
    """Tests that timeout raises APITimeoutError."""
    mock_session.post.side_effect = requests.exceptions.Timeout("Request timed out")

    with pytest.raises(APITimeoutError) as exc_info:
        client.predict(b"bytes", "video.mp4")
    assert "timed out" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "status_code,detail,prop",
    [
        (400, "Uploaded file is empty (0 bytes).", "is_bad_request"),
        (413, "Uploaded file exceeds maximum limit of 50 MiB.", "is_payload_too_large"),
        (415, "Unsupported video format '.txt'.", "is_unsupported_media_type"),
        (422, "Field 'file' is required.", "is_validation_error"),
        (500, "Inference processing failed: Preprocessing error", "is_server_error"),
        (503, "Inference service unavailable: Checkpoint missing", "is_service_unavailable"),
    ],
)
def test_predict_http_error_codes(
    client: LipReadAPIClient,
    mock_session: MagicMock,
    status_code: int,
    detail: str,
    prop: str,
) -> None:
    """Tests that HTTP error status codes are mapped to APIResponseError with properties."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = status_code
    mock_response.json.return_value = {"detail": detail}
    mock_session.post.return_value = mock_response

    with pytest.raises(APIResponseError) as exc_info:
        client.predict(b"bytes", "video.mp4")

    err = exc_info.value
    assert err.status_code == status_code
    assert err.detail == detail
    assert getattr(err, prop) is True


def test_predict_malformed_json_response(
    client: LipReadAPIClient, mock_session: MagicMock
) -> None:
    """Tests that malformed JSON in a 200 response raises LipReadAPIError."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"unexpected_key": "some_value"}
    mock_session.post.return_value = mock_response

    with pytest.raises(LipReadAPIError) as exc_info:
        client.predict(b"bytes", "video.mp4")
    assert "Malformed JSON" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Streamlit Module Importability
# ---------------------------------------------------------------------------


def test_streamlit_app_importable() -> None:
    """Verifies that app.streamlit_app can be imported without syntax or import errors."""
    import app.streamlit_app

    assert hasattr(app.streamlit_app, "main")
    assert hasattr(app.streamlit_app, "render_sidebar")
