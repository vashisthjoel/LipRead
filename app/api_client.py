"""API Client abstraction for communicating with the LipRead FastAPI service.

Handles network calls, parameter serialization, response parsing, and typed
error classification without leaking internal server stack traces to user interfaces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

DEFAULT_API_BASE_URL = os.environ.get("LIPREAD_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT_SECONDS = 90.0


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class LipReadAPIError(Exception):
    """Base exception for all LipRead API communication errors."""


class APIConnectionError(LipReadAPIError):
    """Raised when the client cannot establish a connection to the FastAPI server."""


class APITimeoutError(LipReadAPIError):
    """Raised when an API request exceeds the configured network timeout."""


class APIResponseError(LipReadAPIError):
    """Raised when the FastAPI server responds with an HTTP error status code."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")

    @property
    def is_bad_request(self) -> bool:
        return self.status_code == 400

    @property
    def is_payload_too_large(self) -> bool:
        return self.status_code == 413

    @property
    def is_unsupported_media_type(self) -> bool:
        return self.status_code == 415

    @property
    def is_validation_error(self) -> bool:
        return self.status_code == 422

    @property
    def is_server_error(self) -> bool:
        return self.status_code == 500

    @property
    def is_service_unavailable(self) -> bool:
        return self.status_code == 503


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class HealthStatus:
    """Status container matching the /health response schema."""

    status: str
    service: str
    model_loaded: bool
    device: Optional[str] = None
    checkpoint: Optional[str] = None
    model_version: Optional[str] = None
    reachable: bool = True
    error_message: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        return self.reachable and self.status == "ok" and self.model_loaded


@dataclass
class PredictionResult:
    """Result container matching the /predict response schema."""

    transcription: str
    method: str
    confidence: Optional[float]
    processing_time_seconds: float
    num_frames: int
    raw_transcription: Optional[str] = None
    corrected_transcription: Optional[str] = None
    correction_applied: bool = False
    correction_model: Optional[str] = None


# ---------------------------------------------------------------------------
# Client Implementation
# ---------------------------------------------------------------------------


class LipReadAPIClient:
    """Client for LipRead FastAPI REST service."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_API_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def check_health(self) -> HealthStatus:
        """Queries the /health endpoint and returns the service health state.

        Does not raise exceptions; returns a HealthStatus with reachable=False
        and descriptive error_message upon failure.
        """
        url = f"{self.base_url}/health"
        try:
            resp = self.session.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return HealthStatus(
                    status=data.get("status", "unknown"),
                    service=data.get("service", "LipRead"),
                    model_loaded=bool(data.get("model_loaded", False)),
                    device=data.get("device"),
                    checkpoint=data.get("checkpoint"),
                    model_version=data.get("model_version"),
                    reachable=True,
                )
            # Degraded / Error response from health endpoint
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text or f"HTTP {resp.status_code}"

            return HealthStatus(
                status="error",
                service="LipRead",
                model_loaded=False,
                reachable=True,
                error_message=f"HTTP {resp.status_code}: {detail}",
            )

        except requests.exceptions.ConnectionError:
            return HealthStatus(
                status="unreachable",
                service="LipRead",
                model_loaded=False,
                reachable=False,
                error_message=(
                    f"Cannot connect to FastAPI server at '{self.base_url}'. "
                    "Please ensure the backend service is running."
                ),
            )
        except requests.exceptions.Timeout:
            return HealthStatus(
                status="timeout",
                service="LipRead",
                model_loaded=False,
                reachable=False,
                error_message=f"Health probe timed out after 5.0 seconds.",
            )
        except Exception as exc:
            return HealthStatus(
                status="error",
                service="LipRead",
                model_loaded=False,
                reachable=False,
                error_message=f"Unexpected health check error: {str(exc)}",
            )

    def predict(
        self,
        video_bytes: bytes,
        filename: str,
        method: str = "greedy",
        beam_width: int = 10,
        use_correction: Optional[bool] = None,
    ) -> PredictionResult:
        """Sends a video file to /predict and returns the parsed PredictionResult.

        Args:
            video_bytes: Raw binary content of the video file.
            filename: Original file name (used for extension checking).
            method: CTC decoding method ('greedy' or 'beam').
            beam_width: Beam width for prefix beam search (only used if method='beam').
            use_correction: Optional flag to toggle LLM contextual correction.
                If None, relies on server default configuration.

        Returns:
            PredictionResult with transcription and timing metadata.

        Raises:
            APIConnectionError: If server cannot be reached.
            APITimeoutError: If request exceeds network timeout.
            APIResponseError: If server returns an HTTP 4xx or 5xx status code.
        """
        url = f"{self.base_url}/predict"
        norm_method = method.strip().lower()

        params: Dict[str, Any] = {"method": norm_method}
        if norm_method == "beam":
            params["beam_width"] = beam_width
        if use_correction is not None:
            params["use_correction"] = use_correction

        files = {"file": (filename, video_bytes, "video/mp4")}

        try:
            resp = self.session.post(
                url,
                params=params,
                files=files,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise APIConnectionError(
                f"Failed to connect to LipRead API at '{self.base_url}'. "
                "Ensure FastAPI service is active (`uvicorn api.main:app --port 8000`)."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise APITimeoutError(
                f"Inference request timed out after {self.timeout}s. "
                "The video sequence may be too large or the server is overloaded."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LipReadAPIError(f"Network transport error: {str(exc)}") from exc

        # Handle HTTP error responses
        if resp.status_code != 200:
            detail: str
            try:
                err_json = resp.json()
                detail = err_json.get("detail", str(err_json))
            except Exception:
                detail = resp.text or f"HTTP status {resp.status_code}"

            raise APIResponseError(status_code=resp.status_code, detail=detail)

        # Parse success JSON
        try:
            data = resp.json()
            return PredictionResult(
                transcription=data["transcription"],
                method=data.get("method", norm_method),
                confidence=data.get("confidence"),
                processing_time_seconds=float(data["processing_time_seconds"]),
                num_frames=int(data["num_frames"]),
                raw_transcription=data.get("raw_transcription", data["transcription"]),
                corrected_transcription=data.get("corrected_transcription"),
                correction_applied=bool(data.get("correction_applied", False)),
                correction_model=data.get("correction_model"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise LipReadAPIError(
                f"Malformed JSON response received from API: {resp.text}"
            ) from exc

