"""Unit and integration tests for Stage 9: GenAI Contextual Post-Processing.

Covers:
  - Correction disabled mode (raw == corrected, correction_applied=False)
  - Mocked deterministic LLM correction (viseme substitution)
  - Empty and whitespace-only transcript handling
  - LLM invocation failure and graceful fallback (never crashes)
  - Malformed LLM responses (blank output, quote stripping)
  - Strict byte-for-byte preservation of raw transcription
  - Prompt construction & constraint verification
  - Environment variable configuration parsing
  - Provider factory dispatch and error handling
  - FastAPI /predict integration with use_correction=true
  - FastAPI /predict integration with use_correction=false
  - API client response deserialization and parameter passing

All tests operate 100% hermetically offline without network calls or API keys.
"""

from __future__ import annotations

import io
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from pathlib import Path

from api.dependencies import (
    ModelManager,
    get_contextual_corrector,
    get_model_manager,
)
from api.main import app
from app.api_client import LipReadAPIClient, PredictionResult
from src.genai.config import GenAIConfig
from src.genai.corrector import ContextualCorrector
from src.genai.prompts import get_correction_prompt
from src.genai.providers import MockChatModel, create_chat_model
from src.genai.schemas import CorrectionResult
from src.inference.engine import InferenceEngine, InferenceResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def disabled_corrector() -> ContextualCorrector:
    """Corrector configured with enabled=False."""
    cfg = GenAIConfig(enabled=False)
    return ContextualCorrector(config=cfg)


@pytest.fixture
def mock_corrector() -> ContextualCorrector:
    """Corrector configured with mock provider."""
    cfg = GenAIConfig(enabled=True, provider="mock", model="mock-viseme-corrector")
    return ContextualCorrector(config=cfg)


# ---------------------------------------------------------------------------
# 1. Disabled Mode Tests
# ---------------------------------------------------------------------------


def test_correction_disabled_returns_raw_transcript(disabled_corrector: ContextualCorrector) -> None:
    """When GenAI is disabled, raw_transcription == corrected_transcription and applied=False."""
    raw_input = "bin blue at f two now"
    result: CorrectionResult = disabled_corrector.correct(raw_input)

    assert result.raw_transcription == raw_input
    assert result.corrected_transcription == raw_input
    assert result.correction_applied is False
    assert result.provider is None
    assert result.model is None
    assert result.error is None
    assert result.latency_seconds == 0.0
    assert result.effective_text == raw_input


# ---------------------------------------------------------------------------
# 2. Mocked Successful LLM Correction Tests
# ---------------------------------------------------------------------------


def test_mock_successful_correction(mock_corrector: ContextualCorrector) -> None:
    """When enabled with mock provider, corrects known viseme substitutions."""
    raw_input = "lay freen in zero soon"
    result: CorrectionResult = mock_corrector.correct(raw_input)

    assert result.raw_transcription == raw_input
    assert result.corrected_transcription == "lay green in zero soon"
    assert result.correction_applied is True
    assert result.provider == "mock"
    assert result.model == "mock-viseme-corrector"
    assert result.latency_seconds >= 0.0
    assert result.error is None
    assert result.effective_text == "lay green in zero soon"


def test_mock_no_correction_needed(mock_corrector: ContextualCorrector) -> None:
    """If transcript does not need corrections, correction_applied is False."""
    clean_input = "bin blue at f two now"
    result: CorrectionResult = mock_corrector.correct(clean_input)

    assert result.raw_transcription == clean_input
    assert result.corrected_transcription == clean_input
    assert result.correction_applied is False


# ---------------------------------------------------------------------------
# 3. Empty & Whitespace Transcript Handling
# ---------------------------------------------------------------------------


def test_empty_transcript_handling(mock_corrector: ContextualCorrector) -> None:
    """Empty string is returned immediately without LLM invocation."""
    result: CorrectionResult = mock_corrector.correct("")

    assert result.raw_transcription == ""
    assert result.corrected_transcription == ""
    assert result.correction_applied is False
    assert result.latency_seconds == 0.0


def test_whitespace_transcript_handling(mock_corrector: ContextualCorrector) -> None:
    """Whitespace-only transcript is returned cleaned without LLM invocation."""
    result: CorrectionResult = mock_corrector.correct("    \n\t  ")

    assert result.raw_transcription == "    \n\t  "
    assert result.corrected_transcription == "    \n\t  "
    assert result.correction_applied is False
    assert result.latency_seconds == 0.0


# ---------------------------------------------------------------------------
# 4. Error Handling & Graceful Fallback
# ---------------------------------------------------------------------------


def test_correction_failure_falls_back_to_raw() -> None:
    """When the underlying model raises an exception, corrector never crashes and preserves raw."""
    cfg = GenAIConfig(enabled=True, provider="mock")
    corrector = ContextualCorrector(config=cfg)

    with patch.object(corrector, "_chain") as mock_chain:
        mock_chain.invoke.side_effect = RuntimeError("Simulated LLM API timeout")
        raw_text = "lay freen in zero soon"
        result: CorrectionResult = corrector.correct(raw_text)

        assert result.raw_transcription == raw_text
        assert result.corrected_transcription == raw_text
        assert result.correction_applied is False
        assert result.error is not None
        assert "Simulated LLM API timeout" in result.error
        assert result.effective_text == raw_text


# ---------------------------------------------------------------------------
# 5. Malformed Response Handling
# ---------------------------------------------------------------------------


def test_malformed_blank_response_falls_back() -> None:
    """If the LLM returns an empty string or whitespace, falls back to raw."""
    cfg = GenAIConfig(enabled=True, provider="mock")
    corrector = ContextualCorrector(config=cfg)

    with patch.object(corrector, "_chain") as mock_chain:
        mock_chain.invoke.return_value = "   "
        raw_text = "place red by b three please"
        result = corrector.correct(raw_text)

        assert result.raw_transcription == raw_text
        assert result.corrected_transcription == raw_text
        assert result.correction_applied is False


def test_wrapping_quotes_and_markdown_stripped() -> None:
    """Enclosing quotation marks and markdown fences are cleanly stripped."""
    cfg = GenAIConfig(enabled=True, provider="mock")
    corrector = ContextualCorrector(config=cfg)

    with patch.object(corrector, "_chain") as mock_chain:
        mock_chain.invoke.return_value = ' "bin blue at f two now" '
        raw_text = "bin blue at s two now"
        result = corrector.correct(raw_text)

        assert result.corrected_transcription == "bin blue at f two now"
        assert result.correction_applied is True


# ---------------------------------------------------------------------------
# 6. Preservation of Raw Transcript
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_input",
    [
        "bin blue at a one now",
        "set red in b two soon",
        "place green with c three again",
        "lay white by d four please",
    ],
)
def test_raw_transcript_preservation_across_conditions(raw_input: str) -> None:
    """Raw transcript must strictly equal original input in all scenarios."""
    cfg_disabled = GenAIConfig(enabled=False)
    corrector_dis = ContextualCorrector(config=cfg_disabled)
    assert corrector_dis.correct(raw_input).raw_transcription == raw_input

    cfg_mock = GenAIConfig(enabled=True, provider="mock")
    corrector_mock = ContextualCorrector(config=cfg_mock)
    assert corrector_mock.correct(raw_input).raw_transcription == raw_input


# ---------------------------------------------------------------------------
# 7. Prompt Design Constraints
# ---------------------------------------------------------------------------


def test_correction_prompt_contains_required_directives() -> None:
    """Prompt template must enforce non-inventive, phonetic-specific corrections."""
    prompt = get_correction_prompt()
    template_str = prompt.format(raw_transcript="test input")

    assert "lip-reading transcript" in template_str
    assert "phonetic" in template_str.lower() or "viseme" in template_str.lower()
    assert "preserve" in template_str.lower()
    assert "do not" in template_str.lower()
    assert "test input" in template_str


# ---------------------------------------------------------------------------
# 8. Configuration Parsing
# ---------------------------------------------------------------------------


def test_config_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default configuration reads safe defaults when env is unset."""
    monkeypatch.delenv("GENAI_ENABLED", raising=False)
    monkeypatch.delenv("LIPREAD_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LIPREAD_LLM_MODEL", raising=False)

    cfg = GenAIConfig.from_env()
    assert cfg.enabled is False
    assert cfg.provider == "mock"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.temperature == 0.0


def test_config_from_env_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom environment variables are correctly mapped."""
    monkeypatch.setenv("GENAI_ENABLED", "true")
    monkeypatch.setenv("LIPREAD_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LIPREAD_LLM_MODEL", "llama3")
    monkeypatch.setenv("LIPREAD_LLM_API_BASE", "http://localhost:11434")
    monkeypatch.setenv("LIPREAD_LLM_TEMPERATURE", "0.2")

    cfg = GenAIConfig.from_env()
    assert cfg.enabled is True
    assert cfg.provider == "ollama"
    assert cfg.model == "llama3"
    assert cfg.api_base == "http://localhost:11434"
    assert cfg.temperature == 0.2


# ---------------------------------------------------------------------------
# 9. Provider Factory
# ---------------------------------------------------------------------------


def test_provider_factory_unsupported_raises() -> None:
    """Unknown provider raises a clear ValueError."""
    cfg = GenAIConfig(enabled=True, provider="unknown_llm_provider")
    with pytest.raises(ValueError, match="Unsupported LLM provider 'unknown_llm_provider'"):
        create_chat_model(cfg)


def test_provider_factory_mock_returns_mock_model() -> None:
    """Mock provider returns MockChatModel instance."""
    cfg = GenAIConfig(enabled=True, provider="mock")
    model = create_chat_model(cfg)
    assert isinstance(model, MockChatModel)


# ---------------------------------------------------------------------------
# 10. FastAPI API Integration Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def api_test_client() -> Generator[TestClient, None, None]:
    """TestClient with mocked engine and mock corrector."""
    mock_eng = MagicMock(spec=InferenceEngine)
    mock_eng.device = "cpu"
    mock_eng.decode_method = "greedy"
    mock_eng.beam_width = 10
    mock_eng.transcribe.return_value = InferenceResult(
        text="lay freen in zero soon",
        confidence=-0.123,
        elapsed_ms=45.0,
        num_frames=75,
        decode_method="greedy",
    )

    manager = ModelManager()
    manager._engine = mock_eng
    manager._checkpoint_path = Path("E:/LipRead_Data/checkpoints/best_model.pt")
    manager._device = "cpu"
    manager._error = None

    mock_corr = ContextualCorrector(
        config=GenAIConfig(enabled=True, provider="mock", model="mock-viseme-corrector")
    )

    app.dependency_overrides[get_model_manager] = lambda: manager
    app.dependency_overrides[get_contextual_corrector] = lambda: mock_corr

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()


def test_api_predict_with_correction_enabled(api_test_client: TestClient) -> None:
    """POST /predict?use_correction=true applies contextual correction."""
    fake_video = b"dummy_mp4_bytes_for_testing"
    resp = api_test_client.post(
        "/predict",
        params={"method": "greedy", "use_correction": True},
        files={"file": ("sample.mp4", io.BytesIO(fake_video), "video/mp4")},
    )

    assert resp.status_code == 200
    data = resp.json()

    assert data["raw_transcription"] == "lay freen in zero soon"
    assert data["corrected_transcription"] == "lay green in zero soon"
    assert data["correction_applied"] is True
    assert data["correction_model"] == "mock:mock-viseme-corrector"
    # Backward compatible field contains final effective text
    assert data["transcription"] == "lay green in zero soon"


def test_api_predict_with_correction_disabled(api_test_client: TestClient) -> None:
    """POST /predict?use_correction=false disables contextual correction."""
    fake_video = b"dummy_mp4_bytes_for_testing"
    resp = api_test_client.post(
        "/predict",
        params={"method": "greedy", "use_correction": False},
        files={"file": ("sample.mp4", io.BytesIO(fake_video), "video/mp4")},
    )

    assert resp.status_code == 200
    data = resp.json()

    assert data["raw_transcription"] == "lay freen in zero soon"
    assert data["corrected_transcription"] == "lay freen in zero soon"
    assert data["correction_applied"] is False
    assert data["transcription"] == "lay freen in zero soon"


# ---------------------------------------------------------------------------
# 11. API Client Response Deserialization
# ---------------------------------------------------------------------------


def test_api_client_parses_correction_fields() -> None:
    """LipReadAPIClient correctly maps API JSON response into PredictionResult dataclass."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "transcription": "lay green in zero soon",
        "method": "greedy",
        "confidence": -0.25,
        "processing_time_seconds": 0.45,
        "num_frames": 75,
        "raw_transcription": "lay freen in zero soon",
        "corrected_transcription": "lay green in zero soon",
        "correction_applied": True,
        "correction_model": "mock-viseme-corrector",
    }
    mock_session.post.return_value = mock_response

    client = LipReadAPIClient(base_url="http://testserver:8000", session=mock_session)
    result = client.predict(
        video_bytes=b"dummy",
        filename="test.mp4",
        use_correction=True,
    )

    assert result.transcription == "lay green in zero soon"
    assert result.raw_transcription == "lay freen in zero soon"
    assert result.corrected_transcription == "lay green in zero soon"
    assert result.correction_applied is True
    assert result.correction_model == "mock-viseme-corrector"

    # Verify use_correction query parameter was passed
    _, kwargs = mock_session.post.call_args
    assert kwargs["params"]["use_correction"] is True
