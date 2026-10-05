"""Data schemas and result structures for GenAI contextual post-processing.

Defines typed containers representing raw and LLM-corrected transcriptions
along with operational metadata and execution status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CorrectionResult:
    """Container holding raw and contextually corrected transcription data."""

    raw_transcription: str
    """The original unedited transcription produced by the CTC decoder."""

    corrected_transcription: str
    """The contextually corrected transcription produced by the LLM."""

    correction_applied: bool
    """True if the LLM modified the text; False if unchanged, disabled, or failed."""

    provider: Optional[str] = None
    """LLM provider name (e.g. 'openai', 'anthropic', 'ollama', 'mock')."""

    model: Optional[str] = None
    """Model identifier used for contextual correction."""

    latency_seconds: Optional[float] = None
    """Wall-clock time spent in the LLM post-processing step."""

    error: Optional[str] = None
    """Error message if correction failed and fell back to raw text, else None."""

    @property
    def effective_text(self) -> str:
        """Returns the final text string (corrected if applied, otherwise raw)."""
        return self.corrected_transcription if self.correction_applied else self.raw_transcription

    def to_dict(self) -> dict:
        """Serializes the result to a standard dictionary."""
        return {
            "raw_transcription": self.raw_transcription,
            "corrected_transcription": self.corrected_transcription,
            "correction_applied": self.correction_applied,
            "provider": self.provider,
            "model": self.model,
            "latency_seconds": self.latency_seconds,
            "error": self.error,
        }
