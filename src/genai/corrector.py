"""Contextual corrector implementation for LipRead.

Provides the primary ContextualCorrector abstraction that takes raw CTC
transcriptions, runs them through the LangChain prompt-model pipeline, and returns
a typed CorrectionResult.

Includes guaranteed fallback behavior: if GenAI is disabled, credentials are
missing, or an inference error occurs, the raw transcription is safely preserved
without interrupting the main lip-reading pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.genai.config import GenAIConfig, get_genai_config
from src.genai.prompts import get_correction_prompt
from src.genai.providers import create_chat_model
from src.genai.schemas import CorrectionResult

logger = logging.getLogger("lipread.genai.corrector")


class ContextualCorrector:
    """Orchestrates LangChain contextual post-processing on raw lip-reading transcripts."""

    def __init__(
        self,
        config: Optional[GenAIConfig] = None,
        model: Optional[BaseChatModel] = None,
    ) -> None:
        self.config = config or get_genai_config()
        self._model = model
        self._chain = None

        if self.config.enabled:
            self._initialize_pipeline()
        else:
            logger.info("ContextualCorrector initialized in DISABLED mode (GENAI_ENABLED=false).")

    def _initialize_pipeline(self) -> None:
        """Initializes the underlying LangChain prompt | model | parser pipeline."""
        try:
            if self._model is None:
                self._model = create_chat_model(self.config)

            prompt = get_correction_prompt()
            parser = StrOutputParser()
            self._chain = prompt | self._model | parser
            logger.info(
                "ContextualCorrector pipeline initialized | provider=%s | model=%s",
                self.config.provider,
                self.config.model,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize LLM provider '%s': %s. Contextual correction will fall back to raw output.",
                self.config.provider,
                exc,
            )
            self._chain = None

    @property
    def model(self) -> Optional[BaseChatModel]:
        """Returns the underlying LangChain chat model."""
        return self._model

    @property
    def is_active(self) -> bool:
        """Returns True if GenAI is enabled and the chain is ready to execute."""
        return self.config.enabled and self._chain is not None

    def correct(self, raw_transcript: str) -> CorrectionResult:
        """Corrects phonetic/grammatical errors in a raw lip-reading transcript.

        Args:
            raw_transcript: The unedited text produced by the CTC decoder.

        Returns:
            CorrectionResult containing raw and corrected text, status flag, and timing.
        """
        # 1. Guard against empty inputs or disabled state
        if not raw_transcript or not raw_transcript.strip():
            return CorrectionResult(
                raw_transcription=raw_transcript,
                corrected_transcription=raw_transcript,
                correction_applied=False,
                provider=self.config.provider if self.config.enabled else None,
                model=self.config.model if self.config.enabled else None,
                latency_seconds=0.0,
            )

        if not self.config.enabled:
            return CorrectionResult(
                raw_transcription=raw_transcript,
                corrected_transcription=raw_transcript,
                correction_applied=False,
                provider=None,
                model=None,
                latency_seconds=0.0,
            )

        # Lazy initialize if needed
        if self._chain is None:
            self._initialize_pipeline()
            if self._chain is None:
                return CorrectionResult(
                    raw_transcription=raw_transcript,
                    corrected_transcription=raw_transcript,
                    correction_applied=False,
                    provider=self.config.provider,
                    model=self.config.model,
                    latency_seconds=0.0,
                    error="LLM chain failed to initialize; returned raw transcription.",
                )

        # 2. Invoke LangChain pipeline with timing and error isolation
        t0 = time.perf_counter()
        try:
            raw_output = self._chain.invoke(
                {"raw_transcript": raw_transcript},
                {"timeout": self.config.timeout},
            )
            latency = time.perf_counter() - t0

            # 3. Post-process LLM output (strip quotes, extra spaces, linebreaks)
            cleaned = str(raw_output).strip()
            if (cleaned.startswith('"') and cleaned.endswith('"')) or (
                cleaned.startswith("'") and cleaned.endswith("'")
            ):
                cleaned = cleaned[1:-1].strip()

            # Remove potential markdown code blocks if emitted
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if len(lines) >= 3 and lines[-1].startswith("```"):
                    cleaned = "\n".join(lines[1:-1]).strip()

            # Determine whether substantive correction occurred
            is_changed = bool(cleaned) and (cleaned.lower() != raw_transcript.strip().lower())
            final_text = cleaned if is_changed else raw_transcript

            logger.info(
                "Contextual correction executed in %.3fs | applied=%s | raw='%s' -> corrected='%s'",
                latency,
                is_changed,
                raw_transcript,
                final_text,
            )

            return CorrectionResult(
                raw_transcription=raw_transcript,
                corrected_transcription=final_text,
                correction_applied=is_changed,
                provider=self.config.provider,
                model=self.config.model,
                latency_seconds=round(latency, 4),
            )

        except Exception as exc:
            latency = time.perf_counter() - t0
            logger.warning(
                "LLM contextual correction failed: %s. Preserving raw transcription.",
                exc,
            )
            # NEVER crash the lip-reading pipeline due to an LLM error
            return CorrectionResult(
                raw_transcription=raw_transcript,
                corrected_transcription=raw_transcript,
                correction_applied=False,
                provider=self.config.provider,
                model=self.config.model,
                latency_seconds=round(latency, 4),
                error=str(exc),
            )
