"""GenAI contextual post-processing package for LipRead.

Provides LangChain-orchestrated LLM post-processing to resolve visual speech
phonetic ambiguities (visemes) and grammatical errors on raw CTC transcriptions.
"""

from src.genai.config import GenAIConfig, get_genai_config
from src.genai.corrector import ContextualCorrector
from src.genai.schemas import CorrectionResult

# Global default corrector instance
_default_corrector: ContextualCorrector | None = None


def get_corrector() -> ContextualCorrector:
    """Returns a singleton instance of the default ContextualCorrector."""
    global _default_corrector
    if _default_corrector is None:
        _default_corrector = ContextualCorrector()
    return _default_corrector


__all__ = [
    "ContextualCorrector",
    "CorrectionResult",
    "GenAIConfig",
    "get_corrector",
    "get_genai_config",
]
