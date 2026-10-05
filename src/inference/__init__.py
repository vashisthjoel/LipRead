"""LipRead inference sub-package.

Provides:
  - ``decoder``: CTC greedy and beam-search decoding utilities.
  - ``engine``: ``InferenceEngine`` — loads checkpoint, runs end-to-end transcription.
  - ``cli``: Command-line entry point (``python -m src.inference.cli``).
"""

from src.inference.decoder import ctc_decode, ctc_decode_batch, greedy_decode, beam_search_decode
from src.inference.engine import InferenceEngine, InferenceResult

__all__ = [
    "ctc_decode",
    "ctc_decode_batch",
    "greedy_decode",
    "beam_search_decode",
    "InferenceEngine",
    "InferenceResult",
]
