"""Inference engine for LipRead.

Loads a trained checkpoint and performs end-to-end lip-reading inference:

  Video file / pre-processed tensor
       ↓  (LipPreprocessingPipeline)
  [T, C, H, W] tensor
       ↓  (LipReadModel)
  logits [T, V]
       ↓  (CTC decoder)
  transcription string

Usage:
    from src.inference.engine import InferenceEngine

    engine = InferenceEngine.from_checkpoint("E:/LipRead_Data/checkpoints/best_model.pt")
    result = engine.transcribe("path/to/video.mp4")
    print(result.text)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch

from src.data.dataset_config import VocabularyConfig
from src.inference.decoder import ctc_decode
from src.models.lipread_model import LipReadModel
from src.preprocessing.pipeline import LipPreprocessingPipeline
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.inference.engine")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Container for a single inference result."""

    text: str
    """Decoded transcription string."""

    confidence: float
    """Mean max log-probability across frames (proxy for confidence, ∈ (-∞, 0])."""

    elapsed_ms: float
    """Wall-clock time for the full pipeline in milliseconds."""

    num_frames: int
    """Number of video frames actually processed."""

    decode_method: str = "greedy"
    """Decoding algorithm used (``'greedy'`` or ``'beam'``)."""

    error: Optional[str] = None
    """Error message if inference failed, else None."""

    @property
    def success(self) -> bool:
        return self.error is None

    def __str__(self) -> str:
        if not self.success:
            return f"InferenceResult[FAILED]: {self.error}"
        return (
            f"InferenceResult:\n"
            f"  Text:          '{self.text}'\n"
            f"  Confidence:    {self.confidence:.4f}\n"
            f"  Frames:        {self.num_frames}\n"
            f"  Decode:        {self.decode_method}\n"
            f"  Elapsed:       {self.elapsed_ms:.1f} ms"
        )


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """End-to-end lip-reading inference engine.

    Wraps a trained ``LipReadModel`` with preprocessing and CTC decoding
    for single-video inference.

    Attributes:
        model: The loaded ``LipReadModel`` in eval mode.
        device: Torch device used for inference.
        vocab: Vocabulary configuration used for decoding.
        decode_method: ``'greedy'`` or ``'beam'``.
        beam_width: Beam width (only used when ``decode_method='beam'``).
    """

    def __init__(
        self,
        model: LipReadModel,
        vocab: Optional[VocabularyConfig] = None,
        device: Optional[Union[torch.device, str]] = None,
        decode_method: str = "greedy",
        beam_width: int = 10,
    ) -> None:
        self.vocab = vocab or VocabularyConfig()
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()
        self.decode_method = decode_method
        self.beam_width = beam_width

        logger.info(
            "InferenceEngine initialized | device=%s | method=%s | vocab_size=%d",
            self.device,
            self.decode_method,
            self.vocab.vocab_size,
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        decode_method: str = "greedy",
        beam_width: int = 10,
        device: Optional[Union[torch.device, str]] = None,
    ) -> InferenceEngine:
        """Loads model weights from a checkpoint and returns an engine.

        Args:
            checkpoint_path: Path to ``.pt`` checkpoint file saved by the
                             ``Trainer`` (``best_model.pt`` or ``latest_model.pt``).
            decode_method: ``'greedy'`` (default) or ``'beam'``.
            beam_width: Beam width for beam search decoding.
            device: Target device.  Defaults to CUDA if available.

        Returns:
            Initialised ``InferenceEngine`` ready for inference.

        Raises:
            FileNotFoundError: If checkpoint file is not found.
            RuntimeError: If checkpoint cannot be loaded.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        _device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("Loading checkpoint from %s  →  %s", checkpoint_path, _device)

        checkpoint = torch.load(str(checkpoint_path), map_location=_device, weights_only=False)

        # Build model from config embedded in checkpoint, or use defaults
        model_kwargs = checkpoint.get("model_config", {})
        vocab = VocabularyConfig()
        model = LipReadModel(vocab_size=vocab.vocab_size, **model_kwargs)

        # Load state dict
        state_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
        if state_key in checkpoint:
            model.load_state_dict(checkpoint[state_key])
            logger.info("Loaded model weights from checkpoint (key='%s')", state_key)
        else:
            # Checkpoint is the state dict directly
            model.load_state_dict(checkpoint)
            logger.info("Loaded model weights (bare state dict)")

        epoch = checkpoint.get("epoch", "?")
        val_loss = checkpoint.get("val_loss", float("nan"))
        logger.info("Checkpoint epoch=%s | val_loss=%.4f", epoch, val_loss)

        return cls(
            model=model,
            vocab=vocab,
            device=_device,
            decode_method=decode_method,
            beam_width=beam_width,
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def transcribe_tensor(
        self,
        frames: torch.Tensor,
        decode_method: Optional[str] = None,
        beam_width: Optional[int] = None,
    ) -> InferenceResult:
        """Transcribes from a pre-processed lip-ROI tensor.

        Args:
            frames: Preprocessed tensor of shape [T, C, H, W].
            decode_method: Override instance ``decode_method``.
            beam_width: Override instance ``beam_width``.

        Returns:
            :class:`InferenceResult`.
        """
        t0 = time.perf_counter()
        method = decode_method or self.decode_method
        width = beam_width or self.beam_width

        try:
            # Add batch dimension: [T, C, H, W] → [1, T, C, H, W]
            if frames.ndim == 4:
                frames = frames.unsqueeze(0)
            frames = frames.to(self.device)

            num_frames = frames.shape[1]

            # Forward pass → [1, T, V]
            logits = self.model(frames)
            logits_seq = logits.squeeze(0)  # [T, V]

            # Confidence proxy: mean of max log-probs across frames
            import torch.nn.functional as F
            log_probs = F.log_softmax(logits_seq.float(), dim=-1)
            confidence = float(log_probs.max(dim=-1).values.mean().item())

            # CTC decode
            text = ctc_decode(
                logits=logits_seq,
                vocab=self.vocab,
                blank_idx=self.vocab.blank_idx,
                method=method,
                beam_width=width,
                apply_log_softmax=True,
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return InferenceResult(
                text=text,
                confidence=confidence,
                elapsed_ms=elapsed_ms,
                num_frames=num_frames,
                decode_method=method,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception("Inference failed on tensor input")
            return InferenceResult(
                text="",
                confidence=float("-inf"),
                elapsed_ms=elapsed_ms,
                num_frames=0,
                decode_method=method,
                error=str(exc),
            )

    def transcribe(
        self,
        video_path: Union[str, Path],
        decode_method: Optional[str] = None,
        beam_width: Optional[int] = None,
    ) -> InferenceResult:
        """Full end-to-end transcription from a raw video file.

        Args:
            video_path: Path to any video file readable by OpenCV.
            decode_method: Override instance decode method.
            beam_width: Override instance beam width.

        Returns:
            :class:`InferenceResult`.
        """
        t0 = time.perf_counter()
        method = decode_method or self.decode_method
        width = beam_width or self.beam_width
        video_path = Path(video_path)

        if not video_path.exists():
            return InferenceResult(
                text="",
                confidence=float("-inf"),
                elapsed_ms=0.0,
                num_frames=0,
                decode_method=method,
                error=f"Video file not found: {video_path}",
            )

        try:
            logger.info("Preprocessing video: %s", video_path)
            with LipPreprocessingPipeline() as pipeline:
                prep_result = pipeline.process_video(video_path=video_path)

            if not prep_result.success or prep_result.tensor is None:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return InferenceResult(
                    text="",
                    confidence=float("-inf"),
                    elapsed_ms=elapsed_ms,
                    num_frames=0,
                    decode_method=method,
                    error=f"Preprocessing failed: {prep_result.error_message}",
                )

            # tensor is [T, C, H, W]
            result = self.transcribe_tensor(
                frames=prep_result.tensor,
                decode_method=method,
                beam_width=width,
            )
            # Override elapsed to include preprocessing time
            total_elapsed = (time.perf_counter() - t0) * 1000.0
            return InferenceResult(
                text=result.text,
                confidence=result.confidence,
                elapsed_ms=total_elapsed,
                num_frames=result.num_frames,
                decode_method=result.decode_method,
                error=result.error,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception("End-to-end inference failed for video: %s", video_path)
            return InferenceResult(
                text="",
                confidence=float("-inf"),
                elapsed_ms=elapsed_ms,
                num_frames=0,
                decode_method=method,
                error=str(exc),
            )
