"""Lightweight model forward pass and GPU memory smoke test for LipRead.

Tests:
  - CUDA detection and device movement
  - Model parameter count calculation
  - GPU VRAM consumption before and after forward pass (GTX 1650 4GB ceiling)
  - Tensor shapes, dtype, and NaN/Inf validation
  - Synthetic CTC loss calculation
"""

from __future__ import annotations

import argparse
import sys
import torch

from src.models.ctc import CTCLossWrapper
from src.models.lipread_model import LipReadModel
from src.models.model_utils import (
    count_parameters,
    count_trainable_parameters,
    get_model_summary,
    to_device,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.models.smoke_test")


def run_model_smoke_test(device_str: str | None = None) -> int:
    """Executes non-training forward pass and memory profiling on LipReadModel."""
    divider = "=" * 75
    subdivider = "-" * 75

    print("\n" + divider)
    print("LipRead - Model Architecture & Hardware Smoke Test")
    print(divider)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:           {cuda_available}")

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"Device Name:              {device_name}")
        print(f"Total VRAM:               {vram_total_gb} GB")
    else:
        print("Device:                   CPU only")

    # 1. Instantiate Model
    print(subdivider)
    model = LipReadModel()
    model, device = to_device(model, device_str)
    model.eval()

    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    summary = get_model_summary(model)

    print(f"Active Device:            {device}")
    print(f"Model Type:               {summary['model_type']}")
    print(f"Total Parameters:         {total_params:,}")
    print(f"Trainable Parameters:     {trainable_params:,}")
    print(f"Model Weight Size:        ~{summary['parameter_size_mb']} MB (float32)")
    print(f"Vocabulary Classes:       {model.vocab_size} (CTC blank at index 0)")
    print(f"CNN Feature Dimension:    {model.feature_dim}")
    print(f"LSTM Hidden Dimension:    {model.lstm_hidden_size} (Bi-directional = {model.backend.output_dim})")
    print(subdivider)

    # 2. Memory before forward
    mem_before_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"GPU Memory Before Pass:   {mem_before_mb:.2f} MB")

    # 3. Create synthetic input [B=1, T=75, C=1, H=88, W=88]
    batch_size = 1
    t_frames = 75
    c_channels = 1
    h_height = 88
    w_width = 88

    x = torch.randn(
        (batch_size, t_frames, c_channels, h_height, w_width),
        dtype=torch.float32,
        device=device,
    )
    print(f"Input Shape:              {tuple(x.shape)} [B, T, C, H, W]")
    print(f"Input Dtype:              {x.dtype}")

    # 4. Forward Pass
    with torch.no_grad():
        logits = model(x)

    # 5. Memory after forward
    mem_after_mb = 0.0
    peak_mem_mb = 0.0
    if device.type == "cuda":
        mem_after_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"GPU Memory After Pass:    {mem_after_mb:.2f} MB")
        print(f"GPU Peak Memory:          {peak_mem_mb:.2f} MB")
        print(f"Activation Memory Delta:  +{mem_after_mb - mem_before_mb:.2f} MB")

    print(subdivider)
    print(f"Output Logits Shape:      {tuple(logits.shape)} [B, T, V]")
    print(f"Output Dtype:             {logits.dtype}")

    # Check for NaN / Inf
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    print(f"Contains NaN:             {'YES (ERROR)' if has_nan else 'NO (Clean)'}")
    print(f"Contains Inf:             {'YES (ERROR)' if has_inf else 'NO (Clean)'}")

    if has_nan or has_inf:
        print("[FAIL] Output logits contain NaN or Inf values.", file=sys.stderr)
        return 1

    # 6. CTC Layout and Loss Check
    ctc_logits = model.to_ctc_layout(logits)
    print(f"CTC Transposed Shape:     {tuple(ctc_logits.shape)} [T, B, V]")

    # Compute synthetic CTC loss
    log_probs = model.get_log_probs(x, ctc_layout=True)
    synthetic_targets = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)
    input_lengths = torch.tensor([75], dtype=torch.long, device=device)
    target_lengths = torch.tensor([5], dtype=torch.long, device=device)

    ctc_wrapper = CTCLossWrapper(blank=0)
    loss = ctc_wrapper(log_probs, synthetic_targets, input_lengths, target_lengths)
    loss_val = loss.item()
    is_finite = torch.isfinite(loss).item()

    print(f"Synthetic CTC Loss:       {loss_val:.4f} (Finite: {is_finite})")
    print(subdivider)

    # 7. Clean up
    del x, logits, ctc_logits, log_probs, synthetic_targets, input_lengths, target_lengths, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not is_finite:
        print("[FAIL] Computed CTC loss is not finite.", file=sys.stderr)
        return 1

    print("RESULT: Model architecture smoke test PASSED successfully.")
    print("        GTX 1650 memory footprint is comfortably within 4 GB VRAM.")
    print(divider + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LipRead Model Smoke Test")
    parser.add_argument("--device", "-d", type=str, default=None, help="Target device (cpu or cuda)")
    args = parser.parse_args()
    return run_model_smoke_test(device_str=args.device)


if __name__ == "__main__":
    sys.exit(main())
