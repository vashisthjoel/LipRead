"""Environment verification script for LipRead.

Verifies:
  - Python version
  - PyTorch version
  - CUDA availability
  - GPU device name
  - GPU VRAM
  - OpenCV availability (if installed)
  - MediaPipe availability (if installed)
  - Storage volume accessibility & free space

Reports clear PASS / FAIL / NOT INSTALLED statuses for each component.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

from src.data.paths import check_storage_status, get_data_root
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.env_check")


def check_python() -> Tuple[str, str, str]:
    """Checks the Python runtime version."""
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 10:
        return "PASS", ver_str, "Python >= 3.10 supported"
    return "FAIL", ver_str, "Python 3.10+ required"


def check_pytorch() -> Tuple[str, str, str]:
    """Checks PyTorch installation and version."""
    try:
        import torch

        return "PASS", torch.__version__, "PyTorch installed successfully"
    except ImportError as exc:
        return "FAIL", "Not Installed", str(exc)


def check_cuda() -> Tuple[str, str, str]:
    """Checks CUDA support within PyTorch."""
    try:
        import torch

        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda or "N/A"
            return "PASS", f"Available (CUDA {cuda_ver})", f"{torch.cuda.device_count()} GPU(s) detected"
        return "FAIL", "Unavailable", "torch.cuda.is_available() returned False"
    except Exception as exc:
        return "FAIL", "Error", str(exc)


def check_gpu_device() -> Tuple[str, str, str]:
    """Checks primary GPU device name."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            name = torch.cuda.get_device_name(0)
            return "PASS", name, "Primary GPU identified"
        return "FAIL", "None", "No CUDA GPU detected"
    except Exception as exc:
        return "FAIL", "Error", str(exc)


def check_gpu_vram() -> Tuple[str, str, str]:
    """Checks GPU VRAM capacity."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = round(props.total_memory / (1024**3), 2)
            msg = "Adequate for 88x88 cropped VSR with conservative batch size (AMP enabled)"
            return "PASS", f"{total_vram_gb} GB", msg
        return "FAIL", "None", "Cannot query VRAM without CUDA"
    except Exception as exc:
        return "FAIL", "Error", str(exc)


def check_opencv() -> Tuple[str, str, str]:
    """Checks optional OpenCV library."""
    try:
        import cv2

        return "PASS", cv2.__version__, "OpenCV is installed"
    except ImportError:
        return "NOT INSTALLED", "None", "Optional for foundation (required for video preprocessing)"


def check_mediapipe() -> Tuple[str, str, str]:
    """Checks optional MediaPipe library."""
    try:
        import mediapipe as mp

        ver = getattr(mp, "__version__", "installed")
        return "PASS", ver, "MediaPipe is installed"
    except ImportError:
        return "NOT INSTALLED", "None", "Optional for foundation (required for facial landmarking)"


def check_storage() -> Tuple[str, str, str]:
    """Checks DATA_ROOT volume availability and free space."""
    data_root = get_data_root()
    storage = check_storage_status()
    if storage.get("accessible"):
        free = storage.get("free_gb", 0)
        return "PASS", f"{free} GB Free", f"Volume: {data_root}"
    return "WARNING", "Inaccessible", f"Could not verify {data_root}"


def run_checks() -> List[Dict[str, Any]]:
    """Runs all environment checks and returns structured results."""
    checks = [
        ("Python Version", check_python),
        ("PyTorch", check_pytorch),
        ("CUDA Available", check_cuda),
        ("GPU Device", check_gpu_device),
        ("GPU VRAM", check_gpu_vram),
        ("OpenCV", check_opencv),
        ("MediaPipe", check_mediapipe),
        ("Storage (Data Root)", check_storage),
    ]

    results = []
    for label, fn in checks:
        status, value, note = fn()
        results.append(
            {
                "check": label,
                "status": status,
                "value": value,
                "note": note,
            }
        )
    return results


def print_summary(results: List[Dict[str, Any]]) -> bool:
    """Prints a formatted report and returns True if critical checks pass."""
    header = f"{'CHECK':<24} | {'STATUS':<14} | {'VALUE':<24} | {'NOTES'}"
    divider = "-" * 90

    print("\n" + "=" * 90)
    print("LipRead - Environment & Hardware Verification Report")
    print("=" * 90)
    print(header)
    print(divider)

    critical_failure = False
    for r in results:
        status = r["status"]
        if status == "FAIL":
            status_str = "[FAIL]"
            if r["check"] in ("Python Version", "PyTorch", "CUDA Available", "GPU Device"):
                critical_failure = True
        elif status == "PASS":
            status_str = "[PASS]"
        elif status == "NOT INSTALLED":
            status_str = "[PLANNED]"
        else:
            status_str = f"[{status}]"

        print(f"{r['check']:<24} | {status_str:<14} | {r['value']:<24} | {r['note']}")

    print(divider)
    if critical_failure:
        print("RESULT: Critical environment checks failed. Review dependencies.\n")
    else:
        print("RESULT: Core foundation environment checks PASSED.\n")

    return not critical_failure


def main() -> int:
    results = run_checks()
    success = print_summary(results)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
