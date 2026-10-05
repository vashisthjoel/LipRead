#!/usr/bin/env python3
"""LipRead — Live Read-Only Training Monitor for MVP-2 Experiment (15,000 Samples, 10 Epochs).

Strictly READ-ONLY. Monitors:
- Background training process (PID 109172 / task-1987 running run_mvp_15000)
- Training log (task-1987.log)
- NVIDIA GPU metrics via nvidia-smi
- Tensor cache growth in E:\\LipRead_Data\\processed\\egcllc
- Checkpoint directory timestamps and sizes (E:\\LipRead_Data\\checkpoints\\mvp_15000)
- MLflow metrics and status
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Paths
DEFAULT_LOG_PATH = Path(
    r"C:\Users\HP\.gemini\antigravity-ide\brain\8c396b4c-214b-4237-a609-01db39904c59\.system_generated\tasks\task-1987.log"
)
CHECKPOINT_DIR = Path(r"E:\LipRead_Data\checkpoints\mvp_15000")
CACHE_DIR = Path(r"E:\LipRead_Data\processed\egcllc")

TOTAL_EPOCHS = 10
TOTAL_TRAIN_SAMPLES = 15000
TOTAL_VAL_SAMPLES = 2000
BATCH_SIZE = 4
TOTAL_BATCHES_PER_EPOCH = TOTAL_TRAIN_SAMPLES // BATCH_SIZE  # 3750
VAL_BATCHES = TOTAL_VAL_SAMPLES // BATCH_SIZE  # 500


def is_pid_alive(pid: int) -> bool:
    """Checks whether a process ID is currently alive on Windows."""
    if pid <= 0:
        return False
    try:
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return exit_code.value == 259  # STILL_ACTIVE = 259
    except Exception:
        return False


def find_training_pid() -> Optional[int]:
    """Finds active python process running run_mvp_15000."""
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"CommandLine like '%run_mvp_15000%'\" | "
            "Select-Object -ExpandProperty ProcessId",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                pid = int(line)
                if pid != os.getpid() and is_pid_alive(pid):
                    return pid
    except Exception:
        pass
    return None


def get_gpu_metrics() -> Dict[str, Any]:
    """Queries GPU metrics using nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=4).strip()
        if out:
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            return {
                "gpu_util": float(parts[0]),
                "mem_used_mb": float(parts[1]),
                "mem_total_mb": float(parts[2]),
                "temp_c": float(parts[3]),
                "power_w": float(parts[4]) if len(parts) > 4 else 0.0,
            }
    except Exception:
        pass
    return {}


def get_cache_count() -> int:
    """Gets total preprocessed .pt files in cache directory."""
    if not CACHE_DIR.exists():
        return 0
    try:
        return len([f for f in os.listdir(CACHE_DIR) if f.endswith(".pt")])
    except Exception:
        return 0


def parse_training_log(log_path: Path) -> Dict[str, Any]:
    """Parses task-1987.log for training events."""
    if not log_path.exists():
        return {"exists": False, "epochs": [], "last_lines": []}

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return {"exists": False, "epochs": [], "last_lines": []}

    epochs = []
    # Pattern: Epoch 01/10 | Train Loss: 2.1234 | Val Loss: 1.8432 | CER: 0.7123 | WER: 0.9421 | LR: 1.00e-04 | Time: 1234.5s | Peak VRAM: 1411.0MB
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+Train Loss:\s+([\d\.]+)\s+\|\s+Val Loss:\s+([\d\.]+)\s+\|\s+CER:\s+([\d\.]+)\s+\|\s+WER:\s+([\d\.]+)\s+\|\s+LR:\s+([\d\.eE\-\+]+)\s+\|\s+Time:\s+([\d\.]+)s"
    )

    for line in lines:
        m = pattern.search(line)
        if m:
            epochs.append({
                "epoch": int(m.group(1)),
                "total_epochs": int(m.group(2)),
                "train_loss": float(m.group(3)),
                "val_loss": float(m.group(4)),
                "cer": float(m.group(5)),
                "wer": float(m.group(6)),
                "lr": float(m.group(7)),
                "time_sec": float(m.group(8)),
            })

    tail_lines = [l.strip() for l in lines[-10:] if l.strip()]

    return {
        "exists": True,
        "epochs": epochs,
        "last_lines": tail_lines,
        "total_lines": len(lines),
    }


def render_display(log_path: Path, current_pid: Optional[int]) -> None:
    """Renders the single-frame read-only dashboard."""
    pid = current_pid or find_training_pid()
    alive = is_pid_alive(pid) if pid else False
    gpu = get_gpu_metrics()
    cache_cnt = get_cache_count()
    log_info = parse_training_log(log_path)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
    print("=" * 78)
    print(f"   LIPREAD — MVP-2 LIVE READ-ONLY TRAINING MONITOR")
    print(f"   Timestamp: {now_str} | Refresh: 10s")
    print("=" * 78)

    # Process Status
    status_str = f"RUNNING (PID: {pid})" if alive else f"STOPPED / NOT RUNNING"
    print(f"\n[PROCESS STATUS]: {status_str}")
    print(f"  Target Run:      MVP-2 (15,000 samples, 10 epochs)")
    print(f"  Log File:        {log_path.name} ({log_info['total_lines']} lines)")

    # GPU Status
    if gpu:
        print(f"\n[NVIDIA GPU — GTX 1650]:")
        print(f"  Compute Util:    {gpu['gpu_util']:.0f}%")
        print(f"  VRAM Allocated:  {gpu['mem_used_mb']:.0f} MiB / {gpu['mem_total_mb']:.0f} MiB")
        print(f"  Temperature:     {gpu['temp_c']:.0f}°C | Power: {gpu['power_w']:.1f}W")

    # Dataset & Cache Status
    print(f"\n[DATASET & PREPROCESSED TENSOR CACHE]:")
    print(f"  Cached Tensors:  {cache_cnt:,} .pt files in E:\\LipRead_Data\\processed\\egcllc")
    print(f"  Train Manifest:  15,000 samples (16 speakers)")
    print(f"  Val Manifest:    2,000 samples (s19, s20)")
    print(f"  Test Manifest:   2,000 samples (s14, s23 - held out)")

    # Checkpoints
    print(f"\n[CHECKPOINTS — E:\\LipRead_Data\\checkpoints\\mvp_15000]:")
    best_pt = CHECKPOINT_DIR / "best_model.pt"
    latest_pt = CHECKPOINT_DIR / "latest_model.pt"
    if best_pt.exists():
        best_size = best_pt.stat().st_size / (1024**2)
        best_time = datetime.fromtimestamp(best_pt.stat().st_mtime).strftime("%H:%M:%S")
        print(f"  best_model.pt:   {best_size:.1f} MB (updated {best_time})")
    else:
        print(f"  best_model.pt:   None yet (written after epoch validation)")
    if latest_pt.exists():
        lat_size = latest_pt.stat().st_size / (1024**2)
        lat_time = datetime.fromtimestamp(latest_pt.stat().st_mtime).strftime("%H:%M:%S")
        print(f"  latest_model.pt: {lat_size:.1f} MB (updated {lat_time})")

    # Epoch Progress Table
    epochs = log_info.get("epochs", [])
    print(f"\n[COMPLETED EPOCHS: {len(epochs)}/10]:")
    if epochs:
        print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>9} | {'Val CER':>8} | {'Val WER':>8} | {'LR':>9} | {'Duration':>9}")
        print("  " + "-" * 72)
        for ep in epochs:
            print(
                f"  {ep['epoch']:>5d} | {ep['train_loss']:>10.4f} | {ep['val_loss']:>9.4f} | "
                f"{ep['cer']*100:>7.2f}% | {ep['wer']*100:>7.2f}% | {ep['lr']:>9.2e} | {ep['time_sec']/60:>7.1f}m"
            )
    else:
        print("  Epoch 1 is actively running...")

    # Log Tail
    print(f"\n[RECENT LOG OUTPUT]:")
    for l in log_info["last_lines"][-4:]:
        print(f"  > {l}")

    print("\n" + "=" * 78)
    print("Press Ctrl+C to exit monitor (training continues in background).")


def main():
    parser = argparse.ArgumentParser(description="Live monitor for MVP-2 training.")
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG_PATH), help="Path to training log")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval seconds")
    parser.add_argument("--once", action="store_true", help="Print single status snapshot and exit")
    args = parser.parse_args()

    log_path = Path(args.log)
    pid = find_training_pid()

    if args.once:
        render_display(log_path, pid)
        return

    try:
        while True:
            render_display(log_path, pid)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")


if __name__ == "__main__":
    main()
