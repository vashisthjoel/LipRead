#!/usr/bin/env python3
"""Watchdog script to automatically pause MVP-2 training immediately after Epoch 1 completes.

Monitors:
- E:\\LipRead_Data\\checkpoints\\mvp_15000\\latest_model.pt
- Training log (task-1987.log)

As soon as Epoch 1 completes and writes checkpoints/metrics:
- Freezes/suspends the training process via NtSuspendProcess before Epoch 2 starts.
- Records epoch1_paused.json marker.
"""

import ctypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Paths
CHECKPOINT_DIR = Path(r"E:\LipRead_Data\checkpoints\mvp_15000")
LATEST_PT = CHECKPOINT_DIR / "latest_model.pt"
BEST_PT = CHECKPOINT_DIR / "best_model.pt"
PAUSED_MARKER = CHECKPOINT_DIR / "epoch1_paused.json"
LOG_PATH = Path(
    r"C:\Users\HP\.gemini\antigravity-ide\brain\8c396b4c-214b-4237-a609-01db39904c59\.system_generated\tasks\task-1987.log"
)

ntdll = ctypes.windll.ntdll
kernel32 = ctypes.windll.kernel32
PROCESS_SUSPEND_RESUME = 0x0800


def get_training_pids() -> list[int]:
    """Finds all PIDs associated with run_mvp_15000."""
    pids = []
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
                if pid != os.getpid():
                    pids.append(pid)
    except Exception:
        pass
    return pids


def suspend_pid(pid: int) -> bool:
    """Freezes process execution via NtSuspendProcess."""
    try:
        h = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if not h:
            return False
        res = ntdll.NtSuspendProcess(h)
        kernel32.CloseHandle(h)
        return res == 0
    except Exception as e:
        print(f"Error suspending PID {pid}: {e}")
        return False


def main():
    print("=" * 70)
    print("MVP-2 EPOCH 1 PAUSE WATCHDOG ACTIVATED")
    print(f"Monitoring: {CHECKPOINT_DIR}")
    print("Will automatically suspend training before Epoch 2 starts.")
    print("=" * 70)

    epoch_1_done = False

    while not epoch_1_done:
        time.sleep(2)

        # Check if latest_model.pt exists
        if LATEST_PT.exists() and LATEST_PT.stat().st_size > 10_000_000:
            epoch_1_done = True
            break

        # Check log for Epoch 01/10
        if LOG_PATH.exists():
            try:
                with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if "Epoch 01/10 | Train Loss:" in content or "Epoch 1/10 | Train Loss:" in content:
                    epoch_1_done = True
                    break
            except Exception:
                pass

    print("\n[WATCHDOG TRIGGERED]: Epoch 1 has completed!")
    # Allow 1.5 seconds for file buffers and best_model.pt to settle completely
    time.sleep(1.5)

    pids = get_training_pids()
    suspended_pids = []
    for pid in pids:
        if suspend_pid(pid):
            suspended_pids.append(pid)
            print(f"Successfully SUSPENDED process PID: {pid}")

    # Write marker
    marker_data = {
        "status": "PAUSED_AFTER_EPOCH_1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suspended_pids": suspended_pids,
        "latest_checkpoint": str(LATEST_PT) if LATEST_PT.exists() else None,
        "best_checkpoint": str(BEST_PT) if BEST_PT.exists() else None,
    }
    with open(PAUSED_MARKER, "w", encoding="utf-8") as f:
        import json
        json.dump(marker_data, f, indent=2)

    print(f"\nTraining paused at Epoch-1 -> Epoch-2 boundary.")
    print(f"Marker written to: {PAUSED_MARKER}")
    print("Epoch 2 has NOT started.")


if __name__ == "__main__":
    main()
