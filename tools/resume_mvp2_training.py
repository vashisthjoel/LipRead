#!/usr/bin/env python3
"""Resumes paused MVP-2 training process via NtResumeProcess."""

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

PAUSED_MARKER = Path(r"E:\LipRead_Data\checkpoints\mvp_15000\epoch1_paused.json")

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


def resume_pid(pid: int) -> bool:
    """Unfreezes process execution via NtResumeProcess."""
    try:
        h = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if not h:
            return False
        res = ntdll.NtResumeProcess(h)
        kernel32.CloseHandle(h)
        return res == 0
    except Exception as e:
        print(f"Error resuming PID {pid}: {e}")
        return False


def main():
    print("=" * 70)
    print("RESUMING MVP-2 TRAINING PROCESS")
    print("=" * 70)

    pids = get_training_pids()
    if not pids and PAUSED_MARKER.exists():
        try:
            with open(PAUSED_MARKER, "r", encoding="utf-8") as f:
                data = json.load(f)
                pids = data.get("suspended_pids", [])
        except Exception:
            pass

    if not pids:
        print("No suspended MVP-2 training process found.")
        return

    for pid in pids:
        if resume_pid(pid):
            print(f"Successfully RESUMED process PID: {pid}")

    if PAUSED_MARKER.exists():
        try:
            PAUSED_MARKER.unlink()
            print("Removed pause marker file.")
        except Exception:
            pass

    print("\nMVP-2 training resumed successfully!")


if __name__ == "__main__":
    main()
