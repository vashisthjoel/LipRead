#!/usr/bin/env python3
"""LipRead — Live Read-Only Training Monitor for MVP Experiment.

This monitor is strictly READ-ONLY. It tracks:
- Background training process (task-1057 / PID 36768)
- Training log (task-1057.log)
- NVIDIA GPU metrics via nvidia-smi
- Checkpoint directory timestamps and sizes
- MLflow metrics and status

It NEVER modifies or writes to any checkpoint, metadata, or MLflow files.
"""

from __future__ import annotations

import argparse
import ctypes
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
# Configure stdout to safely handle UTF-8 symbols on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Paths
DEFAULT_LOG_PATH = Path(
    r"C:\Users\HP\.gemini\antigravity-ide\brain\8c396b4c-214b-4237-a609-01db39904c59\.system_generated\tasks\task-1057.log"
)
CHECKPOINT_DIR = Path(r"E:\LipRead_Data\checkpoints\mvp_6000")
MLRUNS_RUN_DIR = Path(
    r"E:\LipRead_Data\mlruns\342918310669993023\52f559c1719d4333b5584966d3926600"
)

TOTAL_EPOCHS = 6
TOTAL_TRAIN_SAMPLES = 6000
TOTAL_VAL_SAMPLES = 1000
BATCH_SIZE = 4
TOTAL_BATCHES_PER_EPOCH = TOTAL_TRAIN_SAMPLES // BATCH_SIZE  # 1500
VAL_BATCHES = TOTAL_VAL_SAMPLES // BATCH_SIZE  # 250


def is_pid_alive(pid: int) -> bool:
    """Checks whether a process ID is currently alive on Windows."""
    if pid <= 0:
        return False
    try:
        # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return exit_code.value == 259  # STILL_ACTIVE
    except Exception:
        return False


def find_training_pid() -> Optional[int]:
    """Finds the PID of the training script if running."""
    # Check known PID first
    known_pid = 36768
    if is_pid_alive(known_pid):
        return known_pid

    # Fallback to query
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"CommandLine like '%run_mvp_6000%'\" | "
            "Where-Object { $_.ProcessId -ne $PID } | Select-Object -ExpandProperty ProcessId",
        ]
        out = subprocess.check_output(cmd, text=True, timeout=3).strip()
        lines = [line.strip() for line in out.splitlines() if line.strip().isdigit()]
        if lines:
            return int(lines[0])
    except Exception:
        pass
    return None


def get_gpu_info() -> Dict[str, Any]:
    """Queries NVIDIA GPU status via nvidia-smi."""
    info: Dict[str, Any] = {
        "model": "NVIDIA GPU",
        "utilization": "N/A",
        "vram_used": "N/A",
        "vram_total": "N/A",
        "temp": "N/A",
        "active_compute": False,
    }
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 5:
                info["model"] = parts[0]
                info["utilization"] = f"{parts[1]}%"
                info["vram_used"] = f"{parts[2]} MiB"
                info["vram_total"] = f"{parts[3]} MiB"
                info["temp"] = f"{parts[4]}°C"

        # Check compute apps
        c_res = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if c_res.returncode == 0:
            pids = [p.strip() for p in c_res.stdout.strip().splitlines() if p.strip()]
            info["active_compute"] = len(pids) > 0
    except Exception:
        pass
    return info


def parse_log(log_path: Path) -> Dict[str, Any]:
    """Parses task-1057.log strictly read-only."""
    data: Dict[str, Any] = {
        "epochs_completed": [],
        "latest_activity": "Log file not accessible",
        "last_log_time": None,
        "is_complete": False,
        "error_detected": None,
        "last_saved_checkpoint": None,
    }

    if not log_path.exists():
        return data

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        data["latest_activity"] = f"Error reading log: {e}"
        return data

    if not lines:
        data["latest_activity"] = "Log file is empty"
        return data

    epoch_re = re.compile(
        r"(\d{2}:\d{2}:\d{2})\s+\|\s+INFO\s+\|\s+src\.training\.trainer\s+\|\s+"
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+Train Loss:\s+([\d\.]+)\s+\|\s+Val Loss:\s+([\d\.]+)\s+\|\s+"
        r"CER:\s+([\d\.]+)\s+\|\s+WER:\s+([\d\.]+)\s+\|\s+LR:\s+([^\s]+)\s+\|\s+Time:\s+([\d\.]+)s"
    )
    ckpt_re = re.compile(
        r"(\d{2}:\d{2}:\d{2})\s+\|\s+INFO\s+\|\s+src\.training\.checkpoint\s+\|\s+Saved\s+([a-z]+)\s+checkpoint:\s+(.*)"
    )

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        data["latest_activity"] = line_clean

        m_epoch = epoch_re.search(line_clean)
        if m_epoch:
            t_str, ep_str, tot_str, tr_loss, val_loss, cer, wer, lr, duration = m_epoch.groups()
            data["epochs_completed"].append({
                "epoch": int(ep_str),
                "total_epochs": int(tot_str),
                "train_loss": float(tr_loss),
                "val_loss": float(val_loss),
                "cer": float(cer),
                "wer": float(wer),
                "lr": lr,
                "duration_sec": float(duration),
                "timestamp": t_str,
            })
            data["last_log_time"] = t_str

        m_ckpt = ckpt_re.search(line_clean)
        if m_ckpt:
            t_str, ckpt_type, path_str = m_ckpt.groups()
            data["last_saved_checkpoint"] = f"{ckpt_type} ({t_str})"
            data["last_log_time"] = t_str

        if "MVP 6-EPOCH TEST RESULTS" in line_clean or "6-EPOCH TRAINING COMPLETED" in line_clean:
            data["is_complete"] = True

        if "Traceback (most recent call last)" in line_clean or "CUDA out of memory" in line_clean:
            data["error_detected"] = line_clean

    return data


def parse_mlflow_metrics(run_dir: Path) -> Dict[str, Any]:
    """Reads latest metric values directly from MLflow text files."""
    metrics: Dict[str, Any] = {
        "status": "UNKNOWN",
        "train_loss": None,
        "val_loss": None,
        "cer": None,
        "wer": None,
        "learning_rate": None,
        "best_val_loss": None,
        "epoch_time_sec": None,
        "steps_recorded": 0,
    }

    if not run_dir.exists():
        return metrics

    # Read meta.yaml
    meta_file = run_dir / "meta.yaml"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("status:"):
                        code = line.split(":", 1)[1].strip()
                        if code == "1":
                            metrics["status"] = "RUNNING"
                        elif code == "3":
                            metrics["status"] = "FINISHED"
                        elif code == "4":
                            metrics["status"] = "FAILED"
                        else:
                            metrics["status"] = code
        except Exception:
            pass

    # Read metrics text files
    metrics_dir = run_dir / "metrics"
    if metrics_dir.exists():
        metric_keys = [
            "train_loss",
            "val_loss",
            "cer",
            "wer",
            "learning_rate",
            "best_val_loss",
            "epoch_time_sec",
        ]
        for k in metric_keys:
            fpath = metrics_dir / k
            if fpath.exists():
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                    if lines:
                        last_line = lines[-1].split()
                        if len(last_line) >= 2:
                            val = float(last_line[1])
                            metrics[k] = val
                        if len(last_line) >= 3 and k == "val_loss":
                            metrics["steps_recorded"] = int(last_line[2])
                except Exception:
                    pass

    return metrics


def check_checkpoints(ckpt_dir: Path) -> Dict[str, Any]:
    """Inspects checkpoint files on disk strictly read-only."""
    info: Dict[str, Any] = {
        "best_exists": False,
        "best_size_mb": 0.0,
        "best_mtime": None,
        "latest_exists": False,
        "latest_size_mb": 0.0,
        "latest_mtime": None,
    }
    if not ckpt_dir.exists():
        return info

    b_path = ckpt_dir / "best_model.pt"
    if b_path.exists():
        stat = b_path.stat()
        info["best_exists"] = True
        info["best_size_mb"] = stat.st_size / (1024 * 1024)
        info["best_mtime"] = datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M:%S")

    l_path = ckpt_dir / "latest_model.pt"
    if l_path.exists():
        stat = l_path.stat()
        info["latest_exists"] = True
        info["latest_size_mb"] = stat.st_size / (1024 * 1024)
        info["latest_mtime"] = datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M:%S")

    return info


def make_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Renders a simple ASCII progress bar."""
    if total <= 0:
        return "[" + "░" * width + "] 0%"
    ratio = max(0.0, min(1.0, current / total))
    filled = int(round(width * ratio))
    bar = "█" * filled + "░" * (width - filled)
    percent = int(ratio * 100)
    return f"[{bar}] {percent:3d}%"


def format_duration(seconds: float) -> str:
    """Formats seconds into human-readable duration."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def render_dashboard(
    log_path: Path,
    training_start_time: Optional[float] = None,
) -> None:
    """Renders one frame of the live dashboard."""
    pid = find_training_pid()
    is_alive = pid is not None
    gpu = get_gpu_info()
    log_data = parse_log(log_path)
    mlflow_data = parse_mlflow_metrics(MLRUNS_RUN_DIR)
    ckpt_data = check_checkpoints(CHECKPOINT_DIR)

    epochs_done = log_data["epochs_completed"]
    last_completed_epoch = len(epochs_done)

    # Determine status & stage
    if log_data["is_complete"]:
        status_line = "STATUS: ✅ TRAINING COMPLETE"
        current_stage = "COMPLETED"
        current_epoch = TOTAL_EPOCHS
    elif not is_alive:
        status_line = "STATUS: 🔴 TRAINING PROCESS STOPPED"
        current_stage = "STOPPED"
        current_epoch = last_completed_epoch
    else:
        status_line = "STATUS: 🟢 RUNNING"
        current_epoch = min(TOTAL_EPOCHS, last_completed_epoch + 1)
        # Stage estimation
        current_stage = "BATCH TRAINING"

    # Batches & Progress calculation
    # If training is active in current epoch:
    if current_stage == "COMPLETED":
        current_batch = TOTAL_BATCHES_PER_EPOCH
        epoch_prog_current = TOTAL_BATCHES_PER_EPOCH
        overall_prog_current = TOTAL_EPOCHS * TOTAL_BATCHES_PER_EPOCH
    elif current_stage == "STOPPED":
        current_batch = 0
        epoch_prog_current = 0
        overall_prog_current = last_completed_epoch * TOTAL_BATCHES_PER_EPOCH
    else:
        # We estimate batch progress within the current epoch using elapsed time from last checkpoint
        # Typical cached epoch takes ~1400-1450s total (~1150s training + ~250s validation)
        # Batch rate is approx 1150s / 1500 batches ~= 0.77s / batch
        sec_since_epoch_start = 0.0
        if ckpt_data["latest_mtime"]:
            try:
                now = datetime.now()
                last_dt = datetime.strptime(ckpt_data["latest_mtime"], "%H:%M:%S")
                last_dt = now.replace(
                    hour=last_dt.hour,
                    minute=last_dt.minute,
                    second=last_dt.second,
                    microsecond=0,
                )
                sec_since_epoch_start = max(0.0, (now - last_dt).total_seconds())
            except Exception:
                sec_since_epoch_start = 0.0

        if sec_since_epoch_start > 1200:
            current_stage = "VALIDATION"
            current_batch = TOTAL_BATCHES_PER_EPOCH
        else:
            current_stage = "BATCH TRAINING"
            est_batch = int(sec_since_epoch_start / 0.77)
            current_batch = min(TOTAL_BATCHES_PER_EPOCH, max(1, est_batch))

        epoch_prog_current = current_batch
        overall_prog_current = (last_completed_epoch * TOTAL_BATCHES_PER_EPOCH) + current_batch

    total_overall_batches = TOTAL_EPOCHS * TOTAL_BATCHES_PER_EPOCH
    global_step = overall_prog_current

    epoch_bar = make_progress_bar(epoch_prog_current, TOTAL_BATCHES_PER_EPOCH, width=20)
    overall_bar = make_progress_bar(overall_prog_current, total_overall_batches, width=20)

    # Best metrics
    best_val_loss = None
    best_epoch = None
    if epochs_done:
        best_entry = min(epochs_done, key=lambda x: x["val_loss"])
        best_val_loss = best_entry["val_loss"]
        best_epoch = best_entry["epoch"]
    elif mlflow_data["best_val_loss"] is not None:
        best_val_loss = mlflow_data["best_val_loss"]

    # Latest epoch metrics
    latest_train_loss = "N/A"
    latest_val_loss = "N/A"
    latest_cer = "N/A"
    latest_wer = "N/A"
    latest_lr = "N/A"

    if epochs_done:
        latest = epochs_done[-1]
        latest_train_loss = f"{latest['train_loss']:.4f}"
        latest_val_loss = f"{latest['val_loss']:.4f}"
        latest_cer = f"{latest['cer']:.4f} ({latest['cer']*100:.2f}%)"
        latest_wer = f"{latest['wer']:.4f} ({latest['wer']*100:.2f}%)"
        latest_lr = latest["lr"]

    # Time calculations
    # Typical cached epoch duration ~ 1437s
    typical_epoch_duration = 1438.0
    if len(epochs_done) >= 2:
        # Use epoch 2 cached duration
        typical_epoch_duration = epochs_done[-1]["duration_sec"]

    remaining_epochs = max(0, TOTAL_EPOCHS - current_epoch)
    est_remaining_sec = (remaining_epochs * typical_epoch_duration) + max(
        0.0, typical_epoch_duration - (epoch_prog_current / TOTAL_BATCHES_PER_EPOCH * typical_epoch_duration)
    )

    if current_stage == "COMPLETED":
        eta_str = "Completed"
    elif not is_alive:
        eta_str = "Process inactive"
    else:
        eta_str = format_duration(est_remaining_sec)

    # Format Output
    output_lines = [
        "============================================================",
        "       LIPREAD — LIVE MVP TRAINING MONITOR              ",
        "============================================================",
        "",
        status_line,
        "",
        f"PID: {pid if pid else 'None'}",
        f"Current Stage: {current_stage}",
        "",
        f"Epoch:         {current_epoch} / {TOTAL_EPOCHS}",
        f"Batch:         {current_batch:,} / {TOTAL_BATCHES_PER_EPOCH:,}",
        f"Epoch Progress:   {epoch_bar}",
        f"Overall Progress: {overall_bar}",
        f"Global Step:   {global_step:,}",
        "",
        "------------------------------------------------------------",
        "LATEST METRICS",
        "------------------------------------------------------------",
        f"Train Loss:          {latest_train_loss}",
        f"Validation Loss:     {latest_val_loss}",
        f"CER:                 {latest_cer}",
        f"WER:                 {latest_wer}",
        f"Learning Rate:       {latest_lr}",
        "",
        f"Best Validation Loss: {f'{best_val_loss:.4f}' if best_val_loss else 'N/A'}",
        f"Best Epoch:           {best_epoch if best_epoch else 'N/A'}",
        "",
        "------------------------------------------------------------",
        "EPOCH HISTORY",
        "------------------------------------------------------------",
        f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'CER':<8} | {'WER':<8} | {'Time':<8}",
        "-" * 60,
    ]

    if not epochs_done:
        output_lines.append("No completed epochs logged yet.")
    else:
        for ep in epochs_done:
            t_fmt = format_duration(ep["duration_sec"])
            output_lines.append(
                f"{ep['epoch']:<6} | {ep['train_loss']:<10.4f} | {ep['val_loss']:<10.4f} | "
                f"{ep['cer']:<8.4f} | {ep['wer']:<8.4f} | {t_fmt:<8}"
            )

    output_lines.extend([
        "",
        "------------------------------------------------------------",
        "GPU",
        "------------------------------------------------------------",
        f"GPU:          {gpu['model']}",
        f"Utilization:  {gpu['utilization']}",
        f"VRAM:         {gpu['vram_used']} / {gpu['vram_total']}",
        f"Temperature:  {gpu['temp']}",
        "",
        "------------------------------------------------------------",
        "TIME",
        "------------------------------------------------------------",
        f"Estimated Remaining: {eta_str}",
        f"Last Checkpoint:     {ckpt_data['latest_mtime'] if ckpt_data['latest_mtime'] else 'None'}",
        "",
        "------------------------------------------------------------",
        "LATEST ACTIVITY",
        "------------------------------------------------------------",
        log_data["latest_activity"],
        "============================================================",
    ])

    print("\n".join(output_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="LipRead Live Training Monitor")
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to training log file",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (snapshot mode)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations before exiting (default: infinite)",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear screen between updates",
    )

    args = parser.parse_args()

    # Verify log path fallback if default doesn't exist
    log_path = args.log
    if not log_path.exists():
        # Check current working directory or alt paths
        alt = Path("task-1057.log")
        if alt.exists():
            log_path = alt

    iteration = 0
    try:
        while True:
            if not args.no_clear and not args.once and sys.stdout.isatty():
                os.system("cls" if os.name == "nt" else "clear")

            render_dashboard(log_path=log_path)
            sys.stdout.flush()
            iteration += 1

            if args.once:
                break
            if args.iterations is not None and iteration >= args.iterations:
                break

            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")


if __name__ == "__main__":
    main()
