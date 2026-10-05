"""Central configuration system for LipRead.

Provides robust, configurable path resolution, environment variable overrides,
and YAML configuration loading with sensible defaults for 4 GB VRAM development.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logging_config import get_logger

logger = get_logger("lipread.config")

# Project root directory: D:\Lipreading end to end detection
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Configuration directory & default YAML path
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH: Path = CONFIG_DIR / "config.yaml"


def resolve_data_root() -> Path:
    """Resolves the root directory for datasets, checkpoints, and MLflow artifacts.

    Resolution precedence:
      1. LIPREAD_DATA_ROOT environment variable.
      2. E:\\LipRead_Data (primary large-storage drive).
      3. Fallback to PROJECT_ROOT / "data" if E: drive is unavailable (e.g. CI environments).

    Returns:
        Path to the resolved data root.
    """
    env_root = os.getenv("LIPREAD_DATA_ROOT")
    if env_root:
        resolved = Path(env_root).resolve()
        logger.debug("Using data root from LIPREAD_DATA_ROOT: %s", resolved)
        return resolved

    # Default to E:\LipRead_Data
    primary_data_root = Path("E:/LipRead_Data").resolve()
    try:
        # Check if the drive or parent root is accessible
        primary_drive = Path(primary_data_root.drive + "\\") if primary_data_root.drive else primary_data_root
        if primary_drive.exists():
            return primary_data_root
    except Exception:
        pass

    # Fallback to local project data directory if external drive is inaccessible
    fallback = (PROJECT_ROOT / "data").resolve()
    logger.warning(
        "Primary data drive '%s' is not accessible. Falling back to local directory: %s",
        primary_data_root,
        fallback,
    )
    return fallback


# Centralized directory paths
DATA_ROOT: Path = resolve_data_root()
RAW_DATA_DIR: Path = DATA_ROOT / "raw"
PROCESSED_DATA_DIR: Path = DATA_ROOT / "processed"
METADATA_DIR: Path = DATA_ROOT / "metadata"
CHECKPOINT_DIR: Path = DATA_ROOT / "checkpoints"
MLFLOW_DIR: Path = DATA_ROOT / "mlruns"


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML file. Defaults to configs/config.yaml.

    Returns:
        Dictionary containing configuration parameters.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        logger.warning("Configuration file %s does not exist. Using empty config.", path)
        return {}

    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except ImportError:
        logger.warning("PyYAML is not installed. Unable to load YAML directly.")
        return {}
    except Exception as exc:
        logger.error("Failed to load configuration from %s: %s", path, exc)
        return {}


def ensure_directories(verbose: bool = False) -> Dict[str, Path]:
    """Ensures all essential data and artifact directories exist.

    Args:
        verbose: If True, logs created directories.

    Returns:
        Dictionary mapping directory names to their Path objects.
    """
    directories = {
        "data_root": DATA_ROOT,
        "raw": RAW_DATA_DIR,
        "processed": PROCESSED_DATA_DIR,
        "metadata": METADATA_DIR,
        "checkpoints": CHECKPOINT_DIR,
        "mlruns": MLFLOW_DIR,
    }

    for name, directory in directories.items():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if verbose:
                logger.info("Directory verified/created: %s (%s)", name, directory)
        except Exception as exc:
            logger.error("Could not create directory %s (%s): %s", name, directory, exc)

    return directories


class Config:
    """Structured configuration wrapper."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._raw = config_dict or load_yaml_config()

    @property
    def project(self) -> Dict[str, Any]:
        return self._raw.get("project", {})

    @property
    def data(self) -> Dict[str, Any]:
        return self._raw.get("data", {})

    @property
    def video(self) -> Dict[str, Any]:
        return self._raw.get("video", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self._raw.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self._raw.get("training", {})

    @property
    def preprocessing(self) -> Dict[str, Any]:
        return self._raw.get("preprocessing", {})

    @property
    def dataset(self) -> Dict[str, Any]:
        return self._raw.get("dataset", {})

    @property
    def loader(self) -> Dict[str, Any]:
        return self._raw.get("loader", {})

    @property
    def mlflow(self) -> Dict[str, Any]:
        return self._raw.get("mlflow", {})

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._raw)


_GLOBAL_CONFIG: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """Returns the singleton Config instance."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None or reload:
        _GLOBAL_CONFIG = Config()
    return _GLOBAL_CONFIG
