"""DataLoader factory and batch generation for LipRead.

Provides configurable, Windows-friendly DataLoader construction with conservative
num_workers defaults for NVIDIA GTX 1650 (4 GB VRAM) architectures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.collate import pad_collate_fn
from src.data.egcllc_dataset import EGCLLCDataset
from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("lipread.data.dataloader")


def create_dataloader(
    dataset: Dataset,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """Constructs a configured PyTorch DataLoader.

    Args:
        dataset: Target PyTorch Dataset (e.g. EGCLLCDataset).
        batch_size: Batch size (default: 2 or from config).
        shuffle: Whether to shuffle samples (default: True).
        num_workers: Number of subprocess workers. Defaults to 0 (safest on Windows).
        pin_memory: Whether to pin memory for CUDA transfers (default: True if CUDA is available).
        collate_fn: Collation function (default: pad_collate_fn).

    Returns:
        PyTorch DataLoader instance.
    """
    cfg = get_config().to_dict()
    loader_cfg = cfg.get("loader", {})

    b_size = batch_size if batch_size is not None else loader_cfg.get("batch_size", 2)
    shuf = shuffle if shuffle is not None else loader_cfg.get("shuffle", True)
    workers = num_workers if num_workers is not None else loader_cfg.get("num_workers", 0)

    # Enable pin_memory only if CUDA is available
    if pin_memory is not None:
        pin = pin_memory
    else:
        pin = loader_cfg.get("pin_memory", torch.cuda.is_available())

    fn = collate_fn if collate_fn is not None else pad_collate_fn

    logger.debug(
        "Creating DataLoader (batch_size=%d, shuffle=%s, num_workers=%d, pin_memory=%s)",
        b_size,
        shuf,
        workers,
        pin,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=b_size,
        shuffle=shuf,
        num_workers=workers,
        pin_memory=pin,
        collate_fn=fn,
    )


def create_split_dataloaders(
    metadata_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = False,
    batch_size: int = 2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience helper to create train, val, and test DataLoaders simultaneously.

    Args:
        metadata_dir: Path to directory containing train.jsonl, val.jsonl, test.jsonl.
        use_cache: Whether to enable preprocessed tensor caching.
        batch_size: Batch size for loaders.
        num_workers: Number of workers (default: 0).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_ds = EGCLLCDataset(split="train", metadata_file=None, use_cache=use_cache)
    val_ds = EGCLLCDataset(split="val", metadata_file=None, use_cache=use_cache)
    test_ds = EGCLLCDataset(split="test", metadata_file=None, use_cache=use_cache)

    train_loader = create_dataloader(
        dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = create_dataloader(
        dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = create_dataloader(
        dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
