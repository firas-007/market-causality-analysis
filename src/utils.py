"""Shared utility helpers for notebooks and source modules."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a compact logging format for notebook execution."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def ensure_directory(path: Path) -> Path:
    """Create a directory when it does not exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_project_root_to_path(project_root: Path) -> None:
    """Add the project root to sys.path for notebook imports."""
    root = str(project_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def save_table(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    """Persist a DataFrame as CSV after creating the parent directory."""
    ensure_directory(path.parent)
    df.to_csv(path, index=index)
