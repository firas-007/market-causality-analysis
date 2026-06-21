"""Preprocessing routines for market prices, differences, and returns."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.utils import ensure_directory

logger = logging.getLogger(__name__)


def clean_market_data(df: pd.DataFrame, method: str = "drop") -> pd.DataFrame:
    """Clean market data by sorting dates and handling missing values."""
    if df.empty:
        raise ValueError("Input market data is empty.")

    clean = df.copy()
    if "Date" in clean.columns:
        clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
        clean = clean.dropna(subset=["Date"]).set_index("Date")

    clean = clean.sort_index()
    clean = clean.apply(pd.to_numeric, errors="coerce")

    if method == "drop":
        clean = clean.dropna()
    elif method == "ffill":
        clean = clean.ffill().dropna()
    elif method == "interpolate":
        clean = clean.interpolate(method="time").ffill().bfill()
    else:
        raise ValueError("method must be one of: 'drop', 'ffill', 'interpolate'.")

    logger.info("Clean market data shape after %s: %s", method, clean.shape)
    return clean


def compute_first_difference(df: pd.DataFrame) -> pd.DataFrame:
    """Compute first differences for each numeric series."""
    return df.diff().dropna()


def compute_returns(df: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Compute simple or log returns for each numeric series."""
    if method == "simple":
        return df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if method == "log":
        return compute_log_returns(df)
    raise ValueError("method must be either 'simple' or 'log'.")


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute continuously compounded log returns."""
    positive = df.where(df > 0)
    return np.log(positive).diff().replace([np.inf, -np.inf], np.nan).dropna()


def save_processed_datasets(
    clean: pd.DataFrame,
    diff: pd.DataFrame,
    returns: pd.DataFrame,
    log_returns: pd.DataFrame,
) -> None:
    """Save processed market datasets to the configured processed-data directory."""
    ensure_directory(PROCESSED_DATA_DIR)
    outputs = {
        "market_data_clean.csv": clean,
        "market_data_first_difference.csv": diff,
        "market_data_returns.csv": returns,
        "market_data_log_returns.csv": log_returns,
    }
    for file_name, data in outputs.items():
        path = PROCESSED_DATA_DIR / file_name
        data.to_csv(path)
        logger.info("Saved %s", path)
