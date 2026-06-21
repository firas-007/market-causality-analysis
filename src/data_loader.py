"""Data loading helpers for financial market CSV files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import CLOSE_COLUMN_CANDIDATES, DATE_COLUMN_CANDIDATES, RAW_DATA_DIR

logger = logging.getLogger(__name__)


def find_date_column(df: pd.DataFrame) -> str:
    """Return the first recognized date column name from a DataFrame."""
    for column in DATE_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "No valid date column found. Expected one of: "
        f"{', '.join(DATE_COLUMN_CANDIDATES)}"
    )


def find_close_column(df: pd.DataFrame) -> str:
    """Return the first recognized close/price column name from a DataFrame."""
    for column in CLOSE_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "No valid close column found. Expected one of: "
        f"{', '.join(CLOSE_COLUMN_CANDIDATES)}"
    )


def load_financial_series(file_path: Path, series_name: str) -> pd.DataFrame:
    """Load one financial series, standardize columns, and validate observations."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing raw data file for {series_name}: {file_path}")

    logger.info("Loading %s from %s", series_name, file_path)
    df = pd.read_csv(file_path)
    date_column = find_date_column(df)
    close_column = find_close_column(df)

    series = df[[date_column, close_column]].copy()
    series.columns = ["Date", series_name]
    series["Date"] = pd.to_datetime(series["Date"], errors="coerce")
    series[series_name] = pd.to_numeric(series[series_name], errors="coerce")
    series = (
        series.dropna(subset=["Date", series_name])
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    if series.empty:
        raise ValueError(f"No valid observations were loaded for {series_name}.")

    return series


def load_all_series(series_files: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Load all configured market and exchange-rate series from raw CSV files."""
    loaded: dict[str, pd.DataFrame] = {}
    for series_name, file_name in series_files.items():
        loaded[series_name] = load_financial_series(RAW_DATA_DIR / file_name, series_name)
    return loaded


def merge_financial_series(series_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge standardized financial series on the Date column using an inner join."""
    if not series_dict:
        raise ValueError("series_dict cannot be empty.")

    iterator = iter(series_dict.values())
    merged = next(iterator).copy()
    for series in iterator:
        merged = pd.merge(merged, series, on="Date", how="inner")

    merged = merged.sort_values("Date").set_index("Date")
    logger.info("Merged market data shape: %s", merged.shape)
    return merged
