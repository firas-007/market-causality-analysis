"""Project-wide configuration constants."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

SERIES_FILES: dict[str, str] = {
    "MASI": "MASI.csv",
    "CAC40": "CAC40.csv",
    "SP500": "SP500.csv",
    "EUR_MAD": "EUR_MAD.csv",
    "USD_MAD": "USD_MAD.csv",
}

DATE_COLUMN_CANDIDATES = ("Date", "date", "DATE")
CLOSE_COLUMN_CANDIDATES = ("Close", "close", "Adj Close", "Price", "Dernier")

