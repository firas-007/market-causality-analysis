"""Outlier detection utilities for financial return series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Return a Boolean outlier mask using the interquartile range rule."""
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (df.lt(lower, axis=1) | df.gt(upper, axis=1)).astype(bool)


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Return a Boolean outlier mask using absolute z-scores."""
    z_scores = pd.DataFrame(
        np.abs(stats.zscore(df, nan_policy="omit")),
        index=df.index,
        columns=df.columns,
    )
    return z_scores.gt(threshold).astype(bool)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize IQR and z-score outlier counts and percentages by variable."""
    iqr_mask = detect_outliers_iqr(df)
    zscore_mask = detect_outliers_zscore(df)
    total = len(df)
    summary = pd.DataFrame(
        {
            "iqr_count": iqr_mask.sum(),
            "iqr_percentage": iqr_mask.sum() / total * 100,
            "zscore_count": zscore_mask.sum(),
            "zscore_percentage": zscore_mask.sum() / total * 100,
        }
    )
    summary.index.name = "variable"
    return summary
