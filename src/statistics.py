"""Statistical summaries and normality tests for financial series."""

from __future__ import annotations

import pandas as pd
from scipy import stats


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics including Jarque-Bera normality diagnostics."""
    rows: list[dict[str, float | str]] = []
    for column in df.columns:
        series = df[column].dropna()
        jb_stat, jb_pvalue = stats.jarque_bera(series)
        rows.append(
            {
                "variable": column,
                "count": float(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "standard_deviation": float(series.std(ddof=1)),
                "min": float(series.min()),
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "jarque_bera_statistic": float(jb_stat),
                "jarque_bera_p_value": float(jb_pvalue),
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def normality_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run Jarque-Bera normality tests for each variable."""
    rows: list[dict[str, float | str]] = []
    for column in df.columns:
        series = df[column].dropna()
        statistic, p_value = stats.jarque_bera(series)
        rows.append(
            {
                "variable": column,
                "jarque_bera_statistic": float(statistic),
                "p_value": float(p_value),
                "conclusion": "Reject normality" if p_value < 0.05 else "Do not reject normality",
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix for numeric series."""
    return df.corr()
