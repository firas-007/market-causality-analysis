"""Stationarity tests for financial time-series analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(series: pd.Series) -> dict[str, Any]:
    """Run the Augmented Dickey-Fuller unit-root test on one series."""
    clean = series.dropna()
    statistic, p_value, used_lag, nobs, critical_values, icbest = adfuller(clean, autolag="AIC")
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "nobs": int(nobs),
        "critical_values": critical_values,
        "icbest": float(icbest),
        "conclusion": "Stationary" if p_value < 0.05 else "Non-stationary",
    }


def kpss_test(series: pd.Series) -> dict[str, Any]:
    """Run the KPSS stationarity test on one series."""
    clean = series.dropna()
    statistic, p_value, used_lags, critical_values = kpss(clean, regression="c", nlags="auto")
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "used_lags": int(used_lags),
        "critical_values": critical_values,
        "conclusion": "Stationary" if p_value >= 0.05 else "Non-stationary",
    }


def stationarity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a combined ADF/KPSS stationarity summary for each variable."""
    rows: list[dict[str, float | str]] = []
    for column in df.columns:
        adf = adf_test(df[column])
        kpss_result = kpss_test(df[column])
        final = (
            "Likely stationary"
            if adf["conclusion"] == "Stationary" and kpss_result["conclusion"] == "Stationary"
            else "Likely non-stationary or mixed evidence"
        )
        rows.append(
            {
                "variable": column,
                "ADF statistic": adf["statistic"],
                "ADF p-value": adf["p_value"],
                "ADF conclusion": adf["conclusion"],
                "KPSS statistic": kpss_result["statistic"],
                "KPSS p-value": kpss_result["p_value"],
                "KPSS conclusion": kpss_result["conclusion"],
                "final conclusion": final,
            }
        )
    return pd.DataFrame(rows)
