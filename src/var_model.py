"""VAR, Granger causality, IRF, and FEVD helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.tsa.api import VAR


def select_optimal_lag(df: pd.DataFrame, maxlags: int = 10) -> pd.DataFrame:
    """Select VAR lag order using AIC, BIC, HQIC, and FPE criteria."""
    clean = df.dropna()
    model = VAR(clean)
    rows: list[dict[str, float | int]] = []
    for lag in range(1, maxlags + 1):
        fitted = model.fit(lag)
        rows.append(
            {
                "lag": lag,
                "AIC": float(fitted.aic),
                "BIC": float(fitted.bic),
                "HQIC": float(fitted.hqic),
                "FPE": float(fitted.fpe),
            }
        )
    return pd.DataFrame(rows)


def fit_var(df: pd.DataFrame, lag: int) -> Any:
    """Fit a Vector Autoregression with the selected lag order."""
    if lag < 1:
        raise ValueError("VAR lag must be at least 1.")
    return VAR(df.dropna()).fit(lag)


def granger_causality_matrix(var_results: Any, variables: list[str]) -> pd.DataFrame:
    """Compute pairwise Granger causality p-values for all variable pairs."""
    matrix = pd.DataFrame(index=variables, columns=variables, dtype=float)
    for caused in variables:
        for causing in variables:
            if caused == causing:
                matrix.loc[caused, causing] = float("nan")
            else:
                test = var_results.test_causality(caused=caused, causing=[causing], kind="f")
                matrix.loc[caused, causing] = float(test.pvalue)
    return matrix


def targeted_granger_tests(
    var_results: Any,
    target: str,
    causes: list[str],
) -> pd.DataFrame:
    """Run Granger causality tests for selected cause variables against one target."""
    rows: list[dict[str, float | str]] = []
    for cause in causes:
        test = var_results.test_causality(caused=target, causing=[cause], kind="f")
        rows.append(
            {
                "caused_variable": target,
                "causing_variable": cause,
                "test_statistic": float(test.test_statistic),
                "p_value": float(test.pvalue),
                "conclusion": (
                    "Reject non-causality" if test.pvalue < 0.05 else "Do not reject non-causality"
                ),
            }
        )
    return pd.DataFrame(rows)


def impulse_response_analysis(var_results: Any, periods: int = 10) -> Any:
    """Create impulse-response analysis results for a fitted VAR model."""
    return var_results.irf(periods)


def variance_decomposition(var_results: Any, periods: int = 10) -> Any:
    """Create forecast error variance decomposition results for a fitted VAR model."""
    return var_results.fevd(periods)


def run_targeted_causality_suite(var_results: Any) -> pd.DataFrame:
    """Run the project-specific causality tests requested for the study."""
    pairs = [
        ("MASI", "SP500"),
        ("MASI", "CAC40"),
        ("MASI", "USD_MAD"),
        ("MASI", "EUR_MAD"),
        ("USD_MAD", "MASI"),
        ("EUR_MAD", "MASI"),
        ("CAC40", "SP500"),
        ("SP500", "CAC40"),
    ]
    rows: list[dict[str, float | str]] = []
    for caused, causing in pairs:
        test = var_results.test_causality(caused=caused, causing=[causing], kind="f")
        rows.append(
            {
                "caused_variable": caused,
                "causing_variable": causing,
                "test_statistic": float(test.test_statistic),
                "p_value": float(test.pvalue),
                "conclusion": (
                    "Reject non-causality" if test.pvalue < 0.05 else "Do not reject non-causality"
                ),
            }
        )
    return pd.DataFrame(rows)
