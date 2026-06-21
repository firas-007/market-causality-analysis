"""Diagnostics for fitted VAR models."""

from __future__ import annotations

from typing import Any

import pandas as pd


def check_residual_autocorrelation(var_results: Any) -> Any:
    """Run the Portmanteau whiteness test for VAR residual autocorrelation."""
    nlags = max(var_results.k_ar + 5, 10)
    return var_results.test_whiteness(nlags=nlags)


def check_residual_normality(var_results: Any) -> Any:
    """Run the VAR residual normality test."""
    return var_results.test_normality()


def check_model_stability(var_results: Any) -> bool:
    """Return True when all VAR roots are outside the unit circle."""
    return bool(var_results.is_stable(verbose=False))


def summarize_var_diagnostics(var_results: Any) -> pd.DataFrame:
    """Summarize stability, residual autocorrelation, and residual normality diagnostics."""
    whiteness = check_residual_autocorrelation(var_results)
    normality = check_residual_normality(var_results)
    rows = [
        {
            "diagnostic": "VAR stability",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "conclusion": "Stable" if check_model_stability(var_results) else "Unstable",
        },
        {
            "diagnostic": "Residual autocorrelation",
            "statistic": float(whiteness.test_statistic),
            "p_value": float(whiteness.pvalue),
            "conclusion": (
                "No significant residual autocorrelation"
                if whiteness.pvalue >= 0.05
                else "Residual autocorrelation detected"
            ),
        },
        {
            "diagnostic": "Residual normality",
            "statistic": float(normality.test_statistic),
            "p_value": float(normality.pvalue),
            "conclusion": (
                "Do not reject normality" if normality.pvalue >= 0.05 else "Reject normality"
            ),
        },
    ]
    return pd.DataFrame(rows)
