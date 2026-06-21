"""Johansen cointegration analysis helpers."""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def johansen_cointegration_test(
    df: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> pd.DataFrame:
    """Run the Johansen trace test and return a tabular summary."""
    clean = df.dropna()
    result = coint_johansen(clean, det_order=det_order, k_ar_diff=k_ar_diff)
    rows: list[dict[str, float | int | str]] = []
    for rank, trace_stat in enumerate(result.lr1):
        rows.append(
            {
                "rank_null": rank,
                "trace_statistic": float(trace_stat),
                "critical_value_90": float(result.cvt[rank, 0]),
                "critical_value_95": float(result.cvt[rank, 1]),
                "critical_value_99": float(result.cvt[rank, 2]),
                "reject_at_5_percent": trace_stat > result.cvt[rank, 1],
            }
        )
    return pd.DataFrame(rows)


def interpret_johansen_results(results: pd.DataFrame) -> str:
    """Produce a concise interpretation of Johansen trace-test results."""
    rejected = results[results["reject_at_5_percent"]]
    if rejected.empty:
        return (
            "At the 5% level, the trace test does not reject the null of no "
            "cointegration. The evidence for a long-run equilibrium relation is weak."
        )
    max_rejected_rank = int(rejected["rank_null"].max())
    return (
        "At the 5% level, the trace test rejects cointegration ranks up to "
        f"{max_rejected_rank}. This suggests at least {max_rejected_rank + 1} "
        "cointegrating relationship(s), subject to lag and deterministic-term choices."
    )
