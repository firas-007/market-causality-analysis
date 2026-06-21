"""Visualization helpers for the market causality analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import ensure_directory


def _save_current_figure(save_path: Path) -> None:
    """Save the current matplotlib figure and close it."""
    ensure_directory(save_path.parent)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_series(df: pd.DataFrame, title: str, save_path: Path) -> None:
    """Plot all series on a shared time axis and save the figure."""
    sns.set_theme(style="whitegrid")
    ax = df.plot(figsize=(12, 6), linewidth=1.3)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Level")
    ax.legend(loc="best")
    _save_current_figure(save_path)


def plot_individual_series(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot each variable separately and save one figure per variable."""
    ensure_directory(save_dir)
    sns.set_theme(style="whitegrid")
    for column in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df[column], linewidth=1.2)
        ax.set_title(f"{column} over time")
        ax.set_xlabel("Date")
        ax.set_ylabel(column)
        _save_current_figure(save_dir / f"{column.lower()}_time_series.png")


def plot_returns(df: pd.DataFrame, save_path: Path) -> None:
    """Plot return series and save the figure."""
    sns.set_theme(style="whitegrid")
    ax = df.plot(figsize=(12, 6), linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Market and FX Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    _save_current_figure(save_path)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """Plot and save a correlation heatmap."""
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="vlag", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    _save_current_figure(save_path)


def plot_boxplots(df: pd.DataFrame, save_path: Path) -> None:
    """Plot and save boxplots for all variables."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, orient="h", ax=ax)
    ax.set_title("Distribution and Outlier Structure")
    ax.set_xlabel("Value")
    _save_current_figure(save_path)


def plot_outlier_counts(outlier_summary: pd.DataFrame, save_path: Path) -> None:
    """Plot and save outlier counts by detection method."""
    plot_data = outlier_summary[["iqr_count", "zscore_count"]].reset_index()
    plot_data = plot_data.melt(
        id_vars="variable",
        value_vars=["iqr_count", "zscore_count"],
        var_name="method",
        value_name="count",
    )
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_data, x="variable", y="count", hue="method", ax=ax)
    ax.set_title("Outlier Counts by Variable")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Number of observations")
    ax.tick_params(axis="x", rotation=30)
    _save_current_figure(save_path)
