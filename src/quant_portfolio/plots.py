"""Small set of high-information figures for the research report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_portfolio.backtest import BacktestResult
from quant_portfolio.metrics import wealth_index


DISPLAY_NAMES = {
    "buy_and_hold_equal_weight": "Buy & hold 1/N",
    "quarterly_equal_weight": "Quarterly 1/N",
    "inverse_volatility": "Inverse volatility",
    "sample_min_variance": "Sample-cov min variance",
    "ledoit_wolf_min_variance": "Ledoit-Wolf min variance",
    "spy": "SPY",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_performance(results: dict[str, BacktestResult], path: Path) -> None:
    """Plot net wealth and drawdowns for all strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name, result in results.items():
        wealth = wealth_index(result.net_returns)
        drawdown = wealth / wealth.cummax().clip(lower=1.0) - 1.0
        width = 2.2 if "min_variance" in name else 1.2
        axes[0].plot(wealth.index, wealth, label=DISPLAY_NAMES[name], linewidth=width)
        axes[1].plot(
            drawdown.index,
            drawdown,
            label=DISPLAY_NAMES[name],
            linewidth=width,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Wealth (log scale)")
    axes[0].set_title("Out-of-sample net performance")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(ncol=3, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(fig, path)


def plot_covariance_diagnostics(diagnostics: pd.DataFrame, path: Path) -> None:
    """Plot estimator conditioning and effective portfolio breadth."""
    selected = diagnostics[
        diagnostics["strategy"].isin(
            ["sample_min_variance", "ledoit_wolf_min_variance"]
        )
    ].copy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name, group in selected.groupby("strategy"):
        label = DISPLAY_NAMES[name]
        axes[0].plot(group.index, group["condition_number"], label=label, linewidth=1.8)
        axes[1].plot(
            group.index,
            group["effective_n_assets"],
            label=label,
            linewidth=1.8,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Condition number (log)")
    axes[0].set_title("Covariance conditioning and portfolio concentration")
    axes[1].set_ylabel("Effective number of assets")
    axes[1].set_xlabel("Rebalance date")
    axes[0].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(fig, path)


def _difference_grid(
    sensitivity: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    pivot = sensitivity.pivot_table(
        index=["lookback", "cost_bps"], columns="strategy", values=metric
    )
    difference = (
        pivot["ledoit_wolf_min_variance"] - pivot["sample_min_variance"]
    )
    return difference.unstack("cost_bps")


def plot_sensitivity(sensitivity: pd.DataFrame, path: Path) -> None:
    """Plot shrinkage-minus-sample differences across robustness settings."""
    volatility = _difference_grid(sensitivity, "annual_volatility") * 10_000
    turnover = _difference_grid(sensitivity, "annualized_turnover")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, grid, title, fmt in [
        (axes[0], volatility, "Annual volatility: LW - sample (bp)", ".1f"),
        (axes[1], turnover, "Annual turnover: LW - sample", ".2f"),
    ]:
        values = grid.to_numpy()
        # A symmetric colour scale makes the sign directly interpretable: negative
        # cells mean Ledoit-Wolf is lower than the sample-covariance portfolio.
        limit = np.nanmax(np.abs(values)) or 1.0
        image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_xticks(range(len(grid.columns)), [f"{x:g} bp" for x in grid.columns])
        axis.set_yticks(range(len(grid.index)), [str(x) for x in grid.index])
        axis.set_xlabel("Transaction cost")
        axis.set_ylabel("Lookback (days)")
        axis.set_title(title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(
                    column,
                    row,
                    format(values[row, column], fmt),
                    ha="center",
                    va="center",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    _save(fig, path)
