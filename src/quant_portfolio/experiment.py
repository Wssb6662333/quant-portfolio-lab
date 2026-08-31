"""End-to-end empirical study and reproducible artifact generation."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yfinance

from quant_portfolio import __version__
from quant_portfolio.backtest import (
    BacktestResult,
    run_strategy_suite,
    simulate_target_weights,
)
from quant_portfolio.config import ExperimentConfig
from quant_portfolio.data import calculate_simple_returns, load_or_download_prices
from quant_portfolio.metrics import performance_statistics, summarise_backtests
from quant_portfolio.plots import (
    plot_covariance_diagnostics,
    plot_performance,
    plot_sensitivity,
)
from quant_portfolio.risk import var_backtest_summary


REGIMES = {
    "pre_covid": ("2013-01-02", "2019-12-31"),
    "pandemic_and_recovery": ("2020-01-01", "2021-12-31"),
    "inflation_and_rate_shock": ("2022-01-01", "2025-12-31"),
}


def _returns_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    return pd.concat(
        [result.net_returns for result in results.values()], axis=1, join="inner"
    )


def _sensitivity_table(
    returns: pd.DataFrame,
    config: ExperimentConfig,
    main_results: dict[str, BacktestResult],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for lookback in config.sensitivity_lookbacks:
        # Targets do not depend on transaction costs, so calculate them once per
        # lookback and reuse them across cost assumptions for a controlled comparison.
        if lookback == config.lookback:
            target_source = main_results
        else:
            target_source = run_strategy_suite(
                returns,
                evaluation_start=config.evaluation_start,
                lookback=lookback,
                transaction_cost_bps=0.0,
                max_weight=config.max_weight,
            )
        for cost_bps in config.sensitivity_costs_bps:
            results = {
                name: simulate_target_weights(
                    returns,
                    result.targets,
                    transaction_cost_bps=cost_bps,
                    name=name,
                    diagnostics=result.diagnostics,
                )
                for name, result in target_source.items()
            }
            summary = summarise_backtests(
                results,
                annualization=config.annualization,
                confidence=config.var_confidence,
            ).reset_index()
            summary.insert(0, "cost_bps", cost_bps)
            summary.insert(0, "lookback", lookback)
            frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def _regime_table(
    returns_by_strategy: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for regime, (start, end) in REGIMES.items():
        subset = returns_by_strategy.loc[start:end]
        for strategy in subset.columns:
            row = performance_statistics(
                subset[strategy],
                annualization=config.annualization,
                confidence=config.var_confidence,
            )
            row.update({"regime": regime, "strategy": strategy})
            rows.append(row)
    return pd.DataFrame(rows).set_index(["regime", "strategy"])


def _diagnostics_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    frames = [
        result.diagnostics
        for result in results.values()
        if not result.diagnostics.empty
    ]
    diagnostics = pd.concat(frames).sort_index()
    diagnostics.index.name = "date"
    return diagnostics


def _research_findings(
    performance: pd.DataFrame,
    diagnostics: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> dict[str, float | int | str]:
    sample = performance.loc["sample_min_variance"]
    shrinkage = performance.loc["ledoit_wolf_min_variance"]
    equal_weight = performance.loc["quarterly_equal_weight"]
    sample_condition = diagnostics.loc[
        diagnostics["strategy"] == "sample_min_variance", "condition_number"
    ].median()
    shrinkage_condition = diagnostics.loc[
        diagnostics["strategy"] == "ledoit_wolf_min_variance", "condition_number"
    ].median()
    # Pair estimators within the same lookback-cost scenario before counting wins.
    paired = sensitivity.pivot_table(
        index=["lookback", "cost_bps"],
        columns="strategy",
        values="annual_volatility",
    )
    wins = int(
        (
            paired["ledoit_wolf_min_variance"]
            < paired["sample_min_variance"]
        ).sum()
    )
    return {
        "research_question": (
            "Does Ledoit-Wolf covariance shrinkage improve out-of-sample risk and "
            "weight stability relative to sample-covariance minimum variance?"
        ),
        "evaluation_start": str(performance.loc["sample_min_variance", "start"]),
        "evaluation_end": str(performance.loc["sample_min_variance", "end"]),
        "evaluation_observations": int(sample["observations"]),
        "ledoit_wolf_minus_sample_annual_volatility_percentage_points": float(
            100 * (shrinkage["annual_volatility"] - sample["annual_volatility"])
        ),
        "ledoit_wolf_minus_sample_max_drawdown_percentage_points": float(
            100 * (shrinkage["max_drawdown"] - sample["max_drawdown"])
        ),
        "ledoit_wolf_minus_sample_annual_turnover": float(
            shrinkage["annualized_turnover"] - sample["annualized_turnover"]
        ),
        "ledoit_wolf_volatility_reduction_vs_quarterly_equal_weight_pct": float(
            100
            * (
                1
                - shrinkage["annual_volatility"]
                / equal_weight["annual_volatility"]
            )
        ),
        "median_covariance_condition_number_reduction_pct": float(
            100 * (1 - shrinkage_condition / sample_condition)
        ),
        "sensitivity_scenarios_with_lower_volatility_than_sample": wins,
        "sensitivity_scenarios": int(len(paired)),
    }


def run_experiment(
    project_root: Path,
    *,
    config: ExperimentConfig | None = None,
    refresh_data: bool = False,
) -> dict[str, object]:
    """Run the full study and write all public result artifacts."""
    config = config or ExperimentConfig()
    project_root = Path(project_root).resolve()
    cache_path = project_root / "data" / "raw" / "etf_adjusted_close.csv"
    tables_dir = project_root / "reports" / "tables"
    figures_dir = project_root / "reports" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prices = load_or_download_prices(
        cache_path,
        config.tickers,
        config.data_start,
        config.data_end,
        refresh=refresh_data,
    )
    returns = calculate_simple_returns(prices)
    results = run_strategy_suite(
        returns,
        evaluation_start=config.evaluation_start,
        lookback=config.lookback,
        transaction_cost_bps=config.transaction_cost_bps,
        max_weight=config.max_weight,
    )
    performance = summarise_backtests(
        results,
        annualization=config.annualization,
        confidence=config.var_confidence,
    )
    returns_by_strategy = _returns_frame(results)
    risk_backtest = var_backtest_summary(
        returns_by_strategy,
        window=config.var_window,
        confidence=config.var_confidence,
    )
    sensitivity = _sensitivity_table(returns, config, results)
    regimes = _regime_table(returns_by_strategy, config)
    diagnostics = _diagnostics_table(results)

    performance.to_csv(tables_dir / "main_performance.csv")
    risk_backtest.to_csv(tables_dir / "var_backtest.csv")
    sensitivity.to_csv(tables_dir / "sensitivity.csv", index=False)
    regimes.to_csv(tables_dir / "regime_metrics.csv")
    diagnostics.to_csv(tables_dir / "covariance_diagnostics.csv")
    returns_by_strategy.to_csv(tables_dir / "strategy_returns.csv", index_label="date")

    plot_performance(results, figures_dir / "performance_and_drawdowns.png")
    plot_covariance_diagnostics(
        diagnostics, figures_dir / "covariance_diagnostics.png"
    )
    plot_sensitivity(sensitivity, figures_dir / "sensitivity_heatmaps.png")

    findings = _research_findings(performance, diagnostics, sensitivity)
    (tables_dir / "research_findings.json").write_text(
        json.dumps(findings, indent=2) + "\n", encoding="utf-8"
    )
    # Record enough environment detail to explain small differences after a data or
    # dependency refresh without embedding the local cache in version control.
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "Yahoo Finance via yfinance; auto-adjusted daily close",
        "price_start": prices.index.min().date().isoformat(),
        "price_end": prices.index.max().date().isoformat(),
        "price_observations": len(prices),
        "configuration": asdict(config),
        "versions": {
            "python": platform.python_version(),
            "quant_portfolio": __version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "yfinance": yfinance.__version__,
        },
    }
    (tables_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "prices": prices,
        "returns": returns,
        "results": results,
        "performance": performance,
        "risk_backtest": risk_backtest,
        "sensitivity": sensitivity,
        "regimes": regimes,
        "diagnostics": diagnostics,
        "findings": findings,
        "metadata": metadata,
    }
