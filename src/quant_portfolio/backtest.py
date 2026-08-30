"""Look-ahead-safe target generation and daily portfolio accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_portfolio.covariance import (
    ledoit_wolf_covariance,
    sample_covariance,
)
from quant_portfolio.optimization import (
    equal_weights,
    inverse_volatility_weights,
    minimum_variance_weights,
)


STRATEGIES = (
    "buy_and_hold_equal_weight",
    "quarterly_equal_weight",
    "inverse_volatility",
    "sample_min_variance",
    "ledoit_wolf_min_variance",
    "spy",
)


@dataclass(frozen=True)
class BacktestResult:
    name: str
    net_returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    cost_fraction: pd.Series
    weights: pd.DataFrame
    targets: pd.DataFrame
    diagnostics: pd.DataFrame


def _validate_return_frame(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns must use a DatetimeIndex")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("return dates must be unique and increasing")
    if len(returns.columns) == 0 or returns.columns.has_duplicates:
        raise ValueError("return columns must be non-empty and unique")

    clean = returns.astype(float)
    values = clean.to_numpy()
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("returns must be finite and greater than -1")
    return clean


def quarterly_rebalance_dates(
    index: pd.DatetimeIndex,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Return the first observed trading date in each calendar quarter."""
    dates = pd.DatetimeIndex(index)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("return dates must be unique and increasing")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else dates.max()
    eligible = dates[(dates >= start_ts) & (dates <= end_ts)]
    if eligible.empty:
        raise ValueError("no return dates fall inside the evaluation interval")
    quarters = eligible.to_period("Q")
    return eligible[~quarters.duplicated()]


def _single_target(weights: pd.Series, date: pd.Timestamp) -> pd.DataFrame:
    target = weights.to_frame().T
    target.index = pd.DatetimeIndex([date], name="date")
    return target


def build_strategy_targets(
    returns: pd.DataFrame,
    strategy: str,
    *,
    evaluation_start: str,
    lookback: int,
    max_weight: float,
    evaluation_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build target weights using only observations strictly before each decision."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    if lookback < 2:
        raise ValueError("lookback must contain at least two observations")
    returns = _validate_return_frame(returns)

    rebalance_dates = quarterly_rebalance_dates(
        returns.index, evaluation_start, evaluation_end
    )
    first_date = rebalance_dates[0]
    labels = returns.columns

    if strategy == "buy_and_hold_equal_weight":
        return _single_target(equal_weights(labels), first_date), pd.DataFrame()
    if strategy == "spy":
        if "SPY" not in labels:
            raise ValueError("SPY benchmark requires an SPY return column")
        weights = pd.Series(0.0, index=labels)
        weights.loc["SPY"] = 1.0
        return _single_target(weights, first_date), pd.DataFrame()

    targets: list[pd.Series] = []
    diagnostic_rows: list[dict[str, object]] = []
    for rebalance_date in rebalance_dates:
        location = returns.index.get_loc(rebalance_date)
        if not isinstance(location, (int, np.integer)):
            raise ValueError("rebalance dates must resolve to one return observation")
        if location < lookback:
            raise ValueError(
                f"insufficient training data before {rebalance_date.date()}: "
                f"need {lookback}, found {location}"
            )
        # Python's exclusive stop index keeps the rebalance-date return out of the
        # estimation window: the final training observation is always t-1.
        training = returns.iloc[location - lookback : location]
        if training.index.max() >= rebalance_date:
            raise AssertionError("training data leaked into the holding period")

        estimate = None
        if strategy == "quarterly_equal_weight":
            weights = equal_weights(labels)
        elif strategy == "inverse_volatility":
            weights = inverse_volatility_weights(training)
        elif strategy == "sample_min_variance":
            estimate = sample_covariance(training)
            weights = minimum_variance_weights(estimate.matrix, max_weight=max_weight)
        else:
            estimate = ledoit_wolf_covariance(training)
            weights = minimum_variance_weights(estimate.matrix, max_weight=max_weight)

        weights.name = rebalance_date
        targets.append(weights)
        diagnostic_rows.append(
            {
                "date": rebalance_date,
                "strategy": strategy,
                "training_start": training.index.min(),
                "training_end": training.index.max(),
                "training_observations": len(training),
                "condition_number": (
                    estimate.condition_number if estimate is not None else np.nan
                ),
                "minimum_eigenvalue": (
                    estimate.minimum_eigenvalue if estimate is not None else np.nan
                ),
                "shrinkage": estimate.shrinkage if estimate is not None else np.nan,
                "effective_n_assets": float(1.0 / np.square(weights).sum()),
                "largest_weight": float(weights.max()),
            }
        )

    target_frame = pd.DataFrame(targets)
    target_frame.index = pd.DatetimeIndex(target_frame.index, name="date")
    diagnostics = pd.DataFrame(diagnostic_rows).set_index("date")
    return target_frame, diagnostics


def simulate_target_weights(
    asset_returns: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    name: str,
    evaluation_end: str | None = None,
    diagnostics: pd.DataFrame | None = None,
) -> BacktestResult:
    """Simulate target trades, weight drift, and costs on traded notional.

    Turnover is ``sum(abs(target - pre_trade_weights))`` and is deliberately not
    divided by two. A target decided from data through t-1 earns the return at t.
    """
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction costs cannot be negative")
    asset_returns = _validate_return_frame(asset_returns)
    if not isinstance(targets, pd.DataFrame) or targets.empty:
        raise ValueError("targets must be a non-empty DataFrame")
    if not isinstance(targets.index, pd.DatetimeIndex):
        raise ValueError("targets must use a DatetimeIndex")
    if targets.index.has_duplicates or not targets.index.is_monotonic_increasing:
        raise ValueError("target dates must be unique and increasing")
    if not targets.columns.equals(asset_returns.columns):
        raise ValueError("targets must be non-empty and match return columns")
    if not targets.index.isin(asset_returns.index).all():
        raise ValueError("every target date must appear in asset returns")
    if (
        not np.isfinite(targets.to_numpy(dtype=float)).all()
        or (targets.to_numpy() < -1e-12).any()
        or not np.allclose(targets.sum(axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError(
            "each target must be a finite, long-only, fully invested vector"
        )

    end = pd.Timestamp(evaluation_end) if evaluation_end else asset_returns.index.max()
    period_returns = asset_returns.loc[targets.index.min() : end]
    if period_returns.empty or period_returns.isna().any().any():
        raise ValueError("evaluation returns must be complete")

    cost_rate = transaction_cost_bps / 10_000.0
    current = pd.Series(0.0, index=asset_returns.columns)
    net_values: list[float] = []
    gross_values: list[float] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []
    weight_rows: list[pd.Series] = []

    for date, daily_returns in period_returns.iterrows():
        # On a rebalance date, the target is installed before the next close-to-close
        # return is earned. On other dates, the previous weights continue to drift.
        if date in targets.index:
            target = targets.loc[date].astype(float)
            traded = float(np.abs(target - current).sum())
            current = target.copy()
        else:
            traded = 0.0
        cost = traded * cost_rate
        if cost >= 1:
            raise ValueError("transaction cost consumes all portfolio wealth")

        weight_rows.append(current.rename(date))
        gross = float(current @ daily_returns)
        # Multiplicative accounting prevents the cost deduction from being treated as
        # an unrelated additive return and keeps the wealth recursion explicit.
        net = (1.0 + gross) * (1.0 - cost) - 1.0
        denominator = 1.0 + gross
        if denominator <= 0:
            raise ValueError("portfolio wealth became non-positive")
        # Asset-level gains change the end-of-day weights even without a target trade.
        current = current * (1.0 + daily_returns) / denominator

        gross_values.append(gross)
        net_values.append(net)
        turnover_values.append(traded)
        cost_values.append(cost)

    index = period_returns.index
    weight_frame = pd.DataFrame(weight_rows)
    weight_frame.index = index
    weight_frame.index.name = "date"
    return BacktestResult(
        name=name,
        net_returns=pd.Series(net_values, index=index, name=name),
        gross_returns=pd.Series(gross_values, index=index, name=name),
        turnover=pd.Series(turnover_values, index=index, name=name),
        cost_fraction=pd.Series(cost_values, index=index, name=name),
        weights=weight_frame,
        targets=targets,
        diagnostics=diagnostics if diagnostics is not None else pd.DataFrame(),
    )


def run_strategy_suite(
    returns: pd.DataFrame,
    *,
    evaluation_start: str,
    lookback: int,
    transaction_cost_bps: float,
    max_weight: float,
    evaluation_end: str | None = None,
) -> dict[str, BacktestResult]:
    """Run all pre-specified strategies on an identical out-of-sample interval."""
    results: dict[str, BacktestResult] = {}
    for strategy in STRATEGIES:
        targets, diagnostics = build_strategy_targets(
            returns,
            strategy,
            evaluation_start=evaluation_start,
            lookback=lookback,
            max_weight=max_weight,
            evaluation_end=evaluation_end,
        )
        results[strategy] = simulate_target_weights(
            returns,
            targets,
            transaction_cost_bps=transaction_cost_bps,
            name=strategy,
            evaluation_end=evaluation_end,
            diagnostics=diagnostics,
        )

    indexes = {tuple(result.net_returns.index) for result in results.values()}
    if len(indexes) != 1:
        raise AssertionError("strategies do not share one evaluation index")
    return results
