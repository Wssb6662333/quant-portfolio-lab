import numpy as np
import pandas as pd
import pytest

from quant_portfolio.backtest import (
    build_strategy_targets,
    simulate_target_weights,
)


def test_target_training_dates_are_strictly_before_rebalance() -> None:
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(300, 3)),
        index=dates,
        columns=["SPY", "TLT", "GLD"],
    )
    _, diagnostics = build_strategy_targets(
        returns,
        "sample_min_variance",
        evaluation_start=str(dates[260].date()),
        lookback=252,
        max_weight=0.8,
    )

    assert (diagnostics["training_end"] < diagnostics.index).all()
    assert (diagnostics["training_observations"] == 252).all()


def test_weight_drift_turnover_and_cost_accounting() -> None:
    dates = pd.date_range("2020-01-01", periods=2)
    returns = pd.DataFrame(
        {"A": [0.10, 0.0], "B": [0.0, 0.0]}, index=dates
    )
    targets = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]], index=dates, columns=returns.columns
    )
    result = simulate_target_weights(
        returns,
        targets,
        transaction_cost_bps=100,
        name="toy",
    )

    expected_drift = np.array([0.55 / 1.05, 0.5 / 1.05])
    expected_second_turnover = np.abs(np.array([0.5, 0.5]) - expected_drift).sum()
    assert np.isclose(result.turnover.iloc[0], 1.0)
    assert np.isclose(result.turnover.iloc[1], expected_second_turnover)
    assert np.isclose(result.net_returns.iloc[0], (1.05 * 0.99) - 1)
    assert np.isclose(result.cost_fraction.iloc[1], expected_second_turnover * 0.01)


def test_backtest_rejects_non_finite_asset_returns() -> None:
    dates = pd.date_range("2020-01-01", periods=2)
    returns = pd.DataFrame({"A": [0.01, np.inf], "B": [0.0, 0.0]}, index=dates)
    targets = pd.DataFrame([[0.5, 0.5]], index=dates[:1], columns=returns.columns)

    with pytest.raises(ValueError, match="finite"):
        simulate_target_weights(
            returns,
            targets,
            transaction_cost_bps=0,
            name="toy",
        )
