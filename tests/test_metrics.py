import numpy as np
import pandas as pd
import pytest

from quant_portfolio.metrics import (
    historical_var_cvar,
    maximum_drawdown,
    performance_statistics,
    wealth_index,
)


def test_drawdown_includes_initial_wealth() -> None:
    returns = pd.Series([-0.10, 0.05])
    assert np.isclose(maximum_drawdown(returns), -0.10)


def test_var_cvar_use_positive_loss_convention() -> None:
    returns = pd.Series([-0.10, -0.04, 0.01, 0.02, 0.03])
    var, cvar = historical_var_cvar(returns, confidence=0.8)
    assert np.isclose(var, 0.052)
    assert np.isclose(cvar, 0.10)


def test_wealth_rejects_total_loss() -> None:
    with pytest.raises(ValueError, match="greater than -1"):
        wealth_index(pd.Series([0.01, -1.0]))


def test_performance_statistics_use_consistent_annualisation() -> None:
    returns = pd.Series(
        [0.01, -0.005, 0.002],
        index=pd.date_range("2020-01-01", periods=3),
    )
    result = performance_statistics(returns, annualization=252, confidence=0.95)

    expected_volatility = returns.std(ddof=1) * np.sqrt(252)
    expected_sharpe = returns.mean() * 252 / expected_volatility
    assert np.isclose(result["annual_volatility"], expected_volatility)
    assert np.isclose(result["sharpe"], expected_sharpe)
    assert "historical_var_95" in result
