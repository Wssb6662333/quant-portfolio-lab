import numpy as np
import pandas as pd
import pytest

from quant_portfolio.risk import (
    kupiec_unconditional_coverage,
    rolling_historical_var,
)


def test_rolling_var_excludes_the_current_return() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, -0.50])
    forecast = rolling_historical_var(returns, window=3, confidence=0.95)

    assert np.isclose(forecast.iloc[-1], 0.0)
    assert -returns.iloc[-1] > forecast.iloc[-1]


def test_kupiec_returns_finite_probability() -> None:
    breaches = pd.Series([False] * 95 + [True] * 5)
    statistic, p_value = kupiec_unconditional_coverage(
        breaches, expected_rate=0.05
    )
    assert np.isclose(statistic, 0.0)
    assert np.isclose(p_value, 1.0)


def test_kupiec_handles_zero_breaches() -> None:
    statistic, p_value = kupiec_unconditional_coverage(
        pd.Series([False] * 100), expected_rate=0.05
    )
    assert np.isfinite(statistic)
    assert statistic > 0
    assert 0 <= p_value <= 1


def test_rolling_var_rejects_missing_returns() -> None:
    returns = pd.Series([0.01, np.nan, -0.02])

    with pytest.raises(ValueError, match="complete"):
        rolling_historical_var(returns, window=2)
