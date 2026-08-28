import numpy as np
import pandas as pd
import pytest

from quant_portfolio.data import calculate_simple_returns, validate_prices


def test_returns_do_not_fill_missing_prices() -> None:
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0, 133.1],
            "B": [100.0, np.nan, 110.0, 121.0],
        },
        index=pd.date_range("2020-01-01", periods=4),
    )
    returns = calculate_simple_returns(prices)

    assert list(returns.index) == [prices.index[-1]]
    assert np.allclose(returns.iloc[0], [0.10, 0.10])


def test_validate_prices_reorders_columns_and_rejects_gaps() -> None:
    prices = pd.DataFrame(
        {"B": [10.0, 11.0], "A": [20.0, 22.0]},
        index=pd.date_range("2020-01-01", periods=2),
    )
    assert list(validate_prices(prices, ["A", "B"]).columns) == ["A", "B"]

    prices.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        validate_prices(prices)


def test_returns_reject_negative_observation_even_when_other_asset_is_missing() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, -1.0, 101.0], "B": [100.0, np.nan, 101.0]},
        index=pd.date_range("2020-01-01", periods=3),
    )
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_simple_returns(prices)
