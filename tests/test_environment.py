"""Minimal package-layout and deterministic fixture contracts."""

from pathlib import Path

import pandas as pd

import quant_portfolio


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_from_src_layout() -> None:
    package_path = Path(quant_portfolio.__file__).resolve()

    assert package_path.parent.name == "quant_portfolio"
    assert package_path.parent.parent.name == "src"


def test_toy_prices_fixture_contract() -> None:
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "toy_prices.csv"
    prices = pd.read_csv(fixture_path, index_col="date", parse_dates=True)

    assert prices.shape == (5, 3)
    assert list(prices.columns) == ["SPY", "TLT", "GLD"]
    assert prices.index.is_monotonic_increasing
    assert prices["TLT"].isna().sum() == 1
