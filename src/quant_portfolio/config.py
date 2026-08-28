"""Experiment configuration and the pre-specified ETF universe."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_TICKERS = (
    "SPY",  # US equities
    "EFA",  # developed-market equities ex US
    "EEM",  # emerging-market equities
    "TLT",  # long-duration US Treasuries
    "TIP",  # US inflation-linked Treasuries
    "LQD",  # investment-grade US corporate bonds
    "GLD",  # gold
    "DBC",  # broad commodities
    "VNQ",  # US real estate
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Pre-specified choices used by the main and robustness experiments."""

    tickers: tuple[str, ...] = DEFAULT_TICKERS
    data_start: str = "2007-01-01"
    data_end: str = "2026-01-01"  # yfinance end dates are exclusive
    evaluation_start: str = "2013-01-02"
    lookback: int = 252
    transaction_cost_bps: float = 10.0
    max_weight: float = 0.40
    annualization: int = 252
    var_confidence: float = 0.95
    var_window: int = 252
    sensitivity_lookbacks: tuple[int, ...] = (126, 252, 504)
    sensitivity_costs_bps: tuple[float, ...] = (0.0, 10.0, 25.0)

    def __post_init__(self) -> None:
        if len(self.tickers) < 2 or len(set(self.tickers)) != len(self.tickers):
            raise ValueError("tickers must contain at least two unique symbols")
        if self.lookback < 2 or self.var_window < 2:
            raise ValueError("lookback windows must contain at least two observations")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must lie in (0, 1]")
        if self.max_weight * len(self.tickers) < 1:
            raise ValueError("max_weight is infeasible for the number of assets")
        if not 0.5 < self.var_confidence < 1:
            raise ValueError("var_confidence must lie in (0.5, 1)")
