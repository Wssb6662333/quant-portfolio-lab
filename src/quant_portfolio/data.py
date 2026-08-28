"""Market-data download, cache, validation, and return calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf


def _normalise_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    index = pd.DatetimeIndex(result.index)
    # Daily bars are compared by calendar date, so timezone metadata must not create
    # false mismatches between cached data and newly downloaded data.
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    result.index.name = "date"
    return result.sort_index()


def validate_prices(
    prices: pd.DataFrame,
    tickers: Sequence[str] | None = None,
    *,
    min_observations: int = 2,
) -> pd.DataFrame:
    """Validate a complete adjusted-close matrix and return a defensive copy."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame")

    clean = _normalise_index(prices).astype(float)
    if clean.index.has_duplicates or not clean.index.is_monotonic_increasing:
        raise ValueError("price dates must be unique and increasing")
    if clean.columns.has_duplicates:
        raise ValueError("ticker columns must be unique")
    if tickers is not None:
        missing = [ticker for ticker in tickers if ticker not in clean.columns]
        if missing:
            raise ValueError(f"missing ticker columns: {missing}")
        clean = clean.loc[:, list(tickers)]
    if len(clean) < min_observations:
        raise ValueError(f"prices require at least {min_observations} observations")
    if clean.isna().any().any():
        counts = clean.isna().sum()
        raise ValueError(f"prices contain missing values: {counts[counts > 0].to_dict()}")
    values = clean.to_numpy()
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("prices must be finite and strictly positive")
    return clean


def download_adjusted_close(
    tickers: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download auto-adjusted daily closes from Yahoo Finance."""
    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no observations")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return validate_prices(prices, tickers)


def load_or_download_prices(
    cache_path: Path,
    tickers: Sequence[str],
    start: str,
    end: str,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Use a deterministic local CSV cache, downloading only when necessary."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, index_col="date", parse_dates=True)
        return validate_prices(cached, tickers)

    prices = download_adjusted_close(tickers, start, end)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path, index_label="date")
    return prices


def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate close-to-close simple returns without silently filling gaps."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame")
    clean = _normalise_index(prices).astype(float)
    values = clean.to_numpy()
    finite_or_missing = np.isfinite(values) | np.isnan(values)
    observed = values[~np.isnan(values)]
    if not finite_or_missing.all() or (observed <= 0).any():
        raise ValueError("observed prices must be finite and strictly positive")

    # Explicitly disable pandas' historical forward-fill behaviour. A missing price
    # must remove that cross-sectional observation instead of creating a false 0% return.
    returns = clean.pct_change(fill_method=None).dropna(how="any")
    if returns.empty or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("price matrix does not produce complete finite returns")
    returns.index.name = "date"
    return returns
