"""Historical VaR forecasts and unconditional-coverage backtesting."""

from __future__ import annotations

from math import log

import numpy as np
import pandas as pd
from scipy.stats import chi2


def rolling_historical_var(
    returns: pd.Series,
    *,
    window: int = 252,
    confidence: float = 0.95,
) -> pd.Series:
    """Forecast positive-loss VaR from the prior window, excluding the current day."""
    if window < 2 or not 0.5 < confidence < 1:
        raise ValueError("invalid rolling VaR configuration")
    if not isinstance(returns, pd.Series) or returns.empty:
        raise ValueError("returns must be a non-empty Series")
    values = returns.to_numpy(dtype=float)
    if returns.isna().any() or not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("returns must be finite, complete, and greater than -1")
    # Shift first so the realised loss at t cannot influence its own VaR forecast.
    shifted_losses = -returns.astype(float).shift(1)
    forecast = shifted_losses.rolling(
        window=window, min_periods=window
    ).quantile(confidence)
    return forecast.clip(lower=0.0).rename("var_forecast")


def kupiec_unconditional_coverage(
    breaches: pd.Series,
    *,
    expected_rate: float,
) -> tuple[float, float]:
    """Return Kupiec's likelihood-ratio statistic and chi-square p-value."""
    if not isinstance(breaches, pd.Series):
        raise ValueError("breaches must be a boolean Series")
    clean = breaches.dropna()
    if not pd.api.types.is_bool_dtype(clean.dtype):
        raise ValueError("breaches must contain boolean values")
    if clean.empty or not 0 < expected_rate < 1:
        raise ValueError(
            "breaches must be non-empty and expected_rate must lie in (0, 1)"
        )
    n_obs = len(clean)
    n_breaches = int(clean.sum())
    observed_rate = n_breaches / n_obs

    def term(count: int, probability: float) -> float:
        # The limit of count * log(probability) is zero when count is zero.
        if count == 0:
            return 0.0
        if probability == 0.0:
            return -np.inf
        return count * log(probability)

    null_log_likelihood = term(n_breaches, expected_rate) + term(
        n_obs - n_breaches, 1.0 - expected_rate
    )
    fitted_log_likelihood = term(n_breaches, observed_rate) + term(
        n_obs - n_breaches, 1.0 - observed_rate
    )
    statistic = max(0.0, float(-2.0 * (null_log_likelihood - fitted_log_likelihood)))
    return statistic, float(chi2.sf(statistic, df=1))


def var_backtest_summary(
    returns_by_strategy: pd.DataFrame,
    *,
    window: int = 252,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarise rolling historical-VaR coverage for each strategy."""
    if not isinstance(returns_by_strategy, pd.DataFrame) or returns_by_strategy.empty:
        raise ValueError("strategy returns must be a non-empty DataFrame")
    if returns_by_strategy.columns.has_duplicates:
        raise ValueError("strategy names must be unique")
    values = returns_by_strategy.to_numpy(dtype=float)
    if (
        returns_by_strategy.isna().any().any()
        or not np.isfinite(values).all()
        or (values <= -1.0).any()
    ):
        raise ValueError("strategy returns must be finite, complete, and greater than -1")
    if len(returns_by_strategy) <= window:
        raise ValueError("strategy returns must be longer than the VaR window")
    rows: list[dict[str, float | int | str]] = []
    expected_rate = 1.0 - confidence
    for strategy in returns_by_strategy.columns:
        returns = returns_by_strategy[strategy]
        forecasts = rolling_historical_var(
            returns, window=window, confidence=confidence
        )
        valid = forecasts.notna()
        losses = -returns.loc[valid]
        # A breach occurs only when the realised positive loss exceeds the forecast.
        breaches = losses > forecasts.loc[valid]
        statistic, p_value = kupiec_unconditional_coverage(
            breaches, expected_rate=expected_rate
        )
        rows.append(
            {
                "strategy": strategy,
                "forecast_observations": int(valid.sum()),
                "breaches": int(breaches.sum()),
                "breach_rate": float(breaches.mean()),
                "expected_breach_rate": expected_rate,
                "kupiec_lr": statistic,
                "kupiec_p_value": p_value,
                "mean_loss_on_breach": (
                    float(losses[breaches].mean()) if breaches.any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")
