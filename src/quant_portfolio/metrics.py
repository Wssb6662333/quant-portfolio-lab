"""Performance and positive-loss tail-risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_portfolio.backtest import BacktestResult


def _clean_return_series(returns: pd.Series, *, min_observations: int = 1) -> pd.Series:
    if not isinstance(returns, pd.Series) or len(returns) < min_observations:
        raise ValueError(f"returns require at least {min_observations} observations")
    clean = returns.astype(float)
    values = clean.to_numpy()
    if clean.isna().any() or not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("returns must be finite, complete, and greater than -1")
    return clean


def wealth_index(returns: pd.Series) -> pd.Series:
    """Return cumulative wealth from a starting value of one."""
    clean = _clean_return_series(returns)
    return (1.0 + clean).cumprod()


def maximum_drawdown(returns: pd.Series) -> float:
    """Return the worst drawdown, including initial wealth 1 in the peak path."""
    # Prepending initial capital ensures that a loss on the first day is measurable.
    wealth = np.concatenate(([1.0], wealth_index(returns).to_numpy()))
    drawdowns = wealth / np.maximum.accumulate(wealth) - 1.0
    return float(drawdowns.min())


def historical_var_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return non-negative VaR and CVaR under a positive-loss convention."""
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must lie in (0.5, 1)")
    clean = _clean_return_series(returns)
    # VaR and CVaR are stored as positive loss magnitudes throughout the project.
    losses = -clean
    var = max(0.0, float(losses.quantile(confidence)))
    tail = losses[losses >= var]
    cvar = max(var, float(tail.mean())) if not tail.empty else var
    return var, cvar


def performance_statistics(
    returns: pd.Series,
    *,
    annualization: int = 252,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Calculate annualised performance and full-sample tail-risk statistics."""
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    clean = _clean_return_series(returns, min_observations=2)
    if not isinstance(clean.index, pd.DatetimeIndex):
        raise ValueError("performance returns must use a DatetimeIndex")
    observations = len(clean)
    final_wealth = float(wealth_index(clean).iloc[-1])
    cagr = final_wealth ** (annualization / observations) - 1.0
    volatility = float(clean.std(ddof=1) * np.sqrt(annualization))
    # Sharpe uses the annualised arithmetic mean, not CAGR, with a zero risk-free rate.
    annual_mean = float(clean.mean() * annualization)
    sharpe = annual_mean / volatility if volatility > 0 else np.nan
    var, cvar = historical_var_cvar(clean, confidence)
    confidence_label = f"{confidence:.0%}".replace("%", "")
    return {
        "start": clean.index.min().date().isoformat(),
        "end": clean.index.max().date().isoformat(),
        "observations": observations,
        "cagr": cagr,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": maximum_drawdown(clean),
        f"historical_var_{confidence_label}": var,
        f"historical_cvar_{confidence_label}": cvar,
        "final_wealth": final_wealth,
    }


def summarise_backtests(
    results: dict[str, BacktestResult],
    *,
    annualization: int = 252,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarise a set of backtests on their common evaluation interval."""
    if not results:
        raise ValueError("at least one backtest result is required")
    rows: list[dict[str, float | int | str]] = []
    for name, result in results.items():
        row = performance_statistics(
            result.net_returns,
            annualization=annualization,
            confidence=confidence,
        )
        years = len(result.net_returns) / annualization
        row.update(
            {
                "strategy": name,
                "total_turnover": float(result.turnover.sum()),
                "annualized_turnover": float(result.turnover.sum() / years),
                "total_cost_fraction": float(result.cost_fraction.sum()),
                "rebalance_count": int((result.turnover > 0).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")
