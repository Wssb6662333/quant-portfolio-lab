"""Long-only allocation rules used in the walk-forward study."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def equal_weights(columns: pd.Index | list[str] | tuple[str, ...]) -> pd.Series:
    labels = pd.Index(columns)
    if len(labels) == 0:
        raise ValueError("at least one asset is required")
    return pd.Series(1.0 / len(labels), index=labels, dtype=float)


def inverse_volatility_weights(returns: pd.DataFrame) -> pd.Series:
    """Allocate in inverse proportion to sample daily volatility."""
    vol = returns.std(ddof=1)
    if vol.isna().any() or (vol <= 0).any() or not np.isfinite(vol.to_numpy()).all():
        raise ValueError("each asset must have positive finite volatility")
    inverse = 1.0 / vol
    return inverse / inverse.sum()


def _validate_covariance(covariance: pd.DataFrame) -> np.ndarray:
    if not isinstance(covariance, pd.DataFrame) or covariance.empty:
        raise ValueError("covariance must be a non-empty labelled DataFrame")
    if not covariance.index.equals(covariance.columns):
        raise ValueError("covariance row and column labels must match")
    matrix = covariance.to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("covariance must be finite and symmetric")
    if np.linalg.eigvalsh(matrix).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    return matrix


def minimum_variance_weights(
    covariance: pd.DataFrame,
    *,
    max_weight: float = 1.0,
) -> pd.Series:
    """Solve a fully invested, long-only minimum-variance problem with SLSQP."""
    matrix = _validate_covariance(covariance)
    n_assets = len(covariance)
    if not 0 < max_weight <= 1 or max_weight * n_assets < 1 - 1e-12:
        raise ValueError("max_weight is outside (0, 1] or makes the problem infeasible")

    # Equal weights provide a feasible starting point under the validated cap.
    initial = np.repeat(1.0 / n_assets, n_assets)
    result = minimize(
        # Portfolio variance is w'Σw; supplying its analytic gradient improves
        # convergence and avoids finite-difference noise near active weight bounds.
        fun=lambda weights: float(weights @ matrix @ weights),
        x0=initial,
        jac=lambda weights: 2.0 * matrix @ weights,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n_assets,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"minimum-variance optimisation failed: {result.message}")

    weights = np.asarray(result.x, dtype=float)
    if (
        not np.isfinite(weights).all()
        or not np.isclose(weights.sum(), 1.0, atol=1e-7)
        or weights.min() < -1e-8
        or weights.max() > max_weight + 1e-8
    ):
        raise RuntimeError("solver returned weights that violate the portfolio constraints")
    # Clip only solver-scale numerical noise, then restore the fully invested sum.
    weights = np.clip(weights, 0.0, max_weight)
    weights /= weights.sum()
    return pd.Series(weights, index=covariance.columns, dtype=float)
