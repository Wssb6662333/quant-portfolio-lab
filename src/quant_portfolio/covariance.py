"""Covariance estimators and numerical diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class CovarianceEstimate:
    """A labelled covariance matrix plus estimator diagnostics."""

    matrix: pd.DataFrame
    condition_number: float
    minimum_eigenvalue: float
    shrinkage: float | None


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.shape[0] < 2:
        raise ValueError("returns must be a DataFrame with at least two rows")
    clean = returns.astype(float)
    if clean.columns.has_duplicates or clean.isna().any().any():
        raise ValueError("returns require unique columns and no missing values")
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("returns must be finite")
    return clean


def _package_estimate(
    matrix: np.ndarray,
    columns: pd.Index,
    shrinkage: float | None,
) -> CovarianceEstimate:
    raw = np.asarray(matrix, dtype=float)
    # Remove floating-point asymmetry before the PSD check and optimisation step.
    symmetric = (raw + raw.T) / 2
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if not np.isfinite(symmetric).all() or eigenvalues.min() < -1e-12:
        raise ValueError("covariance estimate must be finite and positive semidefinite")
    labelled = pd.DataFrame(symmetric, index=columns, columns=columns)
    # The condition number measures numerical sensitivity; it is a diagnostic rather
    # than a portfolio-performance metric.
    return CovarianceEstimate(
        matrix=labelled,
        condition_number=float(np.linalg.cond(symmetric)),
        minimum_eigenvalue=float(eigenvalues.min()),
        shrinkage=shrinkage,
    )


def sample_covariance(returns: pd.DataFrame) -> CovarianceEstimate:
    """Estimate the conventional sample covariance matrix."""
    clean = _validate_returns(returns)
    return _package_estimate(clean.cov().to_numpy(), clean.columns, None)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> CovarianceEstimate:
    """Estimate linear shrinkage covariance with scikit-learn's LedoitWolf."""
    clean = _validate_returns(returns)
    estimator = LedoitWolf(assume_centered=False).fit(clean.to_numpy())
    return _package_estimate(
        estimator.covariance_,
        clean.columns,
        float(estimator.shrinkage_),
    )
