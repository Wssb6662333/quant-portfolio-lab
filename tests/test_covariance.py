import numpy as np
import pandas as pd

from quant_portfolio.covariance import ledoit_wolf_covariance, sample_covariance


def test_ledoit_wolf_regularises_a_singular_sample_covariance() -> None:
    base = np.linspace(-0.02, 0.02, 20)
    returns = pd.DataFrame({"A": base, "B": 2 * base, "C": -base})

    sample = sample_covariance(returns)
    shrinkage = ledoit_wolf_covariance(returns)

    assert shrinkage.minimum_eigenvalue > 0
    assert shrinkage.condition_number < sample.condition_number
    assert 0 <= shrinkage.shrinkage <= 1
    assert shrinkage.matrix.index.equals(returns.columns)
