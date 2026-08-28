import numpy as np
import pandas as pd

from quant_portfolio.optimization import minimum_variance_weights


def test_minimum_variance_weights_respect_constraints_and_reduce_risk() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.002, 0.001], [0.002, 0.01, 0.001], [0.001, 0.001, 0.02]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    weights = minimum_variance_weights(covariance, max_weight=0.6)
    equal = np.repeat(1 / 3, 3)

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
    assert (weights <= 0.6 + 1e-8).all()
    assert weights.to_numpy() @ covariance.to_numpy() @ weights.to_numpy() <= (
        equal @ covariance.to_numpy() @ equal
    )
