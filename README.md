# Covariance Shrinkage in Cross-Asset ETF Portfolios

I am building a reproducible study of whether Ledoit-Wolf covariance shrinkage
improves out-of-sample risk and weight stability relative to sample-covariance
minimum variance in a cross-asset ETF portfolio.

## Current public checkpoint

At this reviewed checkpoint, I have added:

- a fixed nine-ETF research universe and pre-specified experiment configuration;
- adjusted-close download, caching, validation, and gap-safe return calculations;
- labelled sample and scikit-learn Ledoit-Wolf covariance estimates;
- positive-semidefinite and condition-number diagnostics;
- long-only, fully invested SLSQP minimum-variance allocation with a 40% asset cap;
- deterministic tests and a GitHub Actions workflow.

I will add the causal walk-forward backtest, transaction-cost accounting, robustness
analysis, verified result tables, figures, and final research conclusions in later
reviewed commits.

## Current tests

```bash
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
./.venv/Scripts/python.exe -m pytest -q
```

## Implementation boundary

I use scikit-learn's Ledoit-Wolf estimator and SciPy's SLSQP solver rather than
presenting either library method as my own. I implement the surrounding data
contracts, estimator diagnostics, portfolio constraints, and correctness tests.
