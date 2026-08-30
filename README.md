# Covariance Shrinkage in Cross-Asset ETF Portfolios

I am studying whether Ledoit-Wolf covariance shrinkage improves out-of-sample risk
and weight stability relative to sample-covariance minimum variance in a cross-asset
ETF portfolio.

## Current public checkpoint

At this checkpoint, the repository includes:

- a fixed nine-ETF universe and pre-specified experiment configuration;
- adjusted-close download, caching, validation, and gap-safe return calculations;
- sample and scikit-learn Ledoit-Wolf covariance estimates with numerical diagnostics;
- long-only, fully invested SLSQP minimum-variance allocation with a 40% asset cap;
- quarterly targets estimated only from returns available before each effective date;
- daily weight drift, turnover, initial-trade costs, and net portfolio accounting;
- CAGR, volatility, Sharpe ratio, maximum drawdown, and historical 95% VaR/CVaR;
- rolling historical VaR forecasts that exclude the current return, with Kupiec
  unconditional-coverage tests;
- deterministic tests and a GitHub Actions workflow.

The experiment pipeline, result tables, figures, executed research notebook, and
quantitative conclusions will be added only after their own review checkpoints.

## Timing and accounting

For a target effective on date `t`, the estimator receives only the previous
`lookback` returns, ending at `t-1`. The target then earns the close-to-close return
indexed by `t`. Between rebalance dates, asset returns change the realised portfolio
weights before the next trade.

I define turnover as `sum(abs(target - pre_trade_weights))` and do not divide it by
two. A fully invested strategy therefore records turnover 1.0 for its initial trade.
Transaction costs are charged in basis points on traded notional.

## Check the current code

The commands below use Python 3.12 from Windows Git Bash.

```bash
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
./.venv/Scripts/python.exe -m pytest -q
```

## Implementation boundary

I use scikit-learn's Ledoit-Wolf estimator and SciPy's SLSQP solver rather than
presenting either library method as my own. My code handles the data contracts,
portfolio constraints, causal target generation, daily accounting, metrics, and
risk-model validation around those library methods.
