# Covariance Shrinkage in Cross-Asset ETF Portfolios

I am studying whether Ledoit-Wolf covariance shrinkage improves out-of-sample risk
and weight stability relative to sample-covariance minimum variance in a cross-asset
ETF portfolio.

## Current public checkpoint

The repository now contains a reproducible experiment pipeline for comparing:

- buy-and-hold and quarterly equal-weight portfolios;
- inverse-volatility allocation;
- sample-covariance minimum variance;
- Ledoit-Wolf minimum variance; and
- SPY as a single-asset benchmark.

The pipeline downloads or loads validated adjusted-close data for a fixed nine-ETF
universe, generates causal quarterly targets, models daily weight drift and
transaction costs, and writes performance, covariance, regime, sensitivity, and VaR
outputs. The generated result files, figures, executed notebook, and quantitative
conclusions will be published only after the final artifact review.

## Research design

For a target effective on date `t`, each estimator receives only the previous
`lookback` returns, ending at `t-1`. The target then earns the close-to-close return
indexed by `t`. I compare 126-, 252-, and 504-day estimation windows under 0, 10,
and 25 basis-point transaction-cost assumptions.

The two minimum-variance portfolios use long-only, fully invested SciPy SLSQP
optimisation with a maximum 40% allocation per asset. I define turnover as
`sum(abs(target - pre_trade_weights))`, include the initial trade, and allow weights
to drift between rebalance dates.

The full methodology is documented in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Run the experiment

The commands below use Python 3.12 from Windows Git Bash.

```bash
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe scripts/run_experiment.py
```

Use `--refresh-data` when a new vendor download is required. Otherwise the script
reuses the local cache when available and downloads the fixed period when it is not.

## Implementation boundary

I use scikit-learn's Ledoit-Wolf estimator and SciPy's SLSQP solver rather than
presenting either library method as my own. My code handles the data contracts,
portfolio constraints, causal target generation, daily accounting, risk tests,
experiment grid, and reporting around those library methods.
