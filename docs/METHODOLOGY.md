# Methodology

## Pre-specified design

I pre-specify a 252-trading-day estimation window, quarterly rebalancing, a 40%
single-asset cap, and 10 bp costs on traded notional. I fix the test interval at
2013-01-02 through 2025-12-31. For robustness, I cross three lookbacks (126, 252,
504) with three costs (0, 10, 25 bp) on the same evaluation dates. I do not select
these values after observing performance.

## Information timeline

For an effective rebalance date `t`, I pass `returns.iloc[t-L:t]` to the estimator.
The last training observation is therefore `t-1`, and the target first earns the
close-to-close return indexed by `t`. `build_strategy_targets` records
`training_start`, `training_end`, and `training_observations` for every decision;
I use tests to assert that `training_end` is earlier than the decision date.

I interpret this as computing the target after the prior close and trading before the
next measured holding-period return. I do not model auction mechanics or intraday
slippage.

## Portfolio construction

For covariance estimate `S`, SciPy SLSQP solves

```text
minimise       w' S w
subject to     sum(w) = 1
               0 <= w_i <= 0.40
```

I use pandas' sample covariance as the conventional estimator. I compare it with
scikit-learn's `LedoitWolf`, which shrinks the empirical covariance toward a scaled
identity target. I independently check solver success, labels, finiteness, positive
semidefiniteness, weight sum, and bounds.

## Daily accounting

Immediately before a target trade, current weights are the prior weights after market
drift. I define

```text
turnover_t = sum_i |target_i,t - pre_trade_weight_i,t|
cost_t     = turnover_t * cost_bps / 10,000
net_t      = (1 + gross_t) * (1 - cost_t) - 1
```

I include the initial purchase, so a fully invested strategy starts with turnover 1.0.
I do not divide turnover by two. After the asset returns are realised, I drift weights
as `w_i(1+r_i)/(1+r_portfolio)`. I do not allow a target to affect a prior date.

## Metrics and risk

I calculate CAGR with 252 observations per year. I annualise volatility as sample
standard deviation times `sqrt(252)` and set the Sharpe risk-free rate explicitly to
zero. I prepend initial wealth 1 when calculating maximum drawdown so a first-day loss
cannot disappear.

I report historical VaR and CVaR as non-negative losses. At date `t`, I calculate
rolling VaR from the prior 252 strategy returns via `shift(1)`, never the realised
return at `t`. I evaluate coverage with Kupiec's unconditional likelihood-ratio test.
I report CVaR descriptively and do not assign it a formal coverage interpretation.

## Data and comparison boundaries

I evaluate all strategies on identical out-of-sample dates and adjusted-close returns.
I use SPY both as an investable member of the cross-asset universe and as a
single-asset context benchmark. Its strong CAGR is not evidence that the lower-risk
portfolios failed their stated objective; I separate return level from risk reduction.

My fixed ETF universe creates selection and survivorship assumptions. Yahoo Finance
adjustments, data revisions, and the absence of intraday execution are additional
limitations. I report descriptive differences and do not estimate statistical
uncertainty around strategy-performance gaps. The Kupiec test covers unconditional
breach frequency, not breach independence.
