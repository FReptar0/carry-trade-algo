# Algo Validation Audit

**Date**: 2026-03-06 (Protocol Day 33/34)
**System**: Carry Trade V3 (JPY pairs, hourly, OANDA practice)
**Account**: $100,079 | 1 open position (USD/JPY)

This document audits the validation and robustness testing of the carry trade algorithm against industry best practices for algorithmic trading systems. Each item is assessed as DONE, PARTIAL, or MISSING with priority ratings for gaps.

---

## 1. Summary

| Category | Done | Partial | Missing | Total |
|----------|------|---------|---------|-------|
| Backtesting Quality | 4 | 0 | 0 | 4 |
| Robustness Testing | 1 | 3 | 3 | 7 |
| Risk Management | 3 | 1 | 1 | 5 |
| Carry-Trade Specific | 2 | 2 | 1 | 5 |
| Operational | 2 | 0 | 0 | 2 |
| **Total** | **12** | **6** | **5** | **23** |

---

## 2. Done (12 items)

### 2.1 Walk-Forward / Out-of-Sample Testing
- **File**: `src/optimization/walk_forward.py`
- **Implementation**: `WalkForwardAnalyzer` with 180-day train / 30-day test / 30-day step windows
- **What it does**: Slides through historical data, optimizes parameters on train window, validates on test window. Reports parameter stability across windows
- **Industry standard**: Walk-forward is considered the "gold standard" for trading strategy validation (Robert Pardo, 1992)

### 2.2 Monte Carlo VaR
- **File**: `src/risk/risk_metrics.py:289-349`
- **Implementation**: `monte_carlo_var()` resamples historical returns across 10,000 simulated paths
- **What it does**: Answers "How much could I lose on a bad day?" using simulation rather than distributional assumptions
- **Note**: This is VaR estimation, not strategy equity curve simulation (see Missing #3.1)

### 2.3 Historical Stress Testing
- **File**: `scripts/run_stress_test.py` (769 lines)
- **Implementation**: 5 historical crisis scenarios backtested:
  1. BOJ Negative Rates + Brexit (2016)
  2. Flash Crash + Trade War (2019)
  3. COVID Crash + Recovery (2020)
  4. Fed Hikes + BOJ Intervention (2022)
  5. Baseline 2024
- **What it does**: Replays strategy against real crisis data, extracts drawdown events, grades each scenario PASS/CAUTION/FAIL

### 2.4 Slippage Modeling
- **File**: `src/broker/simulator.py:293-304`
- **Implementation**: 0.5-2.0 pip random slippage on all orders, always working against the trader
- **Backtest engine**: `slippage_pct=0.0001` applied via Portfolio class

### 2.5 Commission Modeling
- **File**: `src/broker/simulator.py:306-308`
- **Implementation**: $7/lot commission deducted from balance on each trade

### 2.6 Correlation Monitoring
- **File**: `src/risk/correlation.py`
- **Implementation**: `CorrelationMonitor` with 60-bar rolling correlation matrix. `portfolio_correlation_factor()` returns 0.5-1.0 multiplier for position sizing. Hard cap at 0.85 blocks new entries
- **Connected to**: `DynamicSizer` for live sizing, `runner.py` for entry gate

### 2.7 Execution Quality Validation
- **File**: `src/validation/validator.py:107-345`
- **Implementation**: `PaperTradingValidator` compares backtest vs paper trading on:
  - Fill rate (min 95%)
  - Average slippage
  - Signal-to-fill latency
  - Return degradation (max 30%)
  - Sharpe degradation (max 25%)
  - Drawdown increase (max 20%)

### 2.8 BOJ Intervention Scenario
- **Files**: `scripts/run_stress_test.py:81-86` + `src/news/boj_calendar.py`
- **Implementation**: Explicit stress test with 2022 USD/JPY rally 115->152 followed by BOJ intervention and drop to 127. Plus live BOJ meeting blackout calendar (+-24h around all 8 2026 decision dates)

### 2.9 Paper/Forward Test (34 days)
- **Implementation**: Live OANDA practice account, 33 days completed (day 34 = tomorrow, market closed)
- **Components validated live**: OandaBroker, RegimeDetector, DynamicSizer, CorrelationMonitor, CircuitBreaker, SignalBandit, ExitManager, Reconciler, Watchdog, PerformanceMonitor, AlertManager, ScaleManager

### 2.10 Circuit Breakers / Kill Switches
- **File**: `src/risk/circuit_breakers.py`
- **Implementation**: Daily loss 3%, weekly loss 7%, max drawdown 20%, max positions 6, per-pair exposure limits
- **Behavior**: Halts ALL trading when any limit is breached

### 2.11 Data Quality Validation
- **File**: `src/data/preprocessor.py`
- **Implementation**: Gap detection, outlier detection, duplicate removal, weekend bar filtering, completeness reporting
- **Live**: Real OANDA bid/ask feed, 300 hourly candles per tick (12.5 days lookback), UTC timestamps throughout

### 2.12 State Persistence & Recovery
- **File**: `src/persistence/store.py`
- **Implementation**: SQLite WAL mode, persists positions/equity/protocol/trades every tick. Full state recovery on container restart. Position reconciliation with broker on startup

---

## 3. Missing (5 items)

### 3.1 Statistical Significance Testing -- PRIORITY: HIGH
- **What**: Permutation test or t-test on strategy returns vs random baseline. Computes p-value to determine if edge is real or luck
- **Why it matters**: Without this, a Sharpe of 0.8 could be indistinguishable from random noise. Industry standard: p < 0.05
- **How to implement**: Shuffle daily returns 1,000x, re-compute Sharpe each time, calculate what percentage of shuffled results beat the actual Sharpe. That percentage is the p-value
- **Reference**: Lopez de Prado, "Advances in Financial Machine Learning" (2018), Chapter 11

### 3.2 Monte Carlo Strategy Simulation -- PRIORITY: HIGH
- **What**: Reshuffle/resample the sequence of completed trades 1,000x to generate 1,000 equity curve paths. Different from Monte Carlo VaR
- **Why it matters**: Shows distribution of possible outcomes: 5th percentile max drawdown, probability of ruin, median return, worst-case scenario. Answers "How bad could it realistically get?"
- **How to implement**: Take all backtest trades. For each simulation: randomly resample N trades with replacement, compute cumulative equity curve. Collect max drawdown, final return, Sharpe from each path. Report percentiles
- **Reference**: KJ Trading Systems Monte Carlo guide; StrategyQuant 5 MC methods

### 3.3 Bootstrap Confidence Intervals -- PRIORITY: HIGH
- **What**: Resample returns with replacement 1,000x, compute Sharpe/win-rate/profit-factor each time, report 95% confidence intervals
- **Why it matters**: "Sharpe = 0.8" is meaningless without uncertainty bounds. Bootstrap gives "Sharpe = 0.8 +/- 0.4 (95% CI)". If the CI includes 0, the edge may not be real
- **How to implement**: `np.random.choice(returns, size=len(returns), replace=True)` x 1,000 iterations, compute metric each time, report 2.5th and 97.5th percentiles

### 3.4 Strategy Decay Detection -- PRIORITY: MEDIUM
- **What**: Rolling 30/60/90-day Sharpe ratio trend analysis. Alert if Sharpe is declining month over month
- **Why it matters**: Markets evolve. A strategy that worked in 2023 may stop working in 2025. Decay detection catches this before it drains the account
- **How to implement**: Compute rolling 60-day Sharpe. If current Sharpe < 50% of 6-month average, send DEGRADED alert. Track in monthly review

### 3.5 Correlation Breakdown Crisis Scenario -- PRIORITY: MEDIUM
- **What**: Stress test where all 6 JPY pairs simultaneously hit 0.95+ correlation (as happens during yen carry trade unwinds)
- **Why it matters**: The correlation hard cap (0.85) blocks new entries, but existing open positions are NOT closed when correlation spikes. During Aug 2024 unwind, all JPY pairs moved in lockstep. 6 open positions = 6x the intended risk
- **How to implement**: Synthetic scenario: inject 0.95+ correlation across all pairs, check portfolio drawdown with max positions open. Consider adding "correlation spike exit" for existing positions

---

## 4. Partial (6 items)

### 4.1 Parameter Sensitivity Analysis -- GAP: No formal perturbation test
- **What exists**: Grid search (`src/optimization/grid_search.py`) + walk-forward parameter stability tracking
- **What's missing**: Formal sensitivity analysis that perturbs each parameter by +-5-10% and verifies the strategy remains profitable. A robust strategy should not break from small parameter changes
- **Priority**: MEDIUM (should-do before scaling >$10k)

### 4.2 Overfitting Detection -- GAP: No degrees-of-freedom ratio
- **What exists**: Walk-forward tracks parameter stability across windows
- **What's missing**: Explicit degrees-of-freedom check: ratio of (free parameters) / (independent trades). Rule of thumb: ratio should be < 1/10. With ~8 tunable params and ~50 backtest trades, ratio = 0.16 which is borderline
- **Priority**: MEDIUM

### 4.3 Max Drawdown Duration -- GAP: No continuous "days underwater" metric
- **What exists**: `run_stress_test.py:378-431` extracts drawdown events and recovery days
- **What's missing**: Explicit tracking of maximum time spent below high-water mark. A 5% drawdown lasting 6 months is psychologically and financially different from a 10% drawdown lasting 2 weeks
- **Priority**: LOW (easy to add to metrics)

### 4.4 Tail Risk Beyond VaR -- GAP: No Extreme Value Theory
- **What exists**: Historical VaR, Parametric VaR, Monte Carlo VaR, CVaR (Expected Shortfall) in `risk_metrics.py`
- **What's missing**: Generalized Pareto Distribution (GPD) fitting to tail losses. EVT models the shape of extreme losses more accurately than VaR/CVaR, which assume relatively well-behaved distributions
- **Priority**: LOW (VaR/CVaR is sufficient for $100k account)

### 4.5 Interest Rate Reversal Scenario -- GAP: No dynamic rate shock test
- **What exists**: Vasicek rate model in `interest_rates.py`, hardcoded swap overrides per stress test era
- **What's missing**: Dynamic scenario: "BOJ raises rates by 50bp overnight" -> swap income flips negative -> strategy must exit all positions. Currently the strategy checks `swap > 0` each tick, but no backtest explicitly simulates a sudden rate reversal
- **Priority**: MEDIUM (this is the existential risk for carry trades)

### 4.6 Regime-Conditional Performance Breakdown -- GAP: No per-regime trade P&L
- **What exists**: Regime detection (4 composite regimes), parameter adaptation per regime, regime distribution reporting
- **What's missing**: Trade-level P&L breakdown by regime. Need to answer: "What is the win rate and Sharpe in RANGE_HIGH_VOL specifically?" If all profits come from TREND_LOW_VOL and the strategy loses in all other regimes, that's a concentration risk
- **Priority**: MEDIUM (important for understanding when the strategy actually makes money)

---

## 5. Statistical Significance Gap (Critical)

The industry standard from Lopez de Prado and institutional quant firms:

| Metric | Minimum | Reliable | Institutional | We Have |
|--------|---------|----------|---------------|---------|
| Trades for inference | 30 | 100+ | 200-500 | ~10-15 live |
| Backtest trades | 100+ | 300+ | 500+ across regimes | ~50-80 |
| Forward test duration | 3 months | 6 months | 12+ months | 34 days |
| Market regimes covered | 2+ | 3+ | Bull/bear/range/crisis | 1 (mostly downtrend) |

**This is the single biggest gap.** 34 days with ~15 trades is insufficient for statistical significance. The backtest on 2 years of hourly data provides ~50-80 trades, which is at the low end of "inference possible" but below "reliable."

### Implications
- Cannot compute meaningful p-value with < 30 trades
- Win rate confidence interval with 15 trades is extremely wide (~+-25%)
- Sharpe ratio with 34 days of data has very high estimation error
- A second forward test of 60-90 days (accumulating 50+ trades) is essential

---

## 6. Carry Trade Specific Risks

Risks unique to yen carry trade strategies that require specific validation:

| Risk | Validated? | Notes |
|------|-----------|-------|
| BOJ rate decision | Yes | boj_calendar.py +-24h blackout, 8 dates in 2026 |
| BOJ intervention | Yes | Stress tested with 2022 data (115->152->127) |
| Yen flash crash | Partial | 2019 flash crash in stress test, but no 2024 Aug unwind |
| Interest rate reversal | Partial | No dynamic "BOJ hikes 50bp" scenario |
| All-JPY correlation spike | Missing | No crisis scenario with 0.95+ cross-pair correlation |
| Carry trade crowding | Missing | No measure of how crowded the trade is globally |
| VIX spike / risk-off | Yes | VIX > 25 blocks all entries (runner.py macro gate) |
| Weekend gap | Yes | Friday 20:00 UTC close (weekend_gap_protection) |
| Swap rate change | Partial | Live fetch + defaults, but no reversal scenario |

---

## 7. Action Plan

### Before Real Money (Must-Do)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 1 | Build Monte Carlo strategy simulation | 2-3 hours | Shows distribution of possible outcomes, worst-case DD |
| 2 | Add bootstrap confidence intervals on Sharpe, win rate, profit factor | 1-2 hours | Quantifies uncertainty on all key metrics |
| 3 | Implement permutation test for statistical significance | 1-2 hours | Proves edge is real vs luck (p-value) |
| 4 | Extend forward test to 60-90 more days | Calendar time | Accumulate 50+ live trades for statistical inference |

### Before Scaling > $10k (Should-Do)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 5 | Parameter sensitivity analysis (+-10% perturbation) | 2-3 hours | Proves strategy isn't brittle to exact param values |
| 6 | Per-regime P&L breakdown from backtest | 1-2 hours | Identifies if profits are concentrated in one regime |
| 7 | Strategy decay detector (rolling Sharpe alert) | 1-2 hours | Early warning system for strategy degradation |
| 8 | Correlation crisis scenario test | 2-3 hours | Validates portfolio behavior when all JPY pairs correlate |
| 9 | Interest rate reversal scenario | 2-3 hours | Tests the existential risk for carry trades |

### Before Full Allocation (Nice-to-Have)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 10 | Extreme Value Theory tail modeling | 3-4 hours | Better fat-tail risk estimation |
| 11 | Market impact model (sqrt rule) | 1-2 hours | Required at $1M+ position sizes |
| 12 | Liquidity risk monitoring | 2-3 hours | Spread widening alerts during illiquid sessions |
| 13 | Max drawdown duration metric | 30 min | Easy add to existing metrics |

---

## 8. References

- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley. Chapter 11: Backtesting
- Pardo, R. (1992). "Design, Testing, and Optimization of Trading Systems." Walk-forward analysis
- [Robustness Testing Guide -- Build Alpha](https://www.buildalpha.com/robustness-testing-guide/)
- [5 Monte Carlo Methods -- StrategyQuant](https://strategyquant.com/blog/new-robustness-tests-on-the-strategyquant-codebase-5-monte-carlo-methods-to-bulletproof-your-trading-strategies/)
- [Minimum Trades for Valid Backtest -- BacktestBase](https://www.backtestbase.com/education/how-many-trades-for-backtest)
- [BIS Bulletin #90: Carry Trade Unwind Aug 2024](https://www.bis.org/publ/bisbull90.pdf)
- [Yen Carry Trade Unwinding Risks -- Discovery Alert](https://discoveryalert.com.au/yen-carry-trade-unwinding-risks-2025/)
