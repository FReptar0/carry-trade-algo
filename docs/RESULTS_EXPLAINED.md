# Carry Trade Algorithm - Complete Results Explanation

## Table of Contents
1. [What We Built](#what-we-built)
2. [Glossary of Terms](#glossary-of-terms)
3. [Results by Phase](#results-by-phase)
4. [Key Findings](#key-findings)
5. [Algorithm Validation](#algorithm-validation)
6. [Next Steps for Real Money Testing](#next-steps-for-real-money-testing)

---

## What We Built

A **carry trade** algorithm that:
1. Borrows in low-interest currencies (JPY at ~0%)
2. Invests in high-interest currencies (USD at ~5%)
3. Earns the interest rate differential daily ("swap")
4. Uses technical filters to time entries/exits

**Think of it like this**: You borrow money from a Japanese bank at 0% interest, convert it to USD, and put it in a US bank earning 5%. You pocket the 5% difference. The risk? If the exchange rate moves against you, you could lose more than the interest you earned.

---

## Glossary of Terms

### Trading Terms

| Term | Meaning | Example |
|------|---------|---------|
| **Swap** | Daily interest payment for holding a position overnight | Long USD/JPY earns ~$1.37/day per lot |
| **Pip** | Smallest price movement (0.01 for JPY pairs, 0.0001 for others) | USD/JPY moves from 150.00 to 150.01 = 1 pip |
| **Lot** | Standard trading unit = 100,000 currency units | 0.1 lot = 10,000 units |
| **Long** | Buying a currency pair (betting it goes up) | Long USD/JPY = buy USD, sell JPY |
| **Short** | Selling a currency pair (betting it goes down) | Short USD/JPY = sell USD, buy JPY |
| **Spread** | Difference between buy and sell price (broker's fee) | Bid 150.00, Ask 150.02 = 2 pip spread |
| **Slippage** | Difference between expected and actual execution price | You wanted 150.00, got filled at 150.02 |
| **Margin** | Collateral required to hold a position | 50:1 leverage means $2,000 margin for $100,000 position |
| **Leverage** | Borrowed money to amplify positions | 50:1 = control $50 for every $1 you have |

### Performance Metrics

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **Total Return** | How much money you made/lost as % | Positive is good |
| **Sharpe Ratio** | Return per unit of risk (risk-adjusted return) | > 1.0 is good, > 2.0 is excellent |
| **Sortino Ratio** | Like Sharpe but only penalizes downside risk | > 1.0 is good |
| **Max Drawdown** | Largest peak-to-trough decline | < 20% is acceptable |
| **Win Rate** | % of trades that were profitable | > 50% with good risk/reward |
| **Profit Factor** | Gross profit ÷ Gross loss | > 1.5 is good, > 2.0 is excellent |
| **Calmar Ratio** | Annual return ÷ Max drawdown | > 1.0 is good |

### Risk Terms

| Term | Meaning | Example |
|------|---------|---------|
| **VaR (Value at Risk)** | Maximum expected loss at X% confidence | VaR 95% = $1,000 means 95% of days you lose less than $1,000 |
| **CVaR (Conditional VaR)** | Average loss when you exceed VaR | When you have a bad day (worst 5%), you lose $1,500 on average |
| **ATR (Average True Range)** | Measure of daily price volatility | ATR of 2.0 means price typically moves 2 units per day |
| **Drawdown** | Current decline from peak equity | If peak was $110k, now $100k = 9.1% drawdown |
| **Kelly Criterion** | Mathematically optimal bet size | Tells you to risk X% per trade for maximum growth |

### Technical Indicators

| Indicator | Meaning | Usage |
|-----------|---------|-------|
| **SMA (Simple Moving Average)** | Average price over N periods | Price > SMA = uptrend |
| **RSI (Relative Strength Index)** | Momentum indicator (0-100) | > 70 = overbought, < 30 = oversold |
| **Stop Loss** | Automatic exit if price moves against you | 2% stop = exit if you lose 2% |
| **Take Profit** | Automatic exit when target is reached | 6% TP = exit when you gain 6% |
| **Trailing Stop** | Stop that follows price up, locks in profit | Once up 3%, stop trails price by 2% |

### Abbreviations

| Abbrev | Full Term |
|--------|-----------|
| SL | Stop Loss |
| TP | Take Profit |
| DD | Drawdown |
| HWM | High Water Mark (peak equity) |
| PnL | Profit and Loss |
| ROI | Return on Investment |
| AUM | Assets Under Management |
| OHLCV | Open, High, Low, Close, Volume |
| GBM | Geometric Brownian Motion (price simulation model) |
| FF | Fixed Fractional (position sizing method) |

---

## Results by Phase

### Phase 1: Synthetic Data Generation

**What we did**: Created fake but realistic forex price data to test the strategy.

**Results**:
- Generated 6,264 hourly bars per currency pair (about 1 year)
- 3 pairs: USD/JPY, AUD/JPY, EUR/USD
- Simulated interest rates using Vasicek model (rates that mean-revert like real rates)

**Key insight**: Synthetic data is useful for testing code, but it's a "random walk in a box" - prices bounce around randomly. Real markets have trends driven by economics.

---

### Phase 2: Strategy & Backtesting

**What we did**: Built the carry trade strategy and tested it on synthetic data.

**Strategy rules**:
- **Entry**: Open long position when:
  1. Swap income > 0.5% annualized (worth the carry)
  2. Price > 50-day SMA (trend is up)
  3. RSI < 70 (not overbought)

- **Exit**: Close position when:
  1. Stop loss hit (price drops 2-6% from entry)
  2. Take profit hit (price rises 6-8% from entry)
  3. Trailing stop triggered (price reverses after being profitable)

**Results on synthetic data**:

| Pair | Return | Sharpe | Trades | Max DD |
|------|--------|--------|--------|--------|
| USD/JPY | -0.4% | -0.20 | 2 | 1.2% |
| AUD/JPY | -55% | -2.1 | 238 | 58% |
| EUR/USD | 0% | 0 | 0 | 0% |

**Problem discovered**: AUD/JPY had 238 trades with only 12.6% win rate. The 2% stop loss was getting hit constantly ("stop loss whipsaw"). The strategy was losing money on transaction costs and bad exits.

---

### Phase 3: Optimization

**What we did**: Tested 108 different parameter combinations to find what works.

**Key finding**: **Stop loss is the most important parameter**.

| Stop Loss | Return | Improvement |
|-----------|--------|-------------|
| 2% | -55% | Baseline (disaster) |
| 4% | -12% | Better |
| 6% | -0.24% | Much better |
| 8% | +2% | Best |

**Why?** Tight stops (2%) get triggered by normal market noise. Wider stops (6-8%) let the trade "breathe" and don't exit on every small dip.

**Walk-forward analysis** (testing on out-of-sample data):
- Only 1 out of 7 time periods was profitable
- Conclusion: Synthetic data doesn't have real trends, so the strategy can't work properly

---

### Phase 4: Real Data Integration

**What we did**: Downloaded real forex data from Yahoo Finance and tested the strategy.

**Data quality**:
- 1,042 daily bars (2021-2024) for each pair
- 9,655 hourly bars (recent data)
- 100% completeness, 0 duplicates

**Synthetic vs Real comparison**:

| Metric | Synthetic | Real | Meaning |
|--------|-----------|------|---------|
| Kurtosis | 0.5 | 3.6 | Real data has "fat tails" (more extreme moves) |
| Skewness | 0.01 | -0.48 | Real markets crash faster than they rally |
| Price range | Bounded | Trending | USD/JPY went from 104 to 161 (55% move!) |

**Backtest on real hourly data (default parameters)**:

| Pair | Return | Sharpe | Trades | Max DD |
|------|--------|--------|--------|--------|
| USD/JPY | -1.46% | -0.82 | 18 | 2.45% |
| AUD/JPY | -1.92% | -1.05 | 21 | 3.04% |
| EUR/USD | 0% | 0 | 0 | 0% |

**After optimization** (SL=8%, SMA=50, Size=15%):

| Pair | Return | Sharpe | Trades | Max DD |
|------|--------|--------|--------|--------|
| USD/JPY | -0.84% | -0.45 | 6 | 1.8% |
| AUD/JPY | -0.10% | -0.06 | 4 | 0.9% |

**Key insight**: The strategy is roughly break-even on real data. Not great, but not catastrophic. The main issue is that the strategy doesn't capture the big USD/JPY trend (104→161) because it's too conservative.

---

### Phase 5: Risk Management

**What we did**: Built professional risk management tools.

**VaR Results (Value at Risk)**:

| Pair | VaR 95% | CVaR 95% | Meaning |
|------|---------|----------|---------|
| USD/JPY | $939 (0.94%) | $1,433 (1.43%) | 95% of days, you lose < $939. On bad days (5%), you lose ~$1,433 |
| AUD/JPY | $1,113 (1.11%) | $1,507 (1.51%) | More volatile than USD/JPY |
| EUR/USD | $751 (0.75%) | $1,040 (1.04%) | Least volatile |

**Position sizing comparison** (for 45% win rate, 3.5% avg win, 2.5% avg loss):

| Method | Recommended Size | Rationale |
|--------|------------------|-----------|
| Full Kelly | 5.7% | Mathematically optimal but too aggressive |
| Quarter Kelly | 1.4% | Safer, recommended |
| Fixed Fractional (2% risk) | 20% | Simple, consistent |
| Volatility-based | 20% | Adapts to market conditions |

**Circuit breaker limits**:
- Daily loss: Stop at 3%
- Weekly loss: Stop at 7%
- Max drawdown: Stop at 20%
- Max positions: 3 simultaneous

---

### Phase 6: Paper Trading

**What we did**: Simulated live trading with realistic broker behavior.

**Simulation results** (500 hourly bars):
- Initial: $100,000
- Final: $112,621
- Return: **+12.62%**
- Max Drawdown: **19.84%**

**What this means**: The strategy held one long position in USD/JPY through a favorable period. The large return came from the strong USD trend, but the 20% drawdown shows significant risk.

---

## Key Findings

### 1. Stop Loss is Critical
- 2% stop = disaster (constant whipsaw losses)
- 6-8% stop = workable (let trades breathe)
- ATR-based stops are better (adapt to volatility)

### 2. The Strategy is Marginal
- Doesn't capture big trends well
- Barely breaks even on real data
- Needs improvement before real money

### 3. Risk Management Saves You
- VaR shows you can lose 1-1.5% on bad days
- Circuit breakers prevent emotional decisions
- Position sizing prevents blowing up

### 4. Synthetic Data Has Limits
- Good for testing code
- Bad for predicting real performance
- Real markets have trends, fat tails, and regime changes

---

## Algorithm Validation

Before using any algorithm with real money, you MUST validate it properly.

### Step 1: Out-of-Sample Testing (Already Done)
- ✅ Test on data the algorithm never saw during development
- ✅ Our walk-forward analysis tested this

### Step 2: Monte Carlo Simulation
Run the strategy 1000+ times with:
- Randomized trade order
- Random starting points
- Different market conditions

```bash
# We can add this - let me know if you want it
uv run python scripts/monte_carlo_validation.py
```

### Step 3: Paper Trading (Minimum 1 Month)
```bash
# Run the paper trading demo
uv run python scripts/run_paper_trading.py
```

Watch for:
- Does win rate match backtest? (within 20%)
- Does drawdown stay within limits?
- Do fills happen at expected prices?

### Step 4: Stress Testing
Test what happens during:
- Flash crashes (2015 CHF event: 20% move in minutes)
- High volatility periods (COVID March 2020)
- Low liquidity (holidays, weekends)

### Step 5: Sensitivity Analysis
What happens if:
- Slippage is 2x worse than expected?
- Swap rates change (central banks adjust)?
- Your internet goes down for an hour?

---

## Next Steps for Real Money Testing

### Phase 1: Choose a Broker (2-4 weeks)

**Recommended for beginners**:
| Broker | Min Deposit | Leverage | API | Notes |
|--------|-------------|----------|-----|-------|
| OANDA | $0 | 50:1 | Yes | Great API, educational focus |
| IG | $250 | 30:1 (EU) | Yes | Well-regulated |
| Interactive Brokers | $0 | 50:1 | Yes | Professional-grade |

**What to look for**:
- Regulation (FCA, NFA, ASIC)
- API access for automation
- Low spreads (< 1 pip for majors)
- Demo account available

### Phase 2: Demo Account (4-8 weeks)

1. Open demo account with chosen broker
2. Modify our code to connect to broker API
3. Run strategy on demo for at least 30 days
4. Track all metrics and compare to backtest

**Code modification needed**:
```python
# Replace this:
from src.broker.simulator import BrokerSimulator

# With something like:
from oandapyV20 import API
client = API(access_token="your_demo_token")
```

### Phase 3: Micro Live Trading (4-8 weeks)

**Start TINY**:
- $500-1,000 maximum (money you can lose completely)
- 0.01 lot size only (1,000 units = ~$0.10/pip)
- One pair only (USD/JPY)

**Risk limits**:
| Limit | Value | Action |
|-------|-------|--------|
| Max loss per trade | $10 | Hard stop |
| Max daily loss | $25 | Stop trading for day |
| Max weekly loss | $75 | Stop trading for week |
| Max total loss | $250 | Stop and review strategy |

### Phase 4: Review and Iterate (Ongoing)

After each month:
1. Compare results to backtest expectations
2. Analyze losing trades - what went wrong?
3. Check if market regime changed
4. Adjust parameters if needed (carefully!)

### Budget Recommendation

| Capital | Recommendation |
|---------|----------------|
| < $500 | Stay on paper trading |
| $500-2,000 | Micro lots only, educational |
| $2,000-10,000 | Mini lots (0.1), still learning |
| > $10,000 | Consider professional advice |

### Timeline

```
Month 1-2:    Paper trading, learn the system
Month 3-4:    Demo account with broker API
Month 5-6:    Micro live ($500-1,000)
Month 7-12:   Review, iterate, possibly scale up
```

---

## Honest Assessment

### What This Strategy Does Well
- Captures swap income (small but consistent)
- Has defined risk management rules
- Code is well-tested (157 tests)

### What This Strategy Does Poorly
- Doesn't capture big trends
- Marginal profitability (roughly break-even)
- Needs manual monitoring

### Realistic Expectations

| Expectation | Reality |
|-------------|---------|
| "I'll get rich quick" | No. Even hedge funds struggle to beat 15%/year |
| "The backtest shows X% so I'll make X%" | No. Live results are typically 30-50% worse |
| "I can set it and forget it" | No. Markets change, strategies decay |
| "I'll learn a lot" | Yes! This is the real value |

### The #1 Rule

**Never trade with money you can't afford to lose.**

This is educational. The goal is learning, not profit. If you make money, great. If you lose money, you learned something valuable.

---

## Quick Reference Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Explore synthetic data
uv run python scripts/explore_data.py

# Run backtest
uv run python scripts/run_backtest.py

# Optimize parameters
uv run python scripts/run_optimization.py

# Test with real data
uv run python scripts/run_real_data.py

# Risk analysis
uv run python scripts/run_risk_analysis.py

# Paper trading simulation
uv run python scripts/run_paper_trading.py

# Open Jupyter notebooks
uv run jupyter notebook
```

---

## Final Thoughts

You've built a complete algorithmic trading system:
- Data generation and loading
- Strategy implementation
- Backtesting engine
- Parameter optimization
- Risk management
- Paper trading infrastructure

This is a **significant accomplishment**. Most people who try algo trading never get past "I have an idea."

The strategy itself is marginal - that's normal. Professional quants test hundreds of strategies to find one that works. The value is in the infrastructure and knowledge you've gained.

Next: Focus on understanding WHY trades win or lose. That's how you improve.

Good luck! 🎯
