# Enhancement Ideas

> Prioritized list of improvements to maximize returns and system robustness.

---

## High Impact — More Opportunities

### 1. Multi-Pair Expansion
Add EUR/JPY, NZD/JPY, MXN/JPY, ZAR/JPY. More carry pairs = more entry opportunities and better diversification. The infrastructure already supports it via `config.pairs` — just add them to `.env`.

### 2. Short-Side Carry Trades
Currently only goes LONG. Adding SHORT signals for negative-carry pairs during rate inversions would double the opportunity set. Example: shorting EUR/JPY when ECB cuts below BOJ.

### 3. Ensemble Strategies
Run V3 and V4 simultaneously with independent allocations. When both agree on a signal, size up. Disagreement = smaller size or skip. More signal diversity = more robust.

---

## High Impact — Better Entries

### 4. Cross-Asset Signals
VIX, US 10Y yield, DXY, and equity index futures are leading indicators for carry unwinds. Adding these as features to the bandit would improve its TAKE/SKIP decisions significantly.

### 5. Sentiment Data Integration
Add COT (Commitment of Traders) positioning data and OANDA's retail sentiment from their order book API. Carry trades unwind violently when positioning gets too crowded — this is the #1 risk for carry.

### 6. Time-of-Day Seasonality
Forex carry trades have known intraday patterns (London fix, Tokyo open). Add hour-of-day as a bandit feature and filter entries to optimal session windows.

### 7. Execution Optimization
Use limit orders instead of market orders for entries. Place them 0.5-1 ATR below current price for better fills. Cancel if not filled within N hours. Saves on slippage over time.

---

## Medium Impact — Better Risk Management

### 8. Continuous Volatility Scaling
The system detects regimes but the response is coarse (discrete buckets). A continuous inverse-realized-vol scaling function would size positions more precisely, reducing exposure smoothly as vol rises.

### 9. ~~Drawdown-Aware Rebalancing~~ ✅ Completed 2026-02-02
Quadratic decay position sizing: `factor = max(0.25, 1.0 - (dd/0.20)^2)`. Smoothly reduces new position size from 100% at 0% DD to 25% floor at 20% DD. Circuit breaker remains as backstop. Wired into runner via `PerformanceMonitor.get_snapshot().drawdown`. 6 new tests, all 490 passing.

### 10. Correlation-Based Hedging
When correlation between open positions exceeds 0.8, hedge the smaller one with a partial opposite. The `CorrelationMonitor` already tracks this but doesn't act on it beyond sizing adjustments.

---

## Medium Impact — Better Models

### 11. Deep RL for Position Management
The Gymnasium environment is built but unused beyond the bandit. A PPO or SAC agent trained on historical carry data could learn optimal entry/exit/sizing policies that the rule-based V3 strategy misses.

### 12. Macro Event Model
Train a separate model specifically on NFP, CPI, and rate decision outcomes to predict post-event direction for carry pairs. Go flat before events, then re-enter with conviction if the model has edge.

### 13. Walk-Forward Validation in Production
Run continuous out-of-sample testing by reserving 20% of live trades as a holdout. Compare the bandit's in-sample vs out-of-sample accuracy. Auto-deactivate if it degrades beyond a threshold.

---

## Lower Impact — Operational

### 14. Swap Rate Arbitrage
Currently uses static/default swap rates. Fetching real-time swap rates from OANDA and comparing across sessions could identify when swap rates temporarily spike, optimizing entry timing.

### 15. Multi-Timeframe Entry Refinement
V4 has the infrastructure but over-trades. Use weekly trend strictly as a directional filter (only trade in weekly trend direction) but keep daily V3 logic for actual entry timing.
